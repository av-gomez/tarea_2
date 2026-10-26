"""ASR (faster-whisper / Parakeet vía sherpa-onnx), diarización y exportación SRT/TXT.

Las salidas se cachean en datos/raw/audio/salidas/ para que los notebooks se re-ejecuten rápido.
"""
from pathlib import Path
import json
import time
import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parents[1]
SALIDAS = BASE / "datos/raw/audio/salidas"
MODELOS = BASE / "modelos/descargas"
SALIDAS.mkdir(parents=True, exist_ok=True)

_cache = {}


def _cargar_wav(path):
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == 16000, f"{path}: se espera 16 kHz"
    return x


def _json_path(path, etiqueta):
    return SALIDAS / f"{Path(path).stem}.{etiqueta}.json"


def whisper(path, size="small", vad=True, forzar=False):
    """faster-whisper en CPU int8, idioma fijado a español, VAD Silero activado."""
    etiqueta = f"whisper-{size}{'-vad' if vad else ''}"
    jp = _json_path(path, etiqueta)
    if jp.exists() and not forzar:
        return json.loads(jp.read_text(encoding="utf-8"))
    from faster_whisper import WhisperModel
    if etiqueta not in _cache:
        _cache[etiqueta] = WhisperModel(size, device="cpu", compute_type="int8")
    t0 = time.perf_counter()
    segs, info = _cache[etiqueta].transcribe(str(path), language="es", vad_filter=vad, beam_size=5,
                                             condition_on_previous_text=False)
    segs = [{"inicio": s.start, "fin": s.end, "texto": s.text.strip()} for s in segs]
    res = {"modelo": f"faster-whisper {size} (cpu int8, lang=es, VAD={'on' if vad else 'off'}, beam=5)",
           "segmentos": segs, "texto": " ".join(s["texto"] for s in segs),
           "segundos_proceso": time.perf_counter() - t0, "duracion_seg": info.duration}
    jp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def _parakeet():
    if "parakeet" not in _cache:
        import sherpa_onnx
        d = MODELOS / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
        _cache["parakeet"] = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
            model_type="nemo_transducer", num_threads=4)
    return _cache["parakeet"]


def _vad_trozos(x, max_seg=25.0):
    """Silero VAD (sherpa-onnx) -> trozos de voz <= max_seg s (Parakeet no acepta audios largos)."""
    import sherpa_onnx
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(MODELOS / "silero_vad.onnx")
    cfg.silero_vad.min_silence_duration = 0.3
    cfg.silero_vad.max_speech_duration = max_seg
    cfg.sample_rate = 16000
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=600)
    ventana = cfg.silero_vad.window_size
    for i in range(0, len(x) - ventana + 1, ventana):
        vad.accept_waveform(x[i:i + ventana])
    vad.flush()
    trozos = []
    while not vad.empty():
        trozos.append((vad.front.start / 16000, np.array(vad.front.samples, dtype=np.float32)))
        vad.pop()
    return trozos


def parakeet(path, forzar=False):
    """NVIDIA Parakeet-TDT 0.6B v3 (multilingüe, incluye español), ONNX int8 en CPU, con VAD."""
    jp = _json_path(path, "parakeet-tdt-0.6b-v3")
    if jp.exists() and not forzar:
        return json.loads(jp.read_text(encoding="utf-8"))
    rec = _parakeet()
    x = _cargar_wav(path)
    t0 = time.perf_counter()
    segs = []
    for ini, trozo in _vad_trozos(x):
        st = rec.create_stream()
        st.accept_waveform(16000, trozo)
        rec.decode_stream(st)
        segs.append({"inicio": ini, "fin": ini + len(trozo) / 16000, "texto": st.result.text.strip()})
    res = {"modelo": "Parakeet-TDT 0.6B v3 (sherpa-onnx int8, cpu, VAD Silero)", "segmentos": segs,
           "texto": " ".join(s["texto"] for s in segs if s["texto"]),
           "segundos_proceso": time.perf_counter() - t0, "duracion_seg": len(x) / 16000}
    jp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def diarizar(path, n_hablantes=-1, umbral=0.5, forzar=False):
    """pyannote-segmentation-3.0 + embeddings 3D-Speaker ERes2Net + clustering (sherpa-onnx)."""
    jp = _json_path(path, "diarizacion")
    if jp.exists() and not forzar:
        return json.loads(jp.read_text(encoding="utf-8"))
    import sherpa_onnx
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(MODELOS / "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"))),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(MODELOS / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=n_hablantes, threshold=umbral),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    x = _cargar_wav(path)
    turnos = [{"inicio": r.start, "fin": r.end, "hablante": int(r.speaker)}
              for r in sd.process(x).sort_by_start_time()]
    jp.write_text(json.dumps(turnos, indent=1), encoding="utf-8")
    return turnos


def asignar_hablantes(segmentos, turnos):
    """Etiqueta cada segmento ASR con el hablante de mayor solapamiento temporal."""
    mapa = {}
    for s in segmentos:
        solape = {}
        for t in turnos:
            o = min(s["fin"], t["fin"]) - max(s["inicio"], t["inicio"])
            if o > 0:
                solape[t["hablante"]] = solape.get(t["hablante"], 0) + o
        h = max(solape, key=solape.get) if solape else None
        if h is not None and h not in mapa:
            mapa[h] = len(mapa) + 1
        s["hablante"] = f"HABLANTE_{mapa[h]}" if h is not None else "HABLANTE_?"
    return segmentos


def texto_con_hablantes(segmentos):
    out, actual = [], None
    for s in segmentos:
        if s.get("hablante") != actual:
            actual = s.get("hablante")
            out.append(f"[{actual}]")
        out.append(s["texto"])
    return " ".join(out)


def _ts(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def exportar(id_, segmentos, carpeta=SALIDAS):
    srt = []
    for i, s in enumerate(segmentos, 1):
        pref = f"[{s['hablante']}] " if s.get("hablante") else ""
        srt.append(f"{i}\n{_ts(s['inicio'])} --> {_ts(s['fin'])}\n{pref}{s['texto']}\n")
    (carpeta / f"{id_}.srt").write_text("\n".join(srt), encoding="utf-8")
    (carpeta / f"{id_}.txt").write_text(texto_con_hablantes(segmentos) if segmentos and segmentos[0].get("hablante")
                                         else " ".join(s["texto"] for s in segmentos), encoding="utf-8")
