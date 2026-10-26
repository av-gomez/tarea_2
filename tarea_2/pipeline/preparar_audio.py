"""Arma clips de ~95 s concatenando enunciados de un mismo hablante de OpenSLR 73 (es_pe).

Cada clip = un hablante; la referencia es la transcripción humana oficial de OpenSLR.
Uso: python pipeline/preparar_audio.py   (desde tarea_2/)
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import soundfile as sf

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "datos/raw/audio/openslr73"
OUT = BASE / "datos/raw/audio"
SR = 16000
OBJETIVO_SEG = 95.0
SILENCIO = np.zeros(int(0.6 * SR), dtype=np.float32)


def cargar(genero):
    df = pd.read_csv(SRC / genero / "line_index.tsv", sep="\t", header=None, names=["utt", "texto"])
    df["hablante"] = df.utt.str.split("_").str[1]
    df["genero"] = genero
    return df


def leer(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def main(n_por_genero=10, semilla=42):
    random.seed(semilla)
    filas = []
    k = 1
    for genero in ["female", "male"]:
        df = cargar(genero)
        conteo = df.hablante.value_counts()
        hablantes = sorted(conteo[conteo >= 25].index)
        random.shuffle(hablantes)
        for h in hablantes[:n_por_genero]:
            sub = df[df.hablante == h].sort_values("utt")
            partes, textos, utts, dur = [], [], [], 0.0
            for _, r in sub.iterrows():
                x = leer(SRC / genero / f"{r.utt}.wav")
                partes += [x, SILENCIO]
                textos.append(r.texto.strip())
                utts.append(r.utt)
                dur += (len(x) + len(SILENCIO)) / SR
                if dur >= OBJETIVO_SEG:
                    break
            cid = f"clip_{k:03d}"
            sf.write(OUT / f"{cid}.wav", np.concatenate(partes), SR)
            filas.append({"id": cid, "archivo_audio": f"datos/raw/audio/{cid}.wav",
                          "fuente": "OpenSLR 73 es_pe (CC BY-SA 4.0)",
                          "fuente_url": f"https://www.openslr.org/73/ ({genero}, hablante {h})",
                          "genero": genero, "hablante_fuente": h, "n_enunciados": len(utts),
                          "enunciados": " ".join(utts), "duracion_seg": round(dur, 2),
                          "referencia_fuente": " ".join(textos)})
            k += 1
    pd.DataFrame(filas).to_csv(OUT / "catalogo_openslr.csv", index=False, encoding="utf-8")
    print(f"{len(filas)} clips, {sum(f['duracion_seg'] for f in filas)/60:.1f} min")


if __name__ == "__main__":
    main()
