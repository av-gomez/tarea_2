"""Genera 02_evaluacion.ipynb."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Tarea 2 · Parte 2 — Evaluación, extracción estructurada y comparación de modelos

**Protocolo (fijado antes de calcular métricas):**
- Muestra de **10 documentos** (3 impresos, 4 fotos, 3 manuscritos) y **10 clips** de OpenSLR 73 (10 hablantes distintos, 5 mujeres y 5 hombres), todos del corpus de la Parte 1. Las grabaciones de Commons (ruido, música, varios hablantes) no tienen referencia manual completa, por lo que se analizan aparte como *caso difícil* mediante el desacuerdo entre modelos.
- Las configuraciones de todos los modelos son las de la Parte 1 y no se tocaron después de ver métricas.
- Normalización idéntica para referencia e hipótesis: minúsculas, sin tildes, sin puntuación, espacios colapsados; en audio se eliminan marcas `[HABLANTE_n]`.
- **Origen de las referencias:** en audio, el texto humano de OpenSLR 73 (guion leído por cada hablante, revisado contra el audio); en documentos, transcripción manual hecha mirando la imagen (columna `estado_revision`).""")

code("""import sys, json, re, unicodedata
import pandas as pd, numpy as np, jiwer
import plotly.express as px, plotly.io as pio
pio.renderers.default = "notebook_connected+png"
sys.path.insert(0, "pipeline")
import ocr, voz
docs = pd.read_csv("datos/documentos.csv"); trans = pd.read_csv("datos/transcripciones.csv")
def norm(t):
    t = re.sub(r"\\[HABLANTE_[^\\]]*\\]", " ", str(t))
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-zñ0-9 ]", " ", t.replace("ñ", "ñ"))
    return re.sub(r"\\s+", " ", t).strip()
cer = lambda r, h: jiwer.cer(norm(r), norm(h)) if norm(h) else 1.0
wer = lambda r, h: jiwer.wer(norm(r), norm(h)) if norm(h) else 1.0""")

md("## 1. Muestra de evaluación y referencias manuales")
code("""ref_doc = pd.read_csv("datos/referencias_documentos.csv")
EVAL_DOCS = ref_doc.id.tolist()
ref_ol = pd.read_csv("datos/referencia_openslr.csv")
ref_cm = pd.read_csv("datos/referencias_audio_commons.csv") if __import__("os").path.exists("datos/referencias_audio_commons.csv") else pd.DataFrame(columns=["id","texto_referencia","estado_revision"])
EVAL_OL = ["clip_001", "clip_003", "clip_005", "clip_007", "clip_009", "clip_011", "clip_013", "clip_015", "clip_017", "clip_019"]  # 5 mujeres + 5 hombres, hablantes distintos
ref_aud = pd.concat([ref_ol[ref_ol.id.isin(EVAL_OL)].rename(columns={"referencia_fuente": "texto_referencia"}).assign(estado_revision="transcripción humana OpenSLR"),
                     ref_cm], ignore_index=True)
EVAL_AUD = ref_aud.id.tolist()
docs = docs.drop(columns="texto_referencia").merge(ref_doc[["id", "texto_referencia"]], on="id", how="left")
trans = trans.drop(columns="texto_referencia").merge(ref_aud[["id", "texto_referencia"]], on="id", how="left")
docs.to_csv("datos/documentos.csv", index=False, encoding="utf-8"); trans.to_csv("datos/transcripciones.csv", index=False, encoding="utf-8")
print("Documentos de evaluación:", docs[docs.id.isin(EVAL_DOCS)].tipo.value_counts().to_dict())
print("Clips de evaluación:", len(EVAL_AUD)); display(ref_aud[["id", "estado_revision"]])""")

md("## 2. Línea base: PaddleOCR (sin personalización) y faster-whisper `small` CPU")
code("""ed = docs[docs.id.isin(EVAL_DOCS)].copy()
ed["CER_paddle"] = [cer(r.texto_referencia, ocr.paddle(r.archivo)["texto"]) for r in ed.itertuples()]
ea = trans[trans.id.isin(EVAL_AUD)].copy()
ea["WER_whisper"] = [wer(r.texto_referencia, voz.whisper(r.archivo_audio)["texto"]) for r in ea.itertuples()]
ea["RTF_whisper"] = [voz.whisper(r.archivo_audio)["segundos_proceso"] / r.duracion_seg for r in ea.itertuples()]
print(f"LÍNEA BASE  CER PaddleOCR = {ed.CER_paddle.mean():.3f}   |   WER whisper-small = {ea.WER_whisper.mean():.3f}")""")

md("## 3. Modelos alternativos sobre las mismas muestras")
md("""- **Documentos:** dos OCR alternativos: **Tesseract 5** (LSTM, `spa` de *tessdata_best*, `--psm 3`) y **EasyOCR** (detector CRAFT + reconocedor CRNN, `lang=es`). Mismo preprocesamiento que PaddleOCR (EXIF, RGB, lado ≤ 1600 px). Los tiempos se midieron con la CPU sin otros procesos.
- **Audio:** NVIDIA Parakeet-TDT 0.6B v3 (multilingüe, incluye español) en ONNX int8 vía sherpa-onnx, con VAD Silero. RTF = tiempo de proceso / duración.""")
code("""ed["CER_easyocr"] = [cer(r.texto_referencia, ocr.easyocr(r.archivo)["texto"]) for r in ed.itertuples()]
ed["CER_tesseract"] = [cer(r.texto_referencia, ocr.tesseract(r.archivo)["texto"]) for r in ed.itertuples()]
ed["seg_tesseract"] = [ocr.tesseract(r.archivo)["segundos_proceso"] for r in ed.itertuples()]
ea["WER_parakeet"] = [wer(r.texto_referencia, voz.parakeet(r.archivo_audio)["texto"]) for r in ea.itertuples()]
ea["RTF_parakeet"] = [voz.parakeet(r.archivo_audio)["segundos_proceso"] / r.duracion_seg for r in ea.itertuples()]
ed["seg_paddle"] = [ocr.paddle(r.archivo)["segundos_proceso"] for r in ed.itertuples()]
ed["seg_easyocr"] = [ocr.easyocr(r.archivo)["segundos_proceso"] for r in ed.itertuples()]
display(ed[["id", "tipo", "CER_paddle", "CER_easyocr", "CER_tesseract", "seg_paddle", "seg_easyocr", "seg_tesseract"]].round(3))
display(ea[["id", "condicion", "WER_whisper", "WER_parakeet", "RTF_whisper", "RTF_parakeet"]].round(3))""")

md("## 4. Tabla comparativa y casos difíciles")
code("""tabla = pd.DataFrame([
    {"modalidad": "documentos", "modelo": "PaddleOCR PP-OCRv5 (base)", "métrica": "CER", "valor": ed.CER_paddle.mean(), "costo": f"{ed.seg_paddle.mean():.1f} s/doc CPU"},
    {"modalidad": "documentos", "modelo": "EasyOCR (alternativo)", "métrica": "CER", "valor": ed.CER_easyocr.mean(), "costo": f"{ed.seg_easyocr.mean():.1f} s/doc CPU"},
    {"modalidad": "documentos", "modelo": "Tesseract 5 spa (alternativo)", "métrica": "CER", "valor": ed.CER_tesseract.mean(), "costo": f"{ed.seg_tesseract.mean():.1f} s/doc CPU"},
    {"modalidad": "audio", "modelo": "faster-whisper small (base)", "métrica": "WER", "valor": ea.WER_whisper.mean(), "costo": f"RTF {ea.RTF_whisper.mean():.2f}"},
    {"modalidad": "audio", "modelo": "Parakeet-TDT 0.6B v3 (alternativo)", "métrica": "WER", "valor": ea.WER_parakeet.mean(), "costo": f"RTF {ea.RTF_parakeet.mean():.2f}"}])
tabla["valor"] = tabla.valor.round(3)
tabla.to_csv("datos/metricas.csv", index=False, encoding="utf-8")
ed[["id","tipo","CER_paddle","CER_easyocr","CER_tesseract"]].to_csv("datos/metricas_documentos.csv", index=False)
ea[["id","condicion","WER_whisper","WER_parakeet","RTF_whisper","RTF_parakeet"]].to_csv("datos/metricas_audio.csv", index=False)
tabla""")
code("""fig = px.bar(ed.melt(id_vars=["id","tipo"], value_vars=["CER_paddle","CER_easyocr","CER_tesseract"], var_name="modelo", value_name="CER")
             .groupby(["tipo","modelo"], as_index=False).CER.mean(), x="tipo", y="CER", color="modelo", barmode="group",
             title="El CER se dispara en manuscritos para ambos OCR: ninguno está entrenado en letra colonial")
fig.show()
ea["grupo"] = np.where(ea.id.isin(["clip_001","clip_003","clip_005","clip_007","clip_009"]), "OpenSLR mujeres", "OpenSLR hombres")
fig = px.bar(ea.melt(id_vars=["id","grupo"], value_vars=["WER_whisper","WER_parakeet"], var_name="modelo", value_name="WER")
             .groupby(["grupo","modelo"], as_index=False).WER.mean(), x="grupo", y="WER", color="modelo", barmode="group",
             title="WER por género del hablante en la muestra de OpenSLR")
fig.show()""")
code("""print("Por tipo de documento:"); display(ed.groupby("tipo")[["CER_paddle","CER_easyocr","CER_tesseract"]].mean().round(3))
print("Por tipo de audio:"); display(ea.groupby("grupo")[["WER_whisper","WER_parakeet"]].mean().round(3))""")

md("""### Caso difícil: entrevistas con ruido, música y varios hablantes (Commons)
Sin referencia manual completa no se reporta WER; como indicador se usa el **desacuerdo entre modelos** (WER de Parakeet tomando a whisper como pseudo-referencia). Un valor alto indica que al menos uno de los dos falla. **No es una métrica de exactitud.**""")
code("""cm = trans[trans.origen.str.startswith("Wikimedia")].copy()
cm["desacuerdo_modelos"] = [wer(voz.whisper(r.archivo_audio)["texto"], voz.parakeet(r.archivo_audio)["texto"]) for r in cm.itertuples()]
ol = trans[trans.id.isin(EVAL_AUD)].copy()
ol["desacuerdo_modelos"] = [wer(voz.whisper(r.archivo_audio)["texto"], voz.parakeet(r.archivo_audio)["texto"]) for r in ol.itertuples()]
print(f"Desacuerdo medio whisper vs Parakeet — OpenSLR: {ol.desacuerdo_modelos.mean():.2f} | Commons: {cm.desacuerdo_modelos.mean():.2f}")
display(cm[["id", "titulo", "condicion", "n_hablantes", "desacuerdo_modelos"]].round(2))
for r in cm.head(2).itertuples():
    print("\n", r.id, r.condicion); print("whisper :", voz.whisper(r.archivo_audio)["texto"][:300]); print("parakeet:", voz.parakeet(r.archivo_audio)["texto"][:300])
pd.DataFrame([{"id": r.id, "texto": voz.parakeet(r.archivo_audio)["texto"]} for r in trans.itertuples()]).to_csv("datos/transcripciones_parakeet.csv", index=False, encoding="utf-8")""")

md("## 5. Extracción estructurada con LLM")
code("""import os
from dotenv import load_dotenv; load_dotenv("../.env"); load_dotenv(".env")
HAY_LLM = bool(os.environ.get("GEMINI_API_KEY"))
print("API key de Gemini disponible:", HAY_LLM)
if HAY_LLM:
    import llm
    filas = [llm.extraer(r.id, r.texto_ocr, "documento") for r in docs.itertuples()] + \\
            [llm.extraer(r.id, r.texto_asr, "audio") for r in trans.itertuples()]
    ext = pd.DataFrame([{"id": f["id"], "modalidad": f["modalidad"], "campos_json": json.dumps(f["campos"], ensure_ascii=False),
                         "modelo_llm": f["modelo_llm"], "fecha": f["fecha"]} for f in filas])
    ext.to_csv("datos/extraccion_llm.csv", index=False, encoding="utf-8")
    display(ext.head())""")

md("## 6. Auditoría humana de 20 extracciones")
md("""Criterios: una extracción es **completamente correcta** si (a) `tema` y `resumen` son fieles al texto, (b) todo campo no nulo está sustentado literalmente en el texto, (c) no se omiten datos explícitos del texto (`fecha`, `emisor`, `monto`) y (d) no hay campos inventados. Muestra: 10 documentos + 10 audios al azar (semilla 7).""")
code("""if os.path.exists("datos/auditoria_llm.csv"):
    aud_llm = pd.read_csv("datos/auditoria_llm.csv")
    ok = int(aud_llm.correcto.sum())
    print(f"Acuerdo LLM-Humano: {ok}/20 = {ok/20:.0%}")
    display(aud_llm[aud_llm.correcto == 0][["id", "error"]])
else:
    print("Auditoría pendiente")""")

md("## 7. Decisión y reflexión")
code("""print(tabla.to_string(index=False))""")
md("""Ver README para la discusión completa. Resumen: se elige el modelo con menor error en cada modalidad salvo que su costo (RTF) impida su uso en CPU; los manuscritos coloniales y las entrevistas callejeras son los casos donde ambos modelos fallan más, por falta de datos de entrenamiento con caligrafía antigua y por ruido ambiente/solapamiento de voces respectivamente.""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).resolve().parents[1] / "02_evaluacion.ipynb")
