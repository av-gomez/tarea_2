# 🎙️ Tarea 2 – Voces y documentos peruanos: del archivo escaneado al texto buscable

**Estudiante:** Amanda Gomez (@av-gomez) · **Dashboard:** https://av-gomez-tarea2.streamlit.app

## Pregunta
¿Pueden convertirse documentos escaneados y entrevistas en audio peruanos en un archivo de texto estructurado y buscable?

**Respuesta corta:** sí para impresos y para habla leída o de estudio. En esos casos el texto queda buscable con errores bajos. Con manuscritos coloniales y entrevistas grabadas en la calle, la salida automática sirve para buscar palabras clave, pero todavía necesita corrección humana.

## Pipeline
| Etapa | Herramienta | Dónde |
|---|---|---|
| OCR base | PaddleOCR 3.4.1 · PP-OCRv5 mobile det + `latin_PP-OCRv5_mobile_rec` (es) | `pipeline/ocr.py`, `01_extraccion.ipynb` |
| OCR alternativos | Tesseract 5 (`spa`, tessdata_best, LSTM) y EasyOCR 1.7.2 (es) | `02_evaluacion.ipynb` |
| ASR base | faster-whisper 1.2.1 `small`, CPU int8, `language=es`, VAD Silero, beam 5 | `pipeline/voz.py` |
| ASR alternativo | NVIDIA Parakeet-TDT 0.6B v3 (ONNX int8, sherpa-onnx 2.0) + VAD Silero | `pipeline/voz.py` |
| Diarización | pyannote-segmentation-3.0 + embeddings 3D-Speaker ERes2Net (sherpa-onnx) | `pipeline/voz.py` |
| Seudonimización | Diccionario de personas privadas (no versionado) + reglas DNI/teléfono/correo + revisión manual | `01_extraccion.ipynb` §3 |
| Extracción estructurada | Gemini `gemini-3.5-flash`, temperatura 0, salida JSON, `null` si el dato no aparece | `pipeline/llm.py`, `02_evaluacion.ipynb` |
| Dashboard | Streamlit 1.66 (solo lee los CSV) | `app.py` |

## Corpus, fuentes y licencias (extracción: 25–26 de octubre de 2026)
**A. Documentos: 25 imágenes** (8 impresos, 8 fotos con reflejos o inclinación, 9 manuscritos), en `datos/documentos.csv`:
- **Wikimedia Commons:** documentos de dominio público y fotos CC BY 2.0, CC BY-SA 3.0 y CC BY-SA 4.0. Entre ellos: recibo de la Peruvian Telephone Company (1895), diploma de la Sociedad Geográfica de Lima (1921), acción del Ferrocarril de la Oroya (1878), Acta de la Independencia (1821), partidas de bautismo coloniales, carta de Miguel Grau (1879) y placas de Lima.
- **Internet Archive:** impresos peruanos de 1791 a 1901 en dominio público (Mercurio Peruano, Gaceta del Gobierno de Lima de 1824, bando del virrey Abascal, El Peruano de 1901). Las digitalizaciones son de la John Carter Brown Library, UIUC y BHL. Estos ítems no tienen campo de derechos; se registran como dominio público por la fecha de la obra.
- Hubo **31 candidatos descartados** durante la recolección por baja resolución, falta de texto, datos personales (pasaportes, carnés, cartas privadas recientes), duplicados u origen no peruano. Se reemplazaron hasta completar 25.

**B. Audio: 26 clips, 50.5 min**, en `datos/transcripciones.csv`:
- **OpenSLR 73** (Google, *Crowdsourced high-quality Peruvian Spanish*, **CC BY-SA 4.0**): 20 clips de unos 95 s. Cada clip une enunciados de **un solo hablante** (10 mujeres y 10 hombres) con `pipeline/preparar_audio.py`.
- **Wikimedia Commons** (**CC BY-SA 4.0**): 6 grabaciones en español peruano. Son entrevistas en la calle o en ferias con ruido y música, una conferencia de prensa y una lectura de Wikipedia hablada. `clip_021` tiene **3 hablantes** y se usó para la diarización.
- Se descartaron 11 candidatos de audio: quechua, datos sensibles de personas privadas (orientación sexual, religión, opinión política), licencias NC/ND o dudosas, o audios de menos de 1 min.

La fuente, la licencia y la forma de recuperar cada archivo están en `datos/fuentes.csv`. Los originales (`datos/raw/`) y los modelos no se versionan. En `datos/ejemplos/` solo hay miniaturas de imágenes de dominio público o CC y 4 clips de OpenSLR en MP3; todos permiten redistribución y no contienen datos personales.

## Resultados de evaluación (los mismos 10 documentos y los mismos 10 clips para todos los modelos)
| Modalidad | Modelo | Métrica | Valor | Costo (CPU) |
|---|---|---|---|---|
| Documentos | **PaddleOCR PP-OCRv5 mobile (base)** | CER | **0.476** | 5.6 s/doc |
| Documentos | EasyOCR 1.7.2 | CER | 0.529 | 28.6 s/doc |
| Documentos | Tesseract 5 spa | CER | 0.672 | 2.5 s/doc |
| Audio | **faster-whisper small (base)** | WER | **0.032** | RTF 0.48 |
| Audio | Parakeet-TDT 0.6B v3 | WER | 0.039 | RTF 0.17 |

CER por tipo de documento con PaddleOCR: fotos 0.23 · impresos 0.48 · manuscritos 0.81. En las entrevistas de Commons, que no tienen referencia, el desacuerdo whisper/Parakeet sube de 0.04 a 0.36.

**Acuerdo LLM-humano:** **16/20 (80 %)** de extracciones completamente correctas (`datos/auditoria_llm.csv`).
- **Criterios:** tema y resumen fieles al texto; todo campo no nulo sustentado literalmente; sin omitir fechas, emisores ni montos explícitos; sin campos inventados.
- **Errores encontrados:**
  - Una fecha inventada (doc_010).
  - Ruido OCR copiado como si fuera fecha (doc_012).
  - El emisor de un parte citado tomado como emisor de la gaceta (doc_001).
  - Una cita parafraseada en lugar de textual (clip_026).
- **Quién revisó:** la revisión campo por campo la hizo el asistente de IA, comparando cada campo con el texto, y está marcada así en la columna `revisor`. Debe confirmarla la estudiante.

## Hallazgos
1. **El tipo de escritura importa más que la calidad de la foto.**
   - Las fotos de placas con letra de molde, a pesar de los reflejos y la inclinación, logran el mejor CER (0.23).
   - Los "impresos" evaluados quedan en 0.48 porque mezclan tipografía con datos manuscritos o caligrafía decorativa.
   - Los manuscritos coloniales rondan 0.8 con los tres OCR, porque ninguno está entrenado en paleografía.
2. **En habla leída el ASR ya es casi perfecto; el reto es el audio de campo.**
   - Whisper `small` y Parakeet cometen menos de 4 % de error en OpenSLR.
   - En las entrevistas de calle y feria el desacuerdo entre ambos se multiplica por 9 (de 0.04 a 0.36), y llega a 0.73 con audio a −40 dB.
   - Parakeet es ~3 veces más rápido con precisión casi igual, lo que lo vuelve atractivo para procesar en lote en CPU.

## Limitaciones técnicas
1. **Referencias y muestra de audio.**
   - Las referencias de audio son el texto humano de OpenSLR 73, es decir, el guion que lee cada hablante. Por eso los 10 clips de evaluación son lectura limpia.
   - Las entrevistas de Commons no tienen transcripción manual completa. En ellas solo se mide el desacuerdo entre modelos, que **no es WER**.
   - Las referencias de los documentos se transcribieron mirando la imagen. Las de los manuscritos coloniales pueden tener errores de paleografía.
2. **Cómputo en CPU.**
   - Todo corrió en una laptop sin GPU. Para que el tiempo alcanzara se usaron whisper `small` y PaddleOCR *mobile*, no `large-v3` ni los modelos *server*.
   - La diarización automática (umbral 0.6) sobresegmentó la entrevista en 19 hablantes. Por eso se fijó el número de hablantes en 3, según la descripción del archivo, y solo se diarizó `clip_021`.

## Ejecutar en local
```bash
cd tarea_2
python -m venv .venv && .venv\Scripts\activate      # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Para reproducir los notebooks (Python 3.11): `pip install -r requirements-pipeline.txt`. Luego descarga los datos según `datos/fuentes.csv`, ejecuta `python pipeline/preparar_audio.py` y abre `01_extraccion.ipynb` y `02_evaluacion.ipynb`. Para la extracción con Gemini se necesita un archivo `.env` con `GEMINI_API_KEY=...`, que no se versiona.

## Uso de IA
El proyecto se desarrolló con apoyo de Claude Code. La conversación completa está en `evidencias_ia/` (ver `evidencias_ia/README.md`). En `evidencias_ia/guia_prompting.md` hay lecciones para mejorar los prompts, con ejemplos de esta conversación.
