"""Extracción estructurada (JSON) con Gemini a partir de textos OCR/ASR ya seudonimizados."""
from pathlib import Path
import json
import os
import time

BASE = Path(__file__).resolve().parents[1]
CACHE = BASE / "datos/raw/llm"
CACHE.mkdir(parents=True, exist_ok=True)
MODELO = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

PROMPT = """Eres un asistente de archivística. Extrae información SOLO del texto dado (puede tener errores de {origen}).
Devuelve exclusivamente un objeto JSON con estas claves:
- "modalidad": "{modalidad}"
- "tema": tema principal en 3 a 8 palabras
- "resumen": 1 o 2 oraciones en español
{extra}
Reglas: si un dato no aparece explícitamente en el texto usa null. No inventes ni completes datos.
No agregues claves distintas a las indicadas.

TEXTO:
\"\"\"{texto}\"\"\""""

EXTRA_DOC = """- "tipo_documento": p.ej. carta, acta, decreto, recibo, periódico, aviso; null si no es claro
- "emisor": persona o institución que emite el documento, o null
- "fecha": fecha tal como aparece en el texto, o null
- "lugar": lugar mencionado como origen, o null
- "monto": cantidad de dinero con su moneda tal como aparece, o null"""
EXTRA_AUD = """- "numero_hablantes": entero, solo si se deduce de marcas [HABLANTE_n] del texto; si no hay marcas, null
- "citas_clave": lista de 1 a 2 frases textuales copiadas del texto (o [] si no hay)
- "lugares_mencionados": lista de lugares mencionados (o [])"""


def _cliente():
    from dotenv import load_dotenv
    from google import genai
    load_dotenv(BASE.parent / ".env")
    load_dotenv(BASE / ".env")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def extraer(id_, texto, modalidad, forzar=False):
    jp = CACHE / f"{id_}.json"
    if jp.exists() and not forzar:
        return json.loads(jp.read_text(encoding="utf-8"))
    from google.genai import types
    cli = _cliente()
    doc = modalidad == "documento"
    prompt = PROMPT.format(origen="OCR" if doc else "reconocimiento de voz", modalidad=modalidad,
                           extra=EXTRA_DOC if doc else EXTRA_AUD, texto=texto[:12000])
    for intento in range(4):
        try:
            r = cli.models.generate_content(model=MODELO, contents=prompt, config=types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json"))
            campos = json.loads(r.text)
            break
        except Exception as e:  # cuota o JSON inválido: reintento con espera
            if intento == 3:
                raise
            time.sleep(5 * (intento + 1))
    res = {"id": id_, "modalidad": modalidad, "campos": campos, "modelo_llm": f"{MODELO} (temperature=0, JSON)",
           "fecha": time.strftime("%Y-%m-%d")}
    jp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res
