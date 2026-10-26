"""OCR con PaddleOCR (PP-OCRv5, lang=es) y Tesseract (spa) como alternativa. Resultados cacheados."""
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parents[1]
SALIDAS = BASE / "datos/raw/documentos/ocr_crudo"
RECORTES = BASE / "datos/raw/documentos/recortes"
SALIDAS.mkdir(parents=True, exist_ok=True)
RECORTES.mkdir(parents=True, exist_ok=True)
TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PADDLE_CFG = dict(lang="es", use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False, text_det_limit_side_len=1600)
_ocr = {}


def preprocesar(path, max_lado=1600):
    """Preprocesamiento común: corrige orientación EXIF, RGB, reduce a max_lado px."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((max_lado, max_lado))
    return im


def paddle(path, forzar=False):
    jp = SALIDAS / f"{Path(path).stem}.paddle.json"
    if jp.exists() and not forzar:
        return json.loads(jp.read_text(encoding="utf-8"))
    from paddleocr import PaddleOCR
    import paddleocr
    if "p" not in _ocr:
        _ocr["p"] = PaddleOCR(**PADDLE_CFG)
    im = preprocesar(path)
    t0 = time.perf_counter()
    r = _ocr["p"].predict(np.array(im)[:, :, ::-1])[0]
    seg = time.perf_counter() - t0
    textos, scores, cajas = list(r["rec_texts"]), [float(s) for s in r["rec_scores"]], r["rec_polys"]
    # recortes de las 3 primeras líneas como respaldo visual
    for i, poly in enumerate(cajas[:3]):
        xs, ys = poly[:, 0], poly[:, 1]
        im.crop((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))).save(
            RECORTES / f"{Path(path).stem}_linea{i + 1}.png")
    res = {"modelo": f"PaddleOCR {paddleocr.__version__} PP-OCRv5 (lang=es, max_lado=1600, sin unwarping)",
           "lineas": textos, "scores": scores, "texto": "\n".join(textos), "segundos_proceso": seg}
    jp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def easyocr(path, forzar=False):
    """OCR alternativo: EasyOCR (detector CRAFT + reconocedor CRNN, idioma es), CPU."""
    jp = SALIDAS / f"{Path(path).stem}.easyocr.json"
    if jp.exists() and not forzar:
        return json.loads(jp.read_text(encoding="utf-8"))
    import easyocr as eo
    if "e" not in _ocr:
        _ocr["e"] = eo.Reader(["es"], gpu=False, verbose=False)
    im = preprocesar(path)
    t0 = time.perf_counter()
    r = _ocr["e"].readtext(np.array(im), paragraph=False)
    seg = time.perf_counter() - t0
    # orden de lectura: por fila (y del centro, en bandas de 15 px) y luego x
    r = sorted(r, key=lambda d: (round(np.mean([p[1] for p in d[0]]) / 15), d[0][0][0]))
    lineas = [d[1] for d in r]
    res = {"modelo": f"EasyOCR {eo.__version__} (lang=es, cpu, max_lado=1600)", "lineas": lineas,
           "scores": [float(d[2]) for d in r], "texto": "\n".join(lineas), "segundos_proceso": seg}
    jp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def tesseract(path, forzar=False):
    jp = SALIDAS / f"{Path(path).stem}.tesseract.json"
    if jp.exists() and not forzar:
        return json.loads(jp.read_text(encoding="utf-8"))
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT
    im = preprocesar(path)
    t0 = time.perf_counter()
    txt = pytesseract.image_to_string(im, lang="spa", config="--oem 1 --psm 3")
    seg = time.perf_counter() - t0
    lineas = [l.strip() for l in txt.splitlines() if l.strip()]
    res = {"modelo": f"Tesseract {pytesseract.get_tesseract_version()} (spa, oem 1, psm 3)",
           "lineas": lineas, "texto": "\n".join(lineas), "segundos_proceso": seg}
    jp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res
