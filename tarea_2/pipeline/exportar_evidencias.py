"""Convierte el registro JSONL de la sesión de Claude Code en páginas PNG numeradas (evidencias_ia/).

Incluye, en orden cronológico: mensajes de la estudiante, respuestas de texto del asistente y una línea
por cada herramienta usada (nombre + descripción). Oculta la API key y el correo.
"""
import json
import re
import sys
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

SECRETOS = [(re.compile(r"AQ\.[A-Za-z0-9_\-]{20,}"), "[API KEY OCULTA]"),
            (re.compile(r"AIza[A-Za-z0-9_\-]{20,}"), "[API KEY OCULTA]"),
            (re.compile(r"[\w.]+@gmail\.com"), "[correo oculto]")]


def limpiar(t):
    t = re.sub(r"<system-reminder>.*?</system-reminder>", "", t, flags=re.S)
    t = re.sub(r"<task-notification>.*?</task-notification>", "", t, flags=re.S)
    t = re.sub(r"</?pasted_content[^>]*>", "", t)
    t = re.sub(r"<command-(name|message|args)>.*?</command-\1>", "", t, flags=re.S)
    t = re.sub(r"<local-command-(stdout|caveat)>.*?</local-command-\1>", "", t, flags=re.S)
    t = t.replace("<bash-input>", "$ ").replace("</bash-input>", "")
    t = re.sub(r"</?bash-(stdout|stderr)>", "", t)
    for pat, rep in SECRETOS:
        t = pat.sub(rep, t)
    t = re.sub(r"[🀀-🿿☀-➿️]", "", t)  # emojis (la fuente no los dibuja)
    return t.strip()


eventos = []
for linea in SRC.open(encoding="utf-8"):
    d = json.loads(linea)
    if d.get("type") == "attachment" and not d.get("isSidechain") and d["attachment"].get("type") == "queued_command":
        t = limpiar(str(d["attachment"].get("prompt", "")))
        if t:
            eventos.append(("ESTUDIANTE", d.get("timestamp", "")[:19].replace("T", " ") + " (mensaje enviado mientras Claude trabajaba)", t))
        continue
    if d.get("isSidechain") or d.get("type") not in ("user", "assistant"):
        continue
    m = d.get("message", {})
    if m.get("model") == "<synthetic>":
        continue
    cont = m.get("content")
    ts = d.get("timestamp", "")[:19].replace("T", " ")
    if d["type"] == "user":
        if isinstance(cont, str):
            t = limpiar(cont)
            if t and "SYSTEM NOTIFICATION" not in cont:
                eventos.append(("ESTUDIANTE", ts, t))
        elif isinstance(cont, list):
            for b in cont:
                if b.get("type") == "text":
                    t = limpiar(b["text"])
                    if t and not t.startswith("[Image"):
                        eventos.append(("ESTUDIANTE", ts, t))
    else:
        for b in cont or []:
            if b.get("type") == "text" and b["text"].strip():
                eventos.append(("CLAUDE", ts, limpiar(b["text"])))
            elif b.get("type") == "tool_use":
                inp = b.get("input", {})
                desc = inp.get("description") or inp.get("file_path") or inp.get("pattern") or ""
                eventos.append(("HERRAMIENTA", ts, f"{b['name']}: {limpiar(str(desc))}"))

# --- render ---
W, H, M = 1400, 1900, 50
F = "C:/Windows/Fonts/"
f_txt = ImageFont.truetype(F + "segoeui.ttf", 22)
f_bold = ImageFont.truetype(F + "segoeuib.ttf", 22)
f_tool = ImageFont.truetype(F + "consola.ttf", 18)
COL = {"ESTUDIANTE": "#1f5fbf", "CLAUDE": "#b4541a", "HERRAMIENTA": "#666666"}
LH = 30
ANCHO = 105

lineas = []  # (fuente, color, texto)
for quien, ts, t in eventos:
    if quien == "HERRAMIENTA":
        for w in textwrap.wrap("> herramienta " + t, 125) or [""]:
            lineas.append((f_tool, COL[quien], w))
        continue
    lineas.append((None, None, ""))
    lineas.append((f_bold, COL[quien], f"{'ESTUDIANTE' if quien == 'ESTUDIANTE' else 'CLAUDE CODE'}  ·  {ts} UTC"))
    for par in t.splitlines():
        for w in textwrap.wrap(par, ANCHO) or [""]:
            lineas.append((f_txt, "#111111", w))

por_pag = (H - 2 * M - 80) // LH
paginas = [lineas[i:i + por_pag] for i in range(0, len(lineas), por_pag)]
for n, pag in enumerate(paginas, 1):
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((M, 20), f"Tarea 2 · Conversación con Claude Code (exportada del registro de la sesión, no es captura de pantalla) · pág. {n}/{len(paginas)}",
           font=f_tool, fill="#888888")
    d.line((M, 50, W - M, 50), fill="#dddddd", width=2)
    y = M + 30
    for fuente, color, w in pag:
        if fuente:
            d.text((M, y), w, font=fuente, fill=color)
        y += LH
    im.save(OUT / f"{n:02d}.png", optimize=True)
    if any("API KEY OCULTA" in w for _, _, w in pag):
        print("llave oculta en página", n)
print(len(eventos), "eventos ->", len(paginas), "páginas")
