# Evidencias de uso de IA

- `01.png` … `NN.png`: la conversación completa con **Claude Code**, en orden cronológico. Las páginas se **generaron automáticamente a partir del registro de la sesión** que Claude Code guarda localmente (`~/.claude/projects/.../*.jsonl`) con el script `pipeline/exportar_evidencias.py`. **No son capturas de pantalla**: el contenido es el registro textual de la sesión.
  - Incluyen los mensajes de la estudiante (también los enviados mientras el asistente trabajaba), las respuestas del asistente y una línea por cada herramienta que usó (nombre y descripción del comando).
  - La API key de Gemini y el correo aparecen como `[API KEY OCULTA]` y `[correo oculto]`.
- `guia_prompting.md`: lecciones para mejorar mis prompts, con ejemplos reales de esta conversación.
