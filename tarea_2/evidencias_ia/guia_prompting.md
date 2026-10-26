# Cómo mejorar mis prompts: lecciones de esta tarea

Esta guía usa **mensajes reales** que le escribí a Claude Code en esta tarea. Cada ejemplo muestra qué faltaba y cómo escribirlo mejor la próxima vez.

---

## Paso 1. Empieza con el contexto completo: objetivo, plazo y restricciones

**Lo que escribí:**
> "en esta caprtea vamos a acompletar esta tarea, dime que insumos requeires de mi."

**Qué faltó:** no dije que la entrega era **ese mismo día a las 23:59**. El asistente se enteró por el reloj del sistema cuando ya quedaba 1 h 30 min. Ese dato cambia todo el plan: modelos más ligeros, qué se prioriza y qué se deja para después.

**Mejor:**
> "Vamos a completar esta tarea en esta carpeta. **Se entrega HOY a las 23:59 (son las 21:00).** Mi usuario de GitHub es av-gomez y mi nombre es Amanda Gomez. No tengo GPU. Prioriza tener algo entregable a tiempo y luego mejóralo. Dime qué necesitas de mí."

✅ **Regla:** incluye siempre **fecha límite, recursos disponibles** (GPU, cuentas, llaves) y **qué es lo más importante**.

---

## Paso 2. Responde todas las preguntas, una por una

**Lo que escribí:**
> "nuevo repo . con ubicacnopn en esta capreta. av-gomez y soy amanda gomez 2. lo que puedas solucionar por tu cuenta sea bieno. 3 a 4 se especifico..."

**Qué faltó:** las respuestas venían numeradas, pero mezcladas en una sola línea y con errores de tipeo. Así es fácil malinterpretar algo, por ejemplo qué significaba "3 a 4".

**Mejor:**
> 1. Repo: nuevo, en esta carpeta. Usuario: av-gomez, nombre: Amanda Gomez.
> 2. Datos: búscalos tú en línea.
> 3. LLM: opción (a) Gemini.
> 4. No entendí qué necesitas en este punto. ¿Me lo explicas paso a paso?

✅ **Regla:** **una respuesta por línea, con su número**. Si no entiendes algo, dilo explícitamente: está bien pedir que te lo expliquen.

---

## Paso 3. Nunca pegues secretos en el chat

**Lo que escribí:**
> "AQ.Ab8... esta es la llave"

**Qué pasó:** la llave quedó en el historial de la conversación y en las evidencias. Hubo que ocultarla, y conviene **crear una nueva**.

**Mejor:** pega la llave **solo en el archivo `.env`**, con el Bloc de notas, y escribe en el chat:
> "Listo, ya guardé la llave en `.env`."

✅ **Regla:** contraseñas, llaves API y tokens **nunca van en el chat**. Van en archivos locales que no se suben a GitHub.

---

## Paso 4. Cuando algo falla, copia el error exacto

**Lo que escribí:**
> "los links no fucnionan ningunode los 2"

**Qué faltó:** no dije qué veía al abrir cada link: ¿una página de error 404?, ¿una pantalla para iniciar sesión?, ¿una página en blanco? El asistente tuvo que investigarlo.

**Mejor:**
> "Abrí https://av-gomez-tarea2.streamlit.app y me muestra 'You do not have access to this app'. El de GitHub dice '404 – Page not found'."

**Un buen ejemplo que sí escribí:** cuando pegué el error de Streamlit *"A subdomain can only contain a-z, 0-9, and -..."* y el texto que había escrito (`av-gomez-tarea2`), el problema se diagnosticó de inmediato.

✅ **Regla:** di **qué hiciste, qué esperabas y qué pasó**, y **copia el mensaje de error textual**. Una captura también sirve.

---

## Paso 5. Pide resultados verificables, no solo rapidez

**Lo que escribí:**
> "necesito que te apures" · "termina todo publica. ahora si cumple con tdoo"

**Qué faltó:** pedir rapidez no ayuda a decidir **qué sacrificar**. "Cumple con todo" no dice qué parte importa más.

**Mejor:**
> "Tengo 30 minutos. Prioriza en este orden: (1) links funcionando, (2) notebooks ejecutados, (3) README. Lo demás déjalo en una lista de pendientes. Al final dime qué quedó hecho y qué no."

✅ **Regla:** cuando el tiempo es poco, **ordena las prioridades** y pide un **resumen de lo hecho y lo pendiente**.

---

## Paso 6. Revisa y valida el trabajo de la IA

Varias cosas en esta tarea **requieren revisión humana**, y la IA lo advirtió:
- **Las referencias manuales** de los 10 documentos: las transcribió el asistente mirando las imágenes; debo corregirlas yo.
- **La auditoría de 20 extracciones del LLM:** la hizo el asistente; debo confirmar cada fila.
- **La seudonimización:** la IA encontró y corrigió **dos fugas de nombres reales** gracias a una búsqueda final. Una revisión humana habría servido como doble control.

**Buen prompt para esto:**
> "Antes de publicar, muéstrame una lista de todo lo que hiciste tú y yo debo revisar. Para cada cosa, dime qué archivo abrir y qué buscar."

✅ **Regla:** la IA acelera el trabajo, pero **la responsable del resultado soy yo**. Pide siempre qué debes verificar.

---

## Plantilla para la próxima tarea

```
CONTEXTO: Tarea [nombre], curso [X]. Se entrega [fecha y hora]. Hoy son las [hora].
OBJETIVO: [qué debe quedar listo, p. ej. repo + notebooks + link público].
RECURSOS: GitHub [usuario], [GPU sí/no], llaves en .env: [cuáles].
PRIORIDADES: 1) ... 2) ... 3) ...
RESTRICCIONES: no publicar datos personales; [otras].
QUÉ NECESITO DE TI: [p. ej. dime paso a paso lo que debo hacer yo].
AL TERMINAR: resume qué quedó hecho, qué falta y qué debo revisar yo.
```
