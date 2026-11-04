---
name: thesis-style-spanish-editor
description: Edita y revisa la redacción en español académico de la memoria del TFM (registro formal, terminología médica/técnica consistente, estructura convencional de TFM, citas, evitar anglicismos innecesarios). Usar al redactar o revisar cualquier sección de la memoria escrita en español, o al traducir/adaptar al español un hallazgo técnico obtenido en inglés (código, papers, nombres de métricas).
---

# Editor de estilo académico en español para el TFM

Esta skill edita y revisa texto en español académico para la memoria del TFM sobre clasificación binaria de cáncer gástrico con imágenes histopatológicas (GasHisSDB). No traduce mecánicamente ni reescribe por reescribir: ajusta registro, terminología y estructura a las convenciones de un TFM español, preservando la precisión técnica del contenido original (código, resultados, papers en inglés).

## Registro y estilo

- **Impersonal/pasiva refleja en vez de primera persona**: preferir "se entrenó el modelo con..." o "en este trabajo se propone..." sobre "entrené el modelo" o "yo propongo", salvo que la normativa de la universidad del usuario indique lo contrario (si no se conoce, usar el impersonal por defecto, que es el estándar más extendido en TFMs en español).
- **Evitar calcos sintácticos del inglés**: no traducir literalmente estructuras como "it is important to note that" → "es importante notar que" (preferir "cabe destacar que" / "conviene señalar que"); evitar gerundios de posterioridad ("entrenando el modelo, obteniendo un 90%...") típicos de traducción apresurada de inglés técnico.
- **Formalidad consistente**: evitar coloquialismos ("un montón de datos", "bastante bueno") y cuantificadores vagos sin respaldo numérico ("muchos casos", "la mayoría") cuando existe la cifra exacta disponible — sustituir por el dato preciso.
- **Frases largas de sujeto-verbo distante**: el español académico tolera frases más largas que el inglés, pero si una frase supera ~40-50 palabras o tiene más de dos subordinadas anidadas, proponer partirla — mejora la legibilidad para el tribunal.

## Terminología técnica y médica (consistencia obligatoria)

Mantener SIEMPRE el mismo término español para el mismo concepto a lo largo de toda la memoria (un error muy común es alternar entre sinónimos, lo que un tribunal penaliza como falta de precisión). Términos de referencia para este TFM:

| Inglés | Español preferido | Evitar |
|---|---|---|
| deep learning | aprendizaje profundo | "deep learning" sin cursiva/sin traducir en cuerpo de texto corrido |
| dataset | conjunto de datos | mezclar "dataset" y "conjunto de datos" indistintamente |
| overfitting / underfitting | sobreajuste / infraajuste | dejar sin traducir salvo primera mención con el término en inglés entre paréntesis |
| patch | parche | "mancha", "recorte" (inconsistente) |
| whole slide image (WSI) | imagen de portaobjetos completa (WSI) | introducir el acrónimo la primera vez y usarlo después consistentemente |
| label / ground truth | etiqueta / verdad de referencia | "ground truth" sin traducir en el cuerpo del texto |
| accuracy | exactitud | "precisión" (ojo: "precisión" en español es *precision*/PPV, no *accuracy* — este es el falso amigo más frecuente y grave en TFMs de ML en español, verificar siempre) |
| recall / sensitivity | sensibilidad / exhaustividad | usar "sensibilidad" en contexto clínico (más natural para tribunal médico) y mantenerlo consistente |
| precision (métrica) | precisión / valor predictivo positivo | no confundir con "accuracy" (ver arriba) |
| loss function | función de pérdida | "función de coste" (aceptable pero elegir una y mantenerla) |
| learning rate | tasa de aprendizaje | dejar el término en inglés sin traducir de forma inconsistente |
| batch size | tamaño de lote | "batch size" sin traducir en cuerpo de texto |
| epoch | época | "iteración" (una época no es una iteración — error conceptual, no solo terminológico) |
| data augmentation | aumento de datos | "aumentación" (no es una palabra estándar del español) |
| data leakage | fuga de datos | dejar sin traducir |
| checkpoint | punto de control / checkpoint | aceptar el anglicismo si está extendido en el área, pero mantenerlo consistente |
| benign/malignant, normal/abnormal | según la etiqueta real del dataset (Normal/Abnormal) | no traducir libremente los nombres de clase del dataset — deben coincidir exactamente con la nomenclatura de GasHisSDB al referirse a las carpetas/etiquetas |

**Regla de "precisión" vs "exactitud" — la más importante de esta tabla**: verificar en cada aparición de estos términos en la memoria que "precisión" se usa solo para *precision* (PPV) y "exactitud" para *accuracy*. Esta confusión es el error terminológico más frecuente y más grave en TFMs de ML en español porque invierte el significado de una métrica reportada.

## Estructura convencional de TFM (a verificar si se pide revisión de estructura)

Orden habitual esperado por tribunales en España/Latinoamérica: Resumen/Abstract → Introducción (motivación clínica + objetivos) → Estado del arte → Metodología (dataset, preprocesado, arquitectura, protocolo experimental) → Resultados → Discusión → Conclusiones y trabajo futuro → Bibliografía → Anexos. Si se revisa una sección, verificar que no mezcla contenido de "Resultados" (datos objetivos) con "Discusión" (interpretación) — es una confusión común que resta claridad.

## Citas y referencias

- Verificar formato de citas consistente en toda la memoria (numérico [1] o autor-año, según normativa de la universidad — si no se especifica, mantener el que ya use el resto del documento, no mezclar).
- Al citar el paper original de GasHisSDB o trabajos relacionados, verificar que el nombre del dataset y los autores se citan de forma consistente en todas las apariciones.

## Procedimiento

1. Al editar un fragmento, primero identificar y corregir errores terminológicos graves (especialmente precisión/exactitud), luego registro/estilo, luego fluidez.
2. No alterar cifras, resultados ni afirmaciones técnicas del contenido — esta skill edita forma, no contenido; si detecta una afirmación técnica que parece incorrecta (no solo mal redactada), señalarlo aparte y remitir a la skill de contenido correspondiente (`binary-classification-metrics-medical`, `experiment-design-reviewer`, etc.) en vez de "corregirla" silenciosamente.
3. Al revisar un capítulo completo, producir primero un listado de inconsistencias terminológicas detectadas (mismo concepto con distinto término) antes de reescribir, para que el usuario decida cuál término fijar como estándar en todo el documento.
4. Preferir ediciones mínimas localizadas sobre reescrituras completas, salvo que se pida explícitamente una reescritura de fondo.
