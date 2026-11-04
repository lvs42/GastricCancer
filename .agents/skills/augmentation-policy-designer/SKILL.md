---
name: augmentation-policy-designer
description: Diseña y audita políticas de data augmentation para parches histopatológicos H&E de GasHisSDB, evitando transformaciones que destruyan información diagnóstica o que introduzcan fuga de datos entre train y test. Usar al definir el pipeline de augmentation antes de entrenar, o al revisar si las augmentaciones usadas son apropiadas para tejido teñido con hematoxilina-eosina.
---

# Diseñador de política de augmentation (histopatología H&E)

Esta skill diseña y audita transformaciones de data augmentation para los parches de GasHisSDB, considerando qué es válido en imagen histopatológica teñida con H&E (hematoxilina-eosina) — un dominio con propiedades muy distintas de la fotografía natural, donde muchas augmentaciones "estándar" de librerías genéricas son inapropiadas o incluso contraproducentes.

## Principios del dominio histopatológico

- **El tejido no tiene una orientación canónica**: a diferencia de fotos naturales (donde un perro al revés sigue siendo un perro pero es una escena "rara"), un corte histológico no tiene arriba/abajo intrínseco. Esto hace que las rotaciones (incluidas rotaciones libres, no solo 90°) y los flips horizontal/vertical sean **augmentaciones válidas y recomendables por defecto**, al contrario que en muchas tareas de visión natural donde flip vertical se evita.
- **El color codifica información diagnóstica real**: la tinción H&E tiñe núcleos en tonos azul-violeta (hematoxilina) y citoplasma/estroma en tonos rosa (eosina). La intensidad y distribución de estos colores está relacionada con características morfológicas relevantes (densidad nuclear, atipia). Jitter de color agresivo o aleatorio sin restricción puede destruir esta señal o, peor, crear parches con apariencia de tinción fisiológicamente imposible.
- **Variabilidad de tinción entre laboratorios/lotes (stain variability) es un problema real y distinto del augmentation por color genérico**: en vez de `ColorJitter` estándar de torchvision/albumentations aplicado sin criterio, es preferible considerar augmentation específico de dominio como normalización/perturbación de tinción (p. ej. métodos tipo Macenko o Vahadane para normalización de tinción, o augmentation en el espacio de tinción H&E en vez de en RGB directo) si el TFM tiene margen para ello. Si se usa `ColorJitter` genérico, debe acotarse con parámetros conservadores y justificarse.
- **Los parches son pequeños (80-160px)**: transformaciones que recortan aún más (random crop agresivo) pueden eliminar la estructura relevante casi por completo dado el tamaño ya reducido. Si se usa crop, debe ser suave (p. ej. crop-and-resize con poco margen) y justificarse frente al riesgo de perder contexto morfológico.
- **Ruido y blur deben simular condiciones de escaneo reales** (variación de foco, ruido del escáner de WSI), no ruido genérico de fotografía — augmentation ligera de blur gaussiano y ruido gaussiano de baja intensidad es razonable; blur/ruido agresivos no.

## Checklist al diseñar o auditar una política de augmentation

1. **Geométricas — recomendadas por defecto**: flip horizontal, flip vertical, rotaciones en múltiplos de 90° y opcionalmente rotación libre con relleno por reflexión (no con negro/constante, que introduce bordes artificiales no realistas).
2. **Color — usar con cautela y límites acotados**: si se usa jitter de brillo/contraste/saturación, mantener rangos pequeños (p. ej. ±10-15%) y justificar por qué; preferir, si es viable, augmentation en espacio de tinción H&E sobre RGB genérico.
3. **Evitar por defecto salvo justificación explícita**: distorsiones elásticas agresivas, cutout/random erasing grandes (pueden eliminar la única región diagnóstica relevante en un parche ya pequeño), y cualquier augmentation que cambie la escala de forma que ya no sea consistente con el tamaño de parche declarado (80/120/160px).
4. **Nunca aplicar augmentation a validación/test**: verificar explícitamente en el código que las transformaciones de augmentation (aleatorias) solo están en el `DataLoader`/pipeline de entrenamiento, y que val/test usan únicamente preprocesado determinista (resize/normalización). Este es un error común y grave: aplicar augmentation aleatoria a test infla la varianza de la métrica reportada y puede ocultar fuga si además se reutiliza la misma semilla de forma inconsistente.
5. **Verificar el orden de operaciones respecto al split**: la augmentation debe aplicarse *después* de fijar el split train/val/test, nunca antes (aplicar augmentation antes de separar y luego splitear el conjunto aumentado es una fuga de datos clásica: versiones aumentadas del mismo parche original terminan en train y en test).
6. **Normalización de intensidad**: verificar qué estadísticas de normalización (media/std) se usan — si se usa transfer learning con pesos de ImageNet, debe discutirse si se usan las estadísticas de ImageNet (mismatch de dominio) o estadísticas calculadas sobre el propio dataset GasHisSDB (más correcto para este dominio, debe documentarse la elección).
7. **Documentación reproducible**: la política de augmentation debe estar versionada como código (no aplicada manualmente/ad-hoc) y debe listarse completa en la memoria (transformación, parámetros, probabilidad de aplicación) para que sea replicable.

## Procedimiento

1. Si se pide diseñar la política, proponer una lista concreta de transformaciones con parámetros razonables siguiendo los principios anteriores, distinguiendo claramente train vs val/test.
2. Si se pide auditar un pipeline existente (código de `Dataset`/`transforms`), recorrer la checklist y señalar cada punto que no se cumple, priorizando primero cualquier fuga entre train/test o augmentation aplicada a val/test.
3. Si se pide, redactar la sección de "Aumento de datos" de la memoria en español académico, listando cada transformación con su justificación ligada al dominio histopatológico (no solo "se aplicó augmentation estándar").

No implementes el pipeline de augmentation directamente en el código salvo que se solicite explícitamente — por defecto, esta skill diseña/audita y propone, y espera confirmación antes de escribir código de producción.
