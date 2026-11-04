---
name: baseline-replication-auditor
description: Verifica que las réplicas de baselines/benchmarks del TFM (papers originales de GasHisSDB, arquitecturas de referencia, resultados citados) coincidan realmente en metodología con lo implementado en el código, y que las comparaciones reportadas en la memoria sean justas. Usar al implementar un baseline citado de un paper, al comparar resultados propios contra la literatura, o al revisar una tabla comparativa antes de incluirla en la memoria.
---

# Auditor de replicación de baselines

Esta skill audita si un baseline o resultado de referencia citado en el TFM (típicamente del paper original de GasHisSDB — Hu et al., "GasHisSDB: A new gastric histopathology image dataset for computer aided diagnosis of gastric cancer", 2022 — u otros papers que usan este dataset) está siendo replicado de forma metodológicamente honesta, y si las comparaciones que el TFM hace contra esos resultados son válidas.

## Por qué importa en este proyecto

Es habitual en TFMs comparar "mi modelo obtiene X% accuracy" contra "el paper Y reporta Z% accuracy" sin verificar que ambos midan lo mismo. Con GasHisSDB esto es especialmente delicado porque:
- Distintos papers usan distintos tamaños de parche (80/120/160) y a veces no lo especifican con claridad en el resumen de resultados.
- Los splits train/test no siempre son idénticos ni reproducibles entre papers (ver `dataset-audit-histopathology` sobre el riesgo de fuga a nivel de WSI — muchos papers publicados pueden tener ese mismo problema, lo que infla sus cifras de referencia).
- Las métricas reportadas varían (accuracy vs F1 vs AUC vs sensibilidad/especificidad) y comparar accuracy contra AUC, o macro-F1 contra accuracy, no es una comparación válida.
- El preprocesado (normalización, augmentación, resize) rara vez está descrito con precisión suficiente para una réplica exacta.

## Qué comprobar al auditar una réplica de baseline

1. **Identidad de la tarea**: confirmar que el baseline citado resuelve exactamente el mismo problema (clasificación binaria Normal/Abnormal a nivel de parche) y no una variante (p. ej. clasificación multiclase por subtipo histológico, o segmentación) que se esté comparando de forma inválida.
2. **Tamaño de parche usado**: extraer explícitamente qué tamaño (80/120/160) usa el baseline y verificar que el código del TFM entrena/evalúa sobre el mismo tamaño antes de comparar cifras.
3. **Definición del split**: identificar si el paper de referencia especifica proporción train/val/test, si el split es a nivel de parche o de WSI, y si usa validación cruzada. Si el TFM usa un split distinto, la comparación debe marcarse como aproximada, no como equivalente.
4. **Arquitectura y pesos**: verificar que la arquitectura implementada en el código (capas, nº de parámetros, si usa transfer learning con pesos de ImageNet u otro) coincide con lo descrito en el paper — no solo el nombre genérico (p. ej. "ResNet-50" puede referirse a variantes distintas con distinto preprocesado).
5. **Hiperparámetros de entrenamiento**: optimizador, learning rate, nº de épocas, tamaño de batch, augmentación, criterio de parada temprana. Señalar cualquier hiperparámetro no reportado por el paper original que el TFM haya tenido que asumir, y documentarlo como supuesto explícito.
6. **Métrica comparada**: confirmar que se compara la misma métrica calculada de la misma forma (p. ej. accuracy balanceada vs accuracy simple ante desbalance de clases — ver `binary-classification-metrics-medical`).
7. **Significancia y variabilidad**: si el TFM reporta una sola ejecución contra un promedio de varias corridas del paper (o viceversa), señalarlo — una comparación punto a punto sin intervalos de confianza o desviación estándar es débil.

## Procedimiento al usar esta skill

1. Pide o localiza la fuente exacta del baseline (referencia bibliográfica, tabla, o número citado) y el código/resultado propio que se compara.
2. Construye una tabla de "condiciones de comparación" con columnas: [Aspecto | Baseline citado | Implementación TFM | ¿Coincide?].
3. Marca cada fila como ✅ coincide, ⚠️ diverge (comparación aproximada aceptable si se documenta), o ❌ no comparable (debe eliminarse o reformularse la comparación).
4. Si hay filas ❌, propone cómo reformular la afirmación en la memoria (p. ej. pasar de "superamos al estado del arte" a "obtenemos resultados en el mismo rango que X bajo un protocolo de evaluación distinto, ver limitaciones").
5. Redacta, si se pide, el texto en español académico para la sección de comparación con el estado del arte, dejando explícitas las diferencias metodológicas relevantes (esto es más defendible ante un tribunal que ocultar las diferencias).

No modifiques código de entrenamiento por tu cuenta al ejecutar esta auditoría — el objetivo es diagnóstico y de redacción de la comparación, no reimplementar el baseline salvo que se solicite explícitamente.
