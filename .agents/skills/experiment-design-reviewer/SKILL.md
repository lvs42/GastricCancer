---
name: experiment-design-reviewer
description: Revisa el diseño experimental de un experimento de clasificación binaria antes o después de ejecutarlo — splits, validación cruzada, control de semillas, protocolo de tuning de hiperparámetros, ablaciones, y si las conclusiones que se quieren extraer están soportadas por el diseño. Usar antes de lanzar una tanda de entrenamientos, al planificar el capítulo de metodología del TFM, o al revisar si un experimento ya ejecutado permite las conclusiones que se quieren reportar.
---

# Revisor de diseño experimental

Esta skill revisa el diseño experimental de los experimentos de clasificación binaria (GasHisSDB, Normal vs Abnormal) del TFM, no los resultados en sí. El objetivo es detectar defectos de diseño que invalidan o debilitan las conclusiones **antes** de invertir tiempo de cómputo, o al revisar retrospectivamente si un experimento ya hecho realmente permite afirmar lo que la memoria dice que permite.

## Relación con otras skills de este proyecto

- Para el riesgo específico de fuga de datos por WSI en GasHisSDB, usar `dataset-audit-histopathology`.
- Para elegir/justificar la arquitectura, usar `model-selection-medical-imaging`.
- Para verificar que las métricas elegidas son correctas, usar `binary-classification-metrics-medical`.
- Esta skill se centra en el **diseño del experimento como sistema**: qué se compara, cómo, y si el resultado es atribuible a la variable que se dice estudiar.

## Dimensiones a revisar

### 1. Pregunta experimental explícita
Todo experimento debe tener una pregunta de una frase antes de ejecutarse (p. ej. "¿la arquitectura A generaliza mejor que B en parches de 160px?"). Si no la hay, señálalo — un experimento sin pregunta no permite conclusión, solo un número suelto.

### 2. Splits y validación
- Confirmar train/val/test están separados y que el test no se ha tocado para ninguna decisión de diseño (arquitectura, hiperparámetros, umbral de decisión, selección de checkpoint).
- Si se usa validación cruzada (k-fold), verificar que el fold se estratifica por clase y, críticamente en este dataset, que no mezcla parches de la misma WSI entre folds de train y val (ver `dataset-audit-histopathology`).
- Si el TFM entrena sobre un tamaño de parche (p. ej. 160px) y evalúa transferibilidad a otro (80px/120px), verificar que esto se declara como experimento de generalización cross-scale y no se mezcla con la evaluación principal.
- Detectar "test set peeking": selección de mejor época/checkpoint usando accuracy de test en vez de validación — defecto grave y común.

### 3. Control de variables
- Cuando se compara A vs B (arquitecturas, augmentaciones, tamaños de parche), verificar que solo cambia la variable de interés. Señalar si además cambió el learning rate, el nº de épocas, la semilla, o el preprocesado sin que sea intencional — eso convierte la comparación en no concluyente.
- Verificar que las comparaciones usan el mismo split exacto (mismos archivos en train/val/test) entre condiciones, no splits regenerados aleatoriamente cada vez.

### 4. Reproducibilidad
- Semillas fijadas (Python, NumPy, framework de DL, y si aplica, `cudnn.deterministic`) y documentadas.
- Versión de librerías registrada (requirements.txt actualizado — este repo tiene uno vacío, señalarlo si se ejecutan experimentos sin fijar versiones).
- Código de generación de splits versionado y no regenerado manualmente entre ejecuciones.

### 5. Selección de hiperparámetros
- Verificar que el tuning (grid/random search, learning rate, etc.) se hizo sobre el conjunto de validación, no sobre test.
- Si se prueban múltiples configuraciones y se reporta solo la mejor, señalar el riesgo de "cherry-picking" y sugerir reportar la distribución de resultados o al menos mencionar cuántas configuraciones se probaron (relevante para la honestidad estadística del TFM).

### 6. Repeticiones y variabilidad
- Un único entrenamiento con una semilla no permite afirmar superioridad de un método sobre otro si la diferencia está dentro del ruido típico de inicialización. Sugerir, cuando el presupuesto de cómputo lo permita, repetir con 3-5 semillas y reportar media ± desviación estándar, especialmente para la comparación principal del TFM.
- Si el cómputo es limitado (razonable en un TFM), al menos señalarlo como limitación explícita en vez de presentar un único número como concluyente.

### 7. Ablaciones
- Si el TFM afirma que un componente (p. ej. una augmentación concreta, una capa, un preprocesado de tinción) "mejora los resultados", verificar que existe un experimento de ablación (con/sin ese componente, todo lo demás igual) que lo soporte. Una mejora observada solo por comparación con literatura externa no es una ablación válida.

## Procedimiento

1. Pide o localiza la descripción del experimento (código de entrenamiento, script de split, notebook) y la afirmación que se quiere justificar en la memoria.
2. Revisa cada dimensión anterior y produce un veredicto por punto: ✅ correcto / ⚠️ débil pero defendible si se documenta como limitación / ❌ invalida la conclusión.
3. Si hay ❌, propone el ajuste mínimo necesario (no un rediseño completo) para que el experimento soporte la conclusión deseada.
4. Si se pide, redacta el párrafo de "Diseño experimental" o "Protocolo de evaluación" de la memoria en español académico, siendo honesto sobre las limitaciones detectadas en vez de omitirlas.

No relances entrenamientos ni modifiques hiperparámetros por tu cuenta — esta skill diagnostica el diseño; los cambios de código o relanzar experimentos requieren confirmación explícita del usuario.
