---
name: binary-classification-metrics-medical
description: Verifica que las métricas de evaluación reportadas para la clasificación binaria Normal/Abnormal (GasHisSDB) sean las correctas dado el desbalance de clases y el contexto clínico, y que estén bien calculadas e interpretadas. Usar al definir qué métricas reportar antes de entrenar, al revisar una tabla de resultados antes de incluirla en la memoria, o al interpretar resultados de un modelo ya entrenado.
---

# Métricas de clasificación binaria en contexto médico

Esta skill audita y guía la elección, cálculo e interpretación de métricas para la tarea de clasificación binaria Normal vs Abnormal sobre GasHisSDB, con foco en errores comunes de TFMs de imagen médica.

## Por qué accuracy sola no es suficiente aquí

Los conteos reales del dataset (ver `dataset-audit-histopathology`) muestran desbalance de clases no trivial en las tres resoluciones (p. ej. a 80px: 59151 Abnormal vs 87500 Normal ≈ 40/60; a 160px: 13124 vs 20160 ≈ 39/61). No es un desbalance extremo, pero es suficiente para que:
- Un clasificador que prediga siempre la clase mayoritaria (Normal) obtenga ~60% de accuracy sin haber aprendido nada clínicamente útil.
- Comparar modelos solo por accuracy pueda ocultar diferencias reales en la capacidad de detectar la clase minoritaria.

**Regla por defecto de esta skill: nunca aceptar accuracy como única métrica reportada.** Exigir como mínimo el conjunto siguiente.

## Conjunto mínimo de métricas a exigir

1. **Matriz de confusión completa** (TP, TN, FP, FN) — base de todo lo demás, siempre reportable y debe incluirse aunque sea en anexo.
2. **Sensibilidad / Recall de la clase Abnormal** (TP / (TP+FN)): en cáncer, un falso negativo (tejido anómalo clasificado como Normal) tiene coste clínico mucho mayor que un falso positivo. Esta es probablemente la métrica más importante del TFM y debe destacarse, no enterrarse en una tabla.
3. **Especificidad / Recall de la clase Normal** (TN / (TN+FP)): para entender la tasa de sobrediagnóstico.
4. **Precisión (PPV) de la clase Abnormal**: relevante para entender cuántas alarmas son falsas alarmas.
5. **F1-score** (y opcionalmente F-beta con beta>1 para ponderar más el recall, dado el coste asimétrico de FN) — mejor que accuracy sola ante desbalance, pero no sustituye a reportar sensibilidad/especificidad por separado.
6. **AUC-ROC**: útil como medida agregada independiente del umbral, pero recordar que con desbalance moderado puede ser optimista; complementar con AUC-PR (curva precisión-recall), más informativa cuando la clase positiva es minoritaria.
7. **Accuracy balanceada** (media de sensibilidad y especificidad) si se quiere un número único en vez de accuracy simple — siempre preferible a accuracy a secas dado el desbalance documentado.

## Errores frecuentes a detectar al revisar resultados

- **Reportar solo accuracy** en la tabla principal de resultados de la memoria — señalarlo y pedir el conjunto mínimo anterior.
- **Promediar métricas sin ponderar por clase de forma explícita** (confundir micro-avg con macro-avg en librerías como scikit-learn) sin declarar cuál se usó — verificar el parámetro `average=` en `sklearn.metrics` y que la memoria lo especifique.
- **Umbral de decisión fijo en 0.5 sin justificación**: en un contexto clínico donde FN son más costosos que FP, puede ser razonable justificar un umbral distinto (o al menos discutir la curva ROC/PR para mostrar el trade-off) en vez de asumir 0.5 sin más.
- **Confundir "Abnormal" con "positivo"**: verificar que el código y la matriz de confusión usan consistentemente la misma convención de qué clase es la positiva en todo el pipeline (entrenamiento, cálculo de métricas, gráficas), un error sutil que invierte sensibilidad y especificidad si se mezcla.
- **Reportar una sola cifra sin intervalo/variabilidad**: si hay varias semillas o folds (ver `experiment-design-reviewer`), exigir media ± desviación estándar en vez de un único número puntual.
- **Comparar métricas distintas entre modelos** (p. ej. accuracy de un modelo contra F1 de otro) al construir tablas comparativas — verificar consistencia de columnas.
- **No reportar el tamaño de parche junto a la métrica**: como GasHisSDB tiene tres resoluciones con distinto nivel de dificultad intrínseca, cada tabla de resultados debe dejar claro a qué tamaño corresponde cada cifra.
- **Calibración ignorada**: si el modelo se usaría para estimar probabilidad de malignidad (no solo clase dura), verificar si se discute calibración (reliability diagram, Brier score) — opcional pero valorado en TFMs orientados a aplicación clínica.

## Procedimiento

1. Si se pide definir métricas antes de entrenar, proponer el conjunto mínimo anterior y el criterio de qué clase es positiva, dejándolo explícito en el código (nombrar variables/columnas sin ambigüedad).
2. Si se revisa una tabla de resultados ya generada, verificar cada punto de la lista de errores frecuentes y señalar cuáles aplican.
3. Si se pide interpretar resultados, priorizar la lectura clínica (¿cuántos casos de cáncer se están perdiendo — FN — y qué implicaría eso en un uso real?) sobre la lectura puramente numérica.
4. Si se pide, redactar el párrafo de "Métricas de evaluación" o la interpretación de resultados en español académico, usando terminología estándar en español (sensibilidad, especificidad, valor predictivo positivo/negativo) y evitando anglicismos innecesarios (conectar con `thesis-style-spanish-editor`).

No recalcules métricas ejecutando código de forma autónoma salvo que se solicite explícitamente — esta skill primero diagnostica qué está mal o falta, y propone el cálculo correcto antes de ejecutarlo.
