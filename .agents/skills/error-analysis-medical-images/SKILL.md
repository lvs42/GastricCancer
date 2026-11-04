---
name: error-analysis-medical-images
description: Guía el análisis sistemático de errores de un modelo de clasificación binaria sobre GasHisSDB — falsos negativos, falsos positivos, patrones por tamaño de parche, interpretabilidad con Grad-CAM/mapas de saliencia. Usar después de entrenar un modelo y antes de escribir la sección de discusión/resultados de la memoria, o cuando el rendimiento agregado no explique por qué el modelo falla en casos concretos.
---

# Análisis de errores en imagen médica (GasHisSDB)

Esta skill estructura el análisis de errores de un clasificador Normal/Abnormal ya entrenado, yendo más allá de la métrica agregada para entender **qué tipo de errores comete y por qué**, algo que un tribunal de TFM valora tanto o más que la cifra final de accuracy/F1.

## Por qué esto importa más que la métrica agregada

Dos modelos con el mismo F1 pueden fallar de formas completamente distintas: uno puede fallar en casos ambiguos (border cases razonables), otro puede fallar sistemáticamente en un subgrupo (p. ej. un tamaño de parche concreto, o parches con poca celularidad). Solo el análisis de errores permite distinguir "el modelo es razonablemente robusto con margen de mejora esperable" de "el modelo tiene un sesgo sistemático preocupante". En un contexto clínico, esto es directamente relevante para la sección de limitaciones y trabajo futuro del TFM.

## Estructura del análisis

### 1. Desglose de la matriz de confusión más allá del número agregado
- Separar explícitamente falsos negativos (Abnormal clasificado como Normal — el error clínicamente más grave, tejido canceroso no detectado) de falsos positivos (Normal clasificado como Abnormal).
- Cuantificar cada tipo por tamaño de parche (80/120/160px) por separado — el dataset tiene tres subconjuntos con dificultad intrínseca potencialmente distinta; agregar todo en una sola cifra puede ocultar que el modelo falla mucho más en un tamaño concreto.

### 2. Inspección visual de casos mal clasificados
- Seleccionar y mostrar una muestra de falsos negativos y falsos positivos (no solo un par de ejemplos anecdóticos — proponer un muestreo sistemático, p. ej. los N casos con mayor confianza errónea del modelo, que son los más informativos sobre fallos sistemáticos).
- Para cada caso, describir cualitativamente qué podría explicar el error (baja celularidad, artefactos de tinción, región de tejido ambigua en el borde del parche, solapamiento de patrones normales/atípicos).
- Distinguir errores "razonables" (casos ambiguos donde un patólogo también dudaría) de errores "sistemáticos" (patrón repetido que sugiere un sesgo del modelo o del dataset, p. ej. fallar sistemáticamente en parches muy claros u oscuros por variabilidad de tinción no manejada — conectar con `augmentation-policy-designer`).

### 3. Interpretabilidad con Grad-CAM / mapas de saliencia
- Para arquitecturas CNN, generar mapas Grad-CAM (o equivalente, p. ej. Grad-CAM++, Score-CAM) sobre una muestra de aciertos y errores, verificando si el modelo atiende a regiones morfológicamente plausibles (núcleos, densidad celular) o a artefactos irrelevantes (bordes del parche, ruido de escaneo, marcas de tinción).
- Si el modelo acierta pero atiende a regiones no plausibles clínicamente (p. ej. una esquina sin tejido), señalarlo como señal de posible "shortcut learning" (el modelo aprendió una correlación espuria, no el patrón histológico real) — esto es especialmente relevante de discutir dado el riesgo de fuga de datos por WSI ya señalado en `dataset-audit-histopathology`: si hay fuga, el modelo puede estar memorizando textura de fondo en vez de morfología tumoral.
- Documentar la metodología de generación de los mapas (capa de la que se extraen, si se normalizan) para que sea reproducible.

### 4. Análisis por confianza del modelo
- Graficar la distribución de probabilidad predicha separada por acierto/error — un modelo bien calibrado debe mostrar errores concentrados cerca del umbral de decisión (incertidumbre genuina); errores con alta confianza (p. ej. 0.95 de probabilidad de Normal en un caso Abnormal) son más preocupantes y merecen inspección visual prioritaria.

### 5. Relación entre errores y variables del dataset
- Verificar si hay correlación entre error y tamaño de parche, y si es factible, entre error y cualquier otra variable disponible (p. ej. si se puede aproximar el WSI de origen por rango de índice de archivo, comprobar si los errores se concentran en ciertos rangos, lo que sugeriría dificultad específica de ciertas WSIs de origen).

## Procedimiento

1. Pide o localiza las predicciones del modelo sobre el conjunto de evaluación (idealmente con probabilidad, no solo clase dura) y, si existen, las imágenes correspondientes.
2. Ejecuta el desglose de la matriz de confusión por tamaño de parche y por tipo de error.
3. Propone/genera la selección de casos para inspección visual y Grad-CAM, priorizando errores de alta confianza y falsos negativos.
4. Redacta los hallazgos como una síntesis cualitativa + cuantitativa, distinguiendo errores razonables de sistemáticos, y conectando los hallazgos con implicaciones prácticas (p. ej. "el modelo no es fiable en casos de baja celularidad, lo que limita su uso como herramienta de cribado sin supervisión").
5. Si se pide, redacta la sección de "Análisis de errores" o "Discusión" de la memoria en español académico, con un tono honesto sobre limitaciones (esto se valora positivamente en TFMs aplicados a medicina).

Esta skill es de análisis y redacción; no reentrena el modelo. Si el análisis revela un defecto que requiere cambios en el pipeline (p. ej. augmentation, arquitectura), señálalo y remite a la skill correspondiente (`augmentation-policy-designer`, `model-selection-medical-imaging`) en vez de modificar código directamente sin confirmación.
