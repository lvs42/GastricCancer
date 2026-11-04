---
name: model-selection-medical-imaging
description: Ayuda a elegir y justificar arquitecturas de red para clasificación binaria de imágenes histopatológicas de parches pequeños (80/120/160px, GasHisSDB), considerando transfer learning, tamaño de parche, coste computacional disponible en un TFM y justificación defendible ante un tribunal. Usar al decidir qué arquitectura(s) probar, al justificar por qué se eligió una arquitectura en la memoria, o al evaluar si una arquitectura es excesiva/insuficiente para el tamaño de imagen y de dataset.
---

# Selección de modelo para imagen médica (GasHisSDB)

Esta skill guía la elección y justificación de arquitecturas para clasificación binaria Normal/Abnormal sobre parches histopatológicos pequeños del dataset GasHisSDB (80×80, 120×120, 160×160 px). No es una guía genérica de arquitecturas de CNN: está calibrada a las restricciones concretas de este problema.

## Restricciones específicas del problema

- **Resolución de entrada muy baja** (80-160 px) comparada con benchmarks naturales típicos (224-384px en ImageNet). Arquitecturas diseñadas y preentrenadas para 224×224+ (ResNet, EfficientNet, ViT estándar) requieren upsampling del parche o adaptación de la primera capa/stride, lo que introduce decisiones de diseño que deben justificarse, no darse por sentadas.
- **Dataset de tamaño moderado por clase/tamaño de parche** (ver conteos reales en `dataset-audit-histopathology`): decenas de miles de imágenes por combinación tamaño-clase, suficiente para fine-tuning pero probablemente insuficiente para entrenar una arquitectura grande desde cero de forma robusta.
- **Dominio distinto de ImageNet**: tejido teñido con H&E (hematoxilina-eosina) tiene estadísticas de color y textura muy distintas de imágenes naturales. Transfer learning ayuda pero debe evaluarse, no asumirse ciegamente como superior a entrenar desde cero.
- **Coste computacional de TFM**: normalmente una GPU de consumo o recursos universitarios limitados, no un clúster. La elección de arquitectura debe ser defendible también en términos de "qué era razonable ejecutar con los recursos disponibles", no solo de rendimiento teórico.
- **Interpretabilidad clínica**: en un TFM de aplicación médica, se valora que el modelo permita algún grado de explicabilidad (Grad-CAM, mapas de activación) — esto favorece arquitecturas CNN convolucionales estándar sobre arquitecturas más opacas si no hay una razón de peso para lo contrario, y debe conectarse con `error-analysis-medical-images`.

## Marco de decisión

Al recomendar o revisar una arquitectura, cubrir explícitamente:

1. **Baseline simple primero**: recomendar siempre incluir una CNN pequeña entrenada desde cero (p. ej. una arquitectura tipo VGG reducida o una CNN custom de pocas capas) como referencia inferior, antes de arquitecturas más complejas. Si el TFM salta directo a una arquitectura pesada sin este baseline, señalarlo como hueco metodológico — sin él no se puede argumentar que la complejidad añadida aporta valor.
2. **Transfer learning vs desde cero**: si se usa transfer learning (ImageNet), justificar explícitamente el mismatch de dominio (natural images vs H&E) y recomendar comparar contra entrenamiento desde cero o contra fine-tuning completo vs solo la cabeza, como parte del diseño experimental (enlazar con `experiment-design-reviewer`).
3. **Adaptación a baja resolución**: si se usa una arquitectura preentrenada a 224px, documentar explícitamente cómo se adapta el parche pequeño (resize/upsampling con qué interpolación, o modificación del stride inicial) y discutir el trade-off: upsampling introduce artefactos vs reducir stride pierde compatibilidad con pesos preentrenados.
4. **Complejidad proporcional al dataset**: señalar si el número de parámetros del modelo elegido es desproporcionado frente al tamaño del dataset disponible (riesgo de overfitting) o si es tan pequeño que limita la capacidad de aprender patrones histológicos relevantes (underfitting). Sugerir curvas de aprendizaje (train vs val loss) como evidencia, no solo intuición.
5. **Coste vs beneficio marginal**: si se comparan varias arquitecturas, exigir que la comparación reporte también coste de entrenamiento (tiempo, nº de parámetros, FLOPs aproximados) junto al rendimiento — relevante para la discusión de viabilidad clínica/despliegue en la memoria, un punto que valoran positivamente los tribunales de TFM aplicados.
6. **Arquitecturas candidatas razonables para este problema** (a proponer según el objetivo del TFM, no como lista cerrada): CNNs ligeras tipo ResNet-18/34 o EfficientNet-B0 con fine-tuning como punto medio robusto; una CNN pequeña custom como baseline; y, solo si el TFM tiene foco explícito en ello, un Vision Transformer pequeño (con la advertencia de que ViT suele necesitar más datos o preentrenamiento fuerte para superar a CNNs en datasets de este tamaño).

## Qué evitar

- No recomendar arquitecturas "porque son las más nuevas/de moda" sin conectar la elección con las restricciones anteriores.
- No dar por sentado que una arquitectura más grande da mejor resultado en este dataset — pedir evidencia empírica (curva val accuracy/loss) antes de afirmarlo en la memoria.
- No ignorar el coste de justificar la elección: el tribunal de un TFM valora "por qué esta arquitectura y no otra" tanto como el resultado final.

## Procedimiento

1. Si se pide elegir arquitectura(s), proponer 2-3 candidatas con justificación explícita ligada a los puntos anteriores, no una única recomendación sin alternativas discutidas.
2. Si se pide revisar una elección ya hecha en el código, evaluarla contra el marco de decisión y señalar huecos de justificación.
3. Si se pide, redactar el párrafo de justificación de arquitectura de la memoria en español académico, citando explícitamente las restricciones de resolución, dominio y recursos que motivaron la elección.

No implementes ni entrenes modelos automáticamente al ejecutar esta skill salvo que se solicite explícitamente — el rol por defecto es asesorar y justificar la elección.
