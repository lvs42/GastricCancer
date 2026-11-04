---
name: dataset-audit-histopathology
description: Audita la integridad, estructura y riesgos de fuga de datos del dataset GasHisSDB (u otros datasets histopatológicos de parches) antes de entrenar o reportar resultados. Usar antes de cualquier split train/val/test, antes de reportar estadísticas del dataset en la memoria del TFM, o cuando se sospeche de duplicados, desbalance no documentado o fuga entre WSIs.
---

# Auditor de dataset histopatológico (GasHisSDB)

Esta skill audita el dataset de imágenes histopatológicas de cáncer gástrico usado en el TFM, con foco específico en los riesgos que este dataset concreto plantea. No es una skill genérica de "revisar datos": está calibrada a las particularidades conocidas de GasHisSDB.

## Contexto del dataset (verificado en este repo)

Ruta local: `dataset/GasHisSDB/GasHisSDB/{80,120,160}/{Abnormal,Normal}/`

Estructura confirmada:
- Tres tamaños de parche: 80×80, 120×120, 160×160 px, generados por recorte de las mismas WSIs (Whole Slide Images) a distinta resolución/escala. **No son splits independientes** — el mismo tejido origen puede aparecer en más de un tamaño.
- Dos clases: `Abnormal` (cáncer/tejido tumoral) y `Normal`.
- Recuentos de referencia (última auditoría en este repo):
  - 80px: 59151 Abnormal / 87500 Normal
  - 120px: 24801 Abnormal / 40460 Normal
  - 160px: 13124 Abnormal / 20160 Normal
- Nomenclatura de archivos: `Abnormal-00001.png`, `Normal-00001.png` — **índices secuenciales sin ID de paciente ni de WSI de origen**. Esto es una limitación documentada del release público de GasHisSDB.

## Riesgo crítico: fuga de datos a nivel de WSI

**Este es el hallazgo más importante que debe verificarse siempre en este dataset.** GasHisSDB genera múltiples parches (patches) por cada WSI. Como los nombres de archivo no exponen el WSI de origen, un split aleatorio a nivel de parche (`train_test_split` estándar sobre la lista de imágenes) casi con certeza coloca parches vecinos —recortados de la misma región de tejido— en train y en test simultáneamente. Esto infla artificialmente el rendimiento reportado porque el modelo puede memorizar textura/tinción local en vez de aprender patrones generalizables de malignidad.

Cuando se audite el dataset o el pipeline de carga de datos, comprobar SIEMPRE:

1. **¿Existe algún mapeo a WSI de origen?** Buscar si el TFM tiene acceso a metadatos adicionales (ficheros CSV, IDs de paciente, o el paper original de Hu et al. 2022) que permitan reconstruir la procedencia. Si no existe tal mapeo, documentarlo explícitamente como limitación metodológica en la memoria — no ocultarlo.
2. **¿El split es a nivel de parche o a nivel de paciente/WSI?** Si es a nivel de parche (lo más común en código encontrado online para este dataset), señalar el riesgo de fuga con severidad alta y proponer alternativas:
   - Si hay forma de recuperar WSI de origen: split estratificado por WSI, no por parche.
   - Si no la hay: reportar el resultado como upper bound optimista y discutirlo como limitación explícita, o usar un split por rango de índice de archivo (los parches consecutivos suelen provenir de la misma WSI o región) como aproximación conservadora, documentando el supuesto.
3. **¿Hay overlap espacial entre parches?** Si el pipeline de generación de parches usó ventana deslizante con solapamiento, parches adyacentes comparten píxeles — otra vía de fuga incluso dentro del "mismo" split si no se filtra.
4. **¿Se mezclan los tres tamaños (80/120/160) en el mismo experimento?** Si es así, verificar que no se traten como fuentes de datos independientes para aumentar el tamaño muestral, ya que probablemente derivan de las mismas WSIs a distinta escala — esto duplicaría información entre train/test si se combinan sin control.

## Checklist de auditoría de integridad

Al ejecutar esta skill sobre el dataset o sobre el código de carga de datos (`Dataset`/`DataLoader`, scripts de preprocesado):

- **Conteo y balance**: recontar archivos por clase y tamaño, comparar con los valores de referencia anteriores; señalar cualquier discrepancia (archivos corruptos, movidos, eliminados).
- **Duplicados exactos**: proponer/ejecutar verificación por hash (MD5/SHA1) de las imágenes para detectar copias exactas dentro o entre clases/tamaños.
- **Fugas entre splits**: si ya existen ficheros de split (train.csv, val.csv, test.csv o carpetas), verificar que ningún nombre de archivo aparezca en más de un split.
- **Desbalance de clases**: calcular el ratio Abnormal:Normal por tamaño y verificar que la memoria del TFM lo documente y que la estrategia de entrenamiento/métricas lo tenga en cuenta (ver skill `binary-classification-metrics-medical`).
- **Calidad de imagen**: verificar dimensiones reales de archivo vs. lo esperado por carpeta (80×80, 120×120, 160×160), modo de color (RGB vs RGBA vs escala de grises), y presencia de artefactos evidentes (parches en blanco, fuera de foco) si se pide inspección visual.
- **Reproducibilidad del split**: verificar que el split usa una semilla fija y que el código para generarlo está versionado (no un split ad-hoc hecho una vez y guardado sin trazabilidad).

## Formato de salida

Al finalizar la auditoría, producir un informe breve con:
1. Tabla de conteos por clase/tamaño con cualquier discrepancia frente a la referencia.
2. Veredicto explícito sobre el riesgo de fuga por WSI: **presente / mitigado / no verificable** con la justificación.
3. Lista priorizada de acciones correctivas (si las hay), ordenadas por impacto en la validez de los resultados.
4. Redacción sugerida (en español académico) para la sección de "Limitaciones del dataset" de la memoria, si se detectan riesgos no mitigables.

No reescribas código de carga de datos ni ejecutes reentrenamientos automáticamente — esta skill es de auditoría/diagnóstico. Si se proponen cambios de código, preséntalos y espera confirmación antes de aplicarlos.
