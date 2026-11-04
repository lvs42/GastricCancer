from pathlib import Path
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor, white

OUT = Path('D:/GitHub/GastricCancer/output/memoria/figuras')
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1000, 1270
d = Drawing(W, H)
ink = '#203047'
muted = '#536174'
blue, teal, purple = '#35699A', '#237C77', '#79599B'
d.add(Rect(0, 0, W, H, fillColor=white, strokeColor=None))

def text(x, y, s, size=14, bold=False, color=ink, anchor='middle'):
    d.add(String(x, H-y, s, fontName='Helvetica-Bold' if bold else 'Helvetica',
                 fontSize=size, fillColor=HexColor(color), textAnchor=anchor))

def box(x, y, w, h, title, lines, accent=blue, fill='#F3F6FA', size=14):
    d.add(Rect(x, H-y-h, w, h, rx=9, ry=9, fillColor=HexColor(fill),
               strokeColor=HexColor(accent), strokeWidth=1.1))
    text(x+w/2, y+25, title, 16, True, accent)
    for i, line in enumerate(lines):
        text(x+w/2, y+49+i*19, line, size)

def line(x1,y1,x2,y2,color=muted):
    d.add(Line(x1,H-y1,x2,H-y2,strokeColor=HexColor(color),strokeWidth=1.4))

def arrow(x1,y1,x2,y2,color=muted):
    line(x1,y1,x2,y2,color)
    d.add(Polygon([x2,H-y2,x2-4,H-y2+7,x2+4,H-y2+7],
                  fillColor=HexColor(color),strokeColor=None))

text(500,32,'Pipeline experimental del proyecto',25,True)
text(500,55,'Clasificación de parches histopatológicos gástricos: Normal / Abnormal',14,color=muted)
box(30,76,940,76,'1. Datos y análisis exploratorio',[
    'GasHisSDB · 80 × 80, 120 × 120 y 160 × 160 píxeles · Experimentos separados por tamaño',
    'Inventario y clases · Color · Similitud perceptual · Duplicados RGB · Recuperación de procedencia'])
arrow(500,152,500,172)
box(30,172,940,76,'2. Preprocesamiento común',[
    'Conversión a RGB · Redimensionado a 224 × 224 · Escalado a [0, 1] · Normalización',
    'Media y desviación por tamaño, estimadas previamente sobre una muestra de 5.000 imágenes'])
arrow(500,248,500,268)
box(30,268,940,76,'3. Protocolo de partición y repetición',[
    'Cinco semillas (42-46) · División estratificada por parche: 80 % entrenamiento / 20 % prueba',
    'Mismos índices de partición para comparar las tres variantes en cada tamaño y semilla'])
line(500,344,500,361)
line(180,361,820,361)
for x in (180,500,820): arrow(x,361,x,381)

box(30,381,300,192,'A. Modelo de referencia',[
    'Imágenes originales',
    '',
    'EfficientNetV2-B0',
    'preentrenada en ImageNet',
    'Pesos congelados',
    'Agrupación espacial media',
    'Vector de 1.280 características'],blue,'#F1F6FC')
box(350,381,300,192,'B. Aumento de datos',[
    'Original + 3 vistas geométricas',
    'Rotaciones y reflexiones (D4)',
    'EfficientNetV2-B0 congelada',
    'Vectores de 1.280 características',
    'Vistas adicionales solo al ajustar',
    'Validación y prueba: originales',
    'Se conserva la proporción de clases'],teal,'#F0F8F6')
box(670,381,300,192,'C. Atención CBAM',[
    'EfficientNetV2-B0 congelada',
    'CBAM: atención de canal y espacial',
    'CBAM + clasificador temporal',
    'se entrenan con división interna',
    '90/10 del entrenamiento exterior',
    'Retirada del clasificador temporal',
    'Vector de 1.280 características'],purple,'#F7F3FA')
for x in (180,500,820): line(x,573,x,590)
line(180,590,820,590)
arrow(500,590,500,609)
box(30,609,940,95,'4. Clasificación con CatBoost · Una búsqueda por variante, tamaño y semilla',[
    'Características almacenadas en caché · Búsqueda de 27 configuraciones · Validación cruzada de 5 pliegues',
    'Selección por exactitud equilibrada · Ponderación de clases para tratar el desbalance',
    'Reajuste de CatBoost con la configuración elegida sobre el 80 % de entrenamiento'])
arrow(500,704,500,724)
box(30,724,940,95,'5. Evaluación sobre el 20 % de prueba',[
    'Predicción de clase y probabilidad · Clase positiva: Abnormal',
    'Exactitud · Exactitud equilibrada · Sensibilidad · Especificidad · Precisión · F1 · AUC-ROC',
    'Matrices de confusión · Intervalos bootstrap para las métricas implementadas'])
arrow(500,819,500,839)
box(30,839,940,76,'6. Comparación de resultados',[
    'Tres variantes × tres tamaños × cinco semillas = 45 ejecuciones principales',
    'Media y desviación típica de las cinco ejecuciones · Diferencias pareadas · Contraste de Wilcoxon'])

text(30,947,'ANÁLISIS COMPLEMENTARIOS',13,True,muted,'start')
box(30,963,460,115,'Transferencia entre tamaños',[
    'Modelo de referencia: 80/120, 80/160 y 120/160',
    'Entrenamiento en el 80 % del tamaño de origen',
    'Evaluación en todo el tamaño de destino',
    '6 direcciones × 5 semillas = 30 evaluaciones'],blue,size=14)
box(510,963,460,115,'Inspección de la atención',[
    'Mapas espaciales de CBAM sobre imágenes',
    'Comparación cualitativa y entradas sintéticas',
    'Puntos de control generados para visualización',
    'Análisis del módulo de atención'],purple,size=14)

d.add(Rect(30,H-1100-142,940,142,rx=9,ry=9,fillColor=HexColor('#FAF7EF'),
           strokeColor=HexColor('#C9B77F'),strokeWidth=0.8))
text(48,1123,'CLAVES PARA INTERPRETAR EL ESQUEMA',12,True,'#776235','start')
notes = [
    '• Las flechas representan el flujo lógico; las características congeladas pueden calcularse y guardarse antes de dividir.',
    '• CBAM se ajusta una vez por partición exterior, antes de la validación cruzada de CatBoost; no se reentrena por pliegue.',
    '• Las semillas controlan las particiones y CatBoost; no se fijó explícitamente la inicialización de CBAM en PyTorch.',
    '• No se dispone de identificadores para agrupar por paciente u origen; las particiones se realizan a nivel de parche.',
    '• La auditoría de duplicados no implica su eliminación. La evaluación entre tamaños no constituye validación externa.'
]
for i,s in enumerate(notes): text(48,1146+19*i,s,12,False,muted,'start')
renderPDF.drawToFile(d,str(OUT/'pipeline_completo.pdf'))
renderSVG.drawToFile(d,str(OUT/'pipeline_completo.svg'))
(OUT/'pipeline_completo.tex').write_text(r'''% Requiere \usepackage{graphicx}
\begin{figure}[p]
    \centering
    \includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]{figuras/pipeline_completo.pdf}
    \caption{Esquema del pipeline experimental para la clasificación Normal/Abnormal en GasHisSDB. Se muestran las etapas comunes, las tres variantes comparadas y los análisis complementarios. Las flechas representan el flujo lógico del procesamiento.}
    \label{fig:pipeline-completo}
\end{figure}
''',encoding='utf-8')
print(OUT/'pipeline_completo.pdf')
