from pathlib import Path
import json, csv, re, shutil, itertools, statistics, hashlib, zipfile, html
from PIL import Image, ImageDraw, ImageFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/memoria'
FIG = OUT/'figuras'
DATA = OUT/'datos'
DATA.mkdir(exist_ok=True)
ds={(r,e):json.loads((ROOT/f'results/{e}_{r}px_results.json').read_text(encoding='utf8')) for r in (80,120,160) for e in ('baseline','augmentation','cbam')}
cross=json.loads((ROOT/'results/cross_size_results.json').read_text())
names={'baseline':'Referencia','augmentation':'Aumento D4','cbam':'CBAM'}
metrics={'accuracy':'Exactitud','balanced_accuracy':'Exactitud balanceada','precision_abnormal':'Precisión Abnormal','recall_abnormal':'Sensibilidad Abnormal','specificity':'Especificidad','f1_abnormal':'F1 Abnormal','f1_macro':'F1 macro','auc_roc':'AUC-ROC'}
def f(x,n=2): return f'{x:.{n}f}'.replace('.',',')
def stat(d,m):
    v=d['aggregated_across_seeds'][m]
    return f(v['mean']*100)+' ± '+f(v['std']*100)
def tab(title,header,rows):
    return title+'\n\n|'+'|'.join(header)+'|\n|'+'|'.join(['---']*len(header))+'|\n'+''.join('|'+'|'.join(map(str,row))+'|\n' for row in rows)
def signed_exact(diff):
    nz=[round(z,12) for z in diff if abs(z)>1e-12]
    ranks=[1+sum(abs(t)<abs(z) for t in nz)+(sum(abs(t)==abs(z) for t in nz)-1)/2 for z in nz]
    w=sum(q for q,z in zip(ranks,nz) if z>0)
    totals=[sum(q*s for q,s in zip(ranks,bits)) for bits in itertools.product([0,1],repeat=len(nz))]
    p=min(1,2*min(sum(t<=w for t in totals),sum(t>=w for t in totals))/len(totals))
    return p,ranks

tables={}
counts={r:(ds[r,'baseline']['config']['n_normal'],ds[r,'baseline']['config']['n_abnormal']) for r in (80,120,160)}
tables['INVENTARIO']=tab('Tabla 3.1. Inventario de GasHisSDB. Fuente: recuentos locales y resultados almacenados.',['Tamaño (px)','Normal','Abnormal','Total','Abnormal (%)'],[[r,a,b,a+b,f(100*b/(a+b))] for r,(a,b) in counts.items()])
tables['NORMALIZACION']=tab('Tabla 3.2. Estadísticos RGB utilizados. Fuente: normalization_stats.yaml.',['Tamaño (px)','Media R / G / B','Desviación R / G / B'],[['80','0,9019 / 0,7846 / 0,9066','0,1276 / 0,1786 / 0,0993'],['120','0,9035 / 0,7863 / 0,9070','0,1262 / 0,1778 / 0,0995'],['160','0,9041 / 0,7873 / 0,9079','0,1259 / 0,1774 / 0,0983']])
tables['PARTICIONES']=tab('Tabla 3.3. Tamaños de entrenamiento y prueba en cada repetición.',['Tamaño (px)','Entrenamiento','Prueba','Normal en prueba','Abnormal en prueba'],[[r,ds[r,'baseline']['per_seed'][0]['n_train'],ds[r,'baseline']['per_seed'][0]['n_test'],sum(ds[r,'baseline']['per_seed'][0]['test_metrics']['confusion_matrix'][k] for k in ('tn','fp')),sum(ds[r,'baseline']['per_seed'][0]['test_metrics']['confusion_matrix'][k] for k in ('tp','fn'))] for r in counts])
tables['COLOR']=tab('Tabla 4.1. Medias por clase de la intensidad RGB. Fuente: salida del EDA.',['Tamaño (px)','Clase','R','G','B'],[[80,'Normal','0,9276','0,8136','0,9171'],[80,'Abnormal','0,8639','0,7418','0,8912'],[120,'Normal','0,9267','0,8127','0,9162'],[120,'Abnormal','0,8658','0,7434','0,8920'],[160,'Normal','0,9280','0,8139','0,9173'],[160,'Abnormal','0,8675','0,7464','0,8934']])
tables['DUPLICADOS']=tab('Tabla 4.2. Similitud perceptual en 10.000 imágenes por tamaño. Las parejas con distancia ≤ 5 incluyen las de distancia 0.',['Tamaño (px)','Parejas d = 0','Parejas d ≤ 5','Imágenes implicadas','Muestra (%)'],[[80,55,68,24,'0,24'],[120,4,8,10,'0,10'],[160,3,5,7,'0,07']])
for key,e,num in [('BASELINE','baseline',3),('AUMENTO','augmentation',4),('CBAM','cbam',5)]:
    tables[key]=tab(f'Tabla 4.{num}. {names[e]}: media ± desviación típica (%), cinco ejecuciones. Fuente: JSON principales.',['Métrica','80 px','120 px','160 px'],[[label]+[stat(ds[r,e],m) for r in counts] for m,label in metrics.items()])
cmap={(d['train_resolution'],d['eval_resolution']):d for d in cross}
tables['CROSS_MATRIZ']=tab('Tabla 4.6. Exactitud entre tamaños (%). Diagonal: prueba del 20 %; resto: destino completo.',['Origen / Destino','80 px','120 px','160 px'],[[str(r)+' px']+[stat(ds[r,'baseline'],'accuracy')+' (ref.)' if r==s else stat(cmap[r,s],'accuracy') for s in counts] for r in counts])
tables['CROSS_METRICAS']=tab('Tabla 4.7. Métricas entre tamaños (%), cinco ejecuciones.',['Origen → Destino','Exactitud','Balanceada','Sensibilidad','Especificidad'],[[f"{d['train_resolution']} → {d['eval_resolution']}"]+[stat(d,m) for m in ['accuracy','balanced_accuracy','recall_abnormal','specificity']] for d in cross])
tables['RESUMEN']=tab('Tabla 4.8. Síntesis de las nueve condiciones (%).',['Tamaño','Configuración','Exactitud','Balanceada','F1 Abnormal'],[[r,names[e]]+[stat(ds[r,e],m) for m in ['accuracy','balanced_accuracy','f1_abnormal']] for r in counts for e in names])
comparisons=[]
for r in counts:
    for a,b in [('augmentation','baseline'),('cbam','baseline'),('cbam','augmentation')]:
        x=ds[r,a]['aggregated_across_seeds']['accuracy']['values']; y=ds[r,b]['aggregated_across_seeds']['accuracy']['values']
        diff=[u-v for u,v in zip(x,y)]; p,ranks=signed_exact(diff)
        comparisons.append({'size':r,'a':a,'b':b,'delta_pp':statistics.mean(diff)*100,'wins':sum(z>1e-12 for z in diff),'losses':sum(z< -1e-12 for z in diff),'ties':sum(abs(z)<=1e-12 for z in diff),'p_exact':p,'differences':diff,'ranks_nonzero':ranks})
tables['CONTRASTES']=tab('Tabla 4.9. Comparaciones de exactitud. + / − / = indica mejoras, descensos y empates. Rangos medios en empates de magnitud; exclusión de diferencias nulas.',['Tamaño','Comparación','Diferencia (pp)','+ / − / =','p bilateral'],[[d['size'],names[d['a']]+' − '+names[d['b']],f(d['delta_pp']),f"{d['wins']} / {d['losses']} / {d['ties']}",f(d['p_exact'],4)] for d in comparisons])
cirows=[]; cmrows=[]
for r in counts:
    for e in names:
        s=ds[r,e]['per_seed'][0]; ci=s['test_metrics_bootstrap_ci95']['accuracy']; cm=s['test_metrics']['confusion_matrix']
        cirows.append([r,names[e],f(s['test_metrics']['accuracy']*100),f(ci['ci95_low']*100)+' a '+f(ci['ci95_high']*100)])
        cmrows.append([r,names[e]]+[cm[k] for k in ['tn','fp','fn','tp']])
tables['INTERVALOS']=tab('Tabla 4.10. Exactitud e intervalo bootstrap del 95 % de la semilla 42 (%).',['Tamaño','Configuración','Exactitud','Intervalo del 95 %'],cirows)
tables['CONFUSION']=tab('Tabla 4.11. Matrices de confusión de la semilla 42.',['Tamaño','Configuración','VN','FP','FN','VP'],cmrows)

copyfig={'results/eda/02_balance_clases.png':'balance_clases.png','results/interpretability/atencion_160px_Normal_0.png':'atencion_normal.png','results/interpretability/atencion_160px_Abnormal_0.png':'atencion_abnormal.png','test_atencion_sintetica.png':'atencion_sintetica.png'}
for src,dst in copyfig.items(): shutil.copy2(ROOT/src,FIG/dst)
ims=[Image.open(ROOT/f'results/eda/03_muestras_160_{c}.png').convert('RGB') for c in ['Normal','Abnormal']]
w=sum(im.width for im in ims); h=max(im.height for im in ims); combined=Image.new('RGB',(w,h),'white'); x=0
for im in ims: combined.paste(im,(x,0)); x+=im.width
combined.save(FIG/'muestras_160.png')
# Scientific summary chart, generated from the stored means, with explicit truncated axis.
im=Image.new('RGB',(1800,1000),'white'); dr=ImageDraw.Draw(im)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',30); small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
left,right,top,bottom=150,1710,150,860
def yy(v): return bottom-(v-85)/11*(bottom-top)
palette=['#596773','#578C93','#8A649B']
for v in range(85,97):
    y=yy(v); dr.line((left,y,right,y),fill='#D9DDE1',width=2); dr.text((60,y-16),str(v),fill='black',font=small)
dr.text((left,35),'Exactitud (%) · media y desviación típica de cinco ejecuciones',font=font,fill='black')
for i,r in enumerate(counts):
    center=390+i*510
    for j,e in enumerate(names):
        d=ds[r,e]['aggregated_across_seeds']['accuracy']; v=d['mean']*100; sd=d['std']*100; x=center+(j-1)*125
        dr.rectangle((x-48,yy(v),x+48,bottom),fill=palette[j]); dr.line((x,yy(v-sd),x,yy(v+sd)),fill='black',width=3)
        for vv in [v-sd,v+sd]: dr.line((x-15,yy(vv),x+15,yy(vv)),fill='black',width=3)
        dr.text((x-42,yy(v+sd)-43),f(v),font=small,fill='black')
    dr.text((center-50,bottom+20),str(r)+' px',font=font,fill='black')
for j,e in enumerate(names):
    x=300+j*500; dr.rectangle((x,950,x+30,980),fill=palette[j]); dr.text((x+45,947),names[e],font=small,fill='black')
im.save(FIG/'comparacion.png')

text='\n\n'.join((OUT/p).read_text(encoding='utf8') for p in ['contenido_base.md','resultados_base.md','discusion_base.md'])
for k,v in tables.items(): text=text.replace('{{'+k+'}}',v)
assert '{{' not in text
text=text.replace('Para las comparaciones sin empates corresponde al contraste bilateral exacto de Wilcoxon.','Se utilizan rangos medios cuando existen empates de magnitud; sin ellos, la enumeración coincide con la distribución exacta habitual de Wilcoxon.')

refs=[
'[1] Hu, W. y colaboradores (2022). GasHisSDB: A new gastric histopathology image dataset for computer aided diagnosis of gastric cancer. Computer Methods and Programs in Biomedicine / Computers in Biology and Medicine: ficha bibliográfica del artículo en PubMed. https://pubmed.ncbi.nlm.nih.gov/35016101/',
'[2] Tan, M. y Le, Q. (2021). EfficientNetV2: Smaller Models and Faster Training. Proceedings of Machine Learning Research, 139, 10096-10106. https://proceedings.mlr.press/v139/tan21a.html',
'[3] timm. Ficha de tf_efficientnetv2_b0.in1k, pesos portados a PyTorch e información del modelo. https://huggingface.co/timm/tf_efficientnetv2_b0.in1k',
'[4] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. y Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html',
'[5] Woo, S., Park, J., Lee, J.-Y. y Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. European Conference on Computer Vision. https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html',
'[6] Khayatian, D., Maleki, A., Nasiri, H. y Dorrigiv, M. (2025; publicación electrónica en 2024). Histopathology image analysis for gastric cancer detection: a hybrid deep learning and catboost approach. Multimedia Tools and Applications, 84, 21777-21803. Artículo retractado. https://doi.org/10.1007/s11042-024-19816-2',
'[7] Khayatian, D. y colaboradores (2026). Retraction Note: Histopathology image analysis for gastric cancer detection: a hybrid deep learning and catboost approach. Multimedia Tools and Applications, 85, 565. https://doi.org/10.1007/s11042-026-21725-5',
'[8] Yurdakul, M. y Tasdemir, S. (2025). CoAtNeXt: An Attention-Enhanced ConvNeXtV2-Transformer Hybrid Model for Gastric Tissue Classification. Prepublicación arXiv:2509.09242. https://arxiv.org/abs/2509.09242',
'[9] SciPy. Documentación de scipy.stats.wilcoxon: contraste pareado, hipótesis, diferencias nulas y empates. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html'
]
# The primary paper is in Computers in Biology and Medicine; do not mix journal titles.
refs[0]='[1] Hu, W. y colaboradores (2022). GasHisSDB: A new gastric histopathology image dataset for computer aided diagnosis of gastric cancer. Computers in Biology and Medicine, 142, 105207. https://doi.org/10.1016/j.compbiomed.2021.105207'
text+='\n\n# Bibliografía\n\n'+'\n\n'.join(refs)+'\n\nLas referencias electrónicas se han consultado el 5 de septiembre de 2026. Los resultados propios proceden de los artefactos del proyecto descritos en el anexo D.\n'
text+='\n# Anexo A. Configuración de ejecución\n\nEste anexo recoge la configuración del repositorio utilizada para documentar el entorno. Las versiones de Python y PyTorch registradas en los JSON son 3.11.9 y 2.4.0, respectivamente.\n\n'
req=[line.split('==') for line in (ROOT/'requirements.txt').read_text().splitlines() if '==' in line]
text+=tab('Tabla A.1. Dependencias fijadas en requirements.txt.',['Paquete','Versión'],req)
for title,file in [('A.1. Dockerfile','Dockerfile'),('A.2. Docker Compose','docker-compose.yml')]:
    text+='\n\n## '+title+'\n\n```text\n'+(ROOT/file).read_text()+'\n```\n'
text+='''\n## A.3. Comandos de ejecución\n\nEl siguiente bloque documenta la ejecución en un entorno Linux con el contenedor configurado. Cada comando de clasificación realiza las cinco repeticiones del protocolo. No se han relanzado entrenamientos para elaborar esta memoria.\n\n```text\ndocker compose build\ndocker compose run --rm tfm bash\npython -m src.training.run_experiments --resolution 80 --experiment baseline\npython -m src.training.run_experiments --resolution 120 --experiment baseline\npython -m src.training.run_experiments --resolution 160 --experiment baseline\npython -m src.training.run_experiments --resolution 80 --experiment augmentation\npython -m src.training.run_experiments --resolution 120 --experiment augmentation\npython -m src.training.run_experiments --resolution 160 --experiment augmentation\npython -m src.training.run_experiments --resolution 80 --experiment cbam\npython -m src.training.run_experiments --resolution 120 --experiment cbam\npython -m src.training.run_experiments --resolution 160 --experiment cbam\npython -m src.training.run_validacioncruzada\n```\n\nLa ejecución entre tamaños requiere los resultados de referencia para recuperar sus hiperparámetros. Las rutas del código suponen la estructura del conjunto descrita en el capítulo 3. La reutilización de cachés requiere que inventario, pesos y preprocesamiento sigan correspondiendo a sus ficheros.\n\n'''
text+='# Anexo B. Resultados desglosados por semilla\n\nLas siguientes tablas muestran la selección de hiperparámetros y las métricas principales de cada ejecución. Las cifras se expresan en porcentaje. El resto de las métricas, las matrices de todas las semillas y los intervalos de las cuatro métricas se conservan en los CSV complementarios.\n\n'
for j,(r,e) in enumerate(ds,1):
    rows=[]
    for s in ds[r,e]['per_seed']:
        p=s['best_params']; m=s['test_metrics']; rows.append([s['seed'],p['iterations'],f(p['learning_rate']),p['depth'],f(m['accuracy']*100),f(m['balanced_accuracy']*100),f(m['recall_abnormal']*100)])
    text+=f'## B.{j}. {names[e]}, {r} píxeles\n\n'+tab(f'Tabla B.{j}. Hiperparámetros y evaluación por semilla.',['Semilla','Árboles','Tasa','Prof.','Exactitud','Balanceada','Sensibilidad'],rows)+'\n\n'
text+='# Anexo C. Resultados completos y estructura de datos\n\n'
text+='Se conserva una copia de los diez JSON de resultados: nueve condiciones principales y la evaluación entre tamaños. Los JSON principales incluyen la configuración, el entorno, las medias y desviaciones, los resultados de cada semilla, la rejilla completa de validación y los intervalos bootstrap. El fichero entre tamaños contiene seis bloques, cada uno con sus cinco ejecuciones.\n\n'
text+='Los ficheros resultados_por_semilla.csv, matrices_confusion.csv, intervalos_bootstrap.csv, validacion_cruzada.csv y cross_size_por_semilla.csv facilitan consultar los resultados sin recorrer manualmente las estructuras JSON. Los contrastes calculados para esta memoria se almacenan en contrastes_exactos.json, con diferencias y rangos. No se han creado probabilidades ni predicciones individuales ausentes en los artefactos originales.\n\n'
text+='# Anexo D. Procedencia y correspondencia de las evidencias\n\n'
provenance=[['Inventario y color','eda.ipynb; results/dataset_inventory.csv; configs/normalization_stats.yaml'],['Particiones y etiquetas','src/data/loaders.py'],['Preprocesamiento y D4','src/data/preprocessing.py; src/data/data_augmentation.py'],['Extracción congelada','src/features/extractors.py'],['Atención y aprendizaje','src/models/cbam.py; src/features/cbam_extractors.py'],['Selección y experimentos','src/models/classifiers.py; src/training/run_experiments.py'],['Métricas y bootstrap','src/evaluation/metrics.py'],['Transferencia entre tamaños','src/training/run_validacioncruzada.py; results/cross_size_results.json'],['Visualización y puntos de control','src/visualization/visualize.py; src/visualization/cbam_checkpoint.py'],['Comprobación uniforme','src/test_atencion_sintetica.py; test_atencion_sintetica.png']]
text+=tab('Tabla D.1. Relación entre contenidos y artefactos del proyecto.',['Contenido','Artefactos'],provenance)
text+='\n\nLos recuentos se han contrastado con los ficheros locales y las tablas numéricas se han obtenido de los JSON. El cuaderno aporta las salidas conservadas de la exploración. La documentación contextual complementa las motivaciones y el diagnóstico histórico de atención; no reemplaza al código cuando existen diferencias. Las rutas se expresan respecto a la raíz del repositorio.\n'
(OUT/'memoria_desde_metodologia.md').write_text(text,encoding='utf8')

# Export original evidence and flat tables; verify aggregate arithmetic and confusion identities.
flat=[]; cms=[]; cis=[]; cvs=[]; crossrows=[]; manifest={}
for (r,e),d in ds.items():
    src=ROOT/f'results/{e}_{r}px_results.json'; shutil.copy2(src,DATA/src.name); manifest[src.name]=hashlib.sha256(src.read_bytes()).hexdigest()
    for m,ag in d['aggregated_across_seeds'].items():
        vals=[s['test_metrics'][m] for s in d['per_seed']]
        assert abs(statistics.mean(vals)-ag['mean'])<1e-12
        assert abs(statistics.pstdev(vals)-ag['std'])<1e-12
    for s in d['per_seed']:
        meta={'size':r,'experiment':e,'seed':s['seed']}; m=s['test_metrics']; cm=m['confusion_matrix']; n=sum(cm[k] for k in ['tn','fp','fn','tp'])
        assert n==s['n_test']; assert abs((cm['tn']+cm['tp'])/n-m['accuracy'])<1e-12
        flat.append(meta|s['best_params']|{k:v for k,v in m.items() if isinstance(v,(float,int))})
        cms.append(meta|{k:cm[k] for k in ['tn','fp','fn','tp']})
        for key,ci in s['test_metrics_bootstrap_ci95'].items(): cis.append(meta|{'metric':key}|ci)
        for conf in s['cv']['all_results']:
            for fold,score in enumerate(conf['cv_fold_scores'],1): cvs.append(meta|conf['params']|{'fold':fold,'balanced_accuracy':score})
for d in cross:
    for s in d['per_seed']: crossrows.append({'origin':d['train_resolution'],'destination':d['eval_resolution'],'seed':s['seed']}|{k:v for k,v in s['metrics'].items() if isinstance(v,(float,int))})
for name,rows in [('resultados_por_semilla.csv',flat),('matrices_confusion.csv',cms),('intervalos_bootstrap.csv',cis),('validacion_cruzada.csv',cvs),('cross_size_por_semilla.csv',crossrows)]:
    with (DATA/name).open('w',encoding='utf-8-sig',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
shutil.copy2(ROOT/'results/cross_size_results.json',DATA/'cross_size_results.json')
manifest['cross_size_results.json']=hashlib.sha256((DATA/'cross_size_results.json').read_bytes()).hexdigest()
(DATA/'contrastes_exactos.json').write_text(json.dumps(comparisons,indent=2,ensure_ascii=False),encoding='utf8')
(DATA/'sha256_resultados.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')

# Minimal block reader shared by PDF and LaTeX.
def blocks(md):
    ls=md.splitlines(); i=0
    while i<len(ls):
        line=ls[i]
        if not line.strip(): i+=1; continue
        if line.startswith('```'):
            i+=1; v=[]
            while i<len(ls) and not ls[i].startswith('```'): v.append(ls[i]); i+=1
            yield 'code','\n'.join(v); i+=1; continue
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#')); yield 'h'+str(n),line[n:].strip(); i+=1; continue
        if line.startswith('!['):
            m=re.match(r'!\[(.*?)\]\((.*?)\)',line); yield 'image',m.groups(); i+=1; continue
        if line.startswith('|'):
            rows=[]
            while i<len(ls) and ls[i].startswith('|'):
                if not re.fullmatch(r'[|\-: ]+',ls[i]): rows.append([c.strip() for c in ls[i].strip('|').split('|')])
                i+=1
            yield 'table',rows; continue
        v=[line]; i+=1
        while i<len(ls) and ls[i].strip() and not ls[i].startswith(('#','|','![','```')): v.append(ls[i]); i+=1
        yield 'p',' '.join(v)

pdfmetrics.registerFont(TTFont('ThesisSerif','C:/Windows/Fonts/times.ttf'))
pdfmetrics.registerFont(TTFont('ThesisSans','C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('ThesisMono','C:/Windows/Fonts/consola.ttf'))
body=ParagraphStyle('body',fontName='ThesisSerif',fontSize=11.2,leading=15.1,alignment=TA_JUSTIFY,spaceAfter=8,splitLongWords=True,allowWidows=0,allowOrphans=0)
hs={1:ParagraphStyle('h1',fontName='ThesisSans',fontSize=20,leading=25,spaceBefore=24,spaceAfter=22,keepWithNext=True),2:ParagraphStyle('h2',fontName='ThesisSans',fontSize=14,leading=18,spaceBefore=17,spaceAfter=10,keepWithNext=True),3:ParagraphStyle('h3',fontName='ThesisSans',fontSize=11.6,leading=15.5,spaceBefore=13,spaceAfter=7,keepWithNext=True)}
caption=ParagraphStyle('caption',parent=body,fontSize=9.2,leading=12,alignment=TA_CENTER,spaceBefore=6,spaceAfter=12)
tablecap=ParagraphStyle('tablecap',parent=caption,keepWithNext=True)
cell=ParagraphStyle('cell',fontName='ThesisSerif',fontSize=9,leading=11.6,spaceAfter=0)
code=ParagraphStyle('code',fontName='ThesisMono',fontSize=8,leading=10.5,spaceAfter=0)
width=A4[0]-142
story=[]
for kind,val in blocks(text):
    if kind.startswith('h'):
        level=int(kind[1]);
        if level==1 and story: story.append(PageBreak())
        story.append(Paragraph(html.escape(val),hs[level]))
    elif kind=='p': story.append(Paragraph(html.escape(val),tablecap if val.startswith('Tabla ') else body))
    elif kind=='table':
        n=len(val[0]); widths=[width/n]*n
        if n==4 and val[0][0]=='Métrica': widths=[width*.34]+[width*.22]*3
        if n==2: widths=[width*.32,width*.68]
        rows=[[Paragraph(html.escape(c),cell) for c in row] for row in val]
        t=Table(rows,colWidths=widths,repeatRows=1,hAlign='CENTER')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EBEDF0')),('LINEBELOW',(0,0),(-1,0),.6,colors.grey),('LINEBELOW',(0,-1),(-1,-1),.5,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,12)])
    elif kind=='image':
        cap,path=val; pic=Image.open(OUT/path); w=width; h=w*pic.height/pic.width
        story.append(KeepTogether([RLImage(str(OUT/path),width=w,height=h),Paragraph(html.escape(cap),caption)]))
    else:
        for line in val.splitlines(): story.append(Paragraph(html.escape(line).replace(' ','&#160;') or '&#160;',code))
        story.append(Spacer(1,12))
def footer(c,d):
    c.setFont('ThesisSerif',9); c.drawRightString(A4[0]-71,38,str(d.page))
doc=SimpleDocTemplate(str(OUT/'memoria_desde_metodologia.pdf'),pagesize=A4,leftMargin=71,rightMargin=71,topMargin=65,bottomMargin=60,title='Clasificación de tejido gástrico en GasHisSDB: metodología, resultados y discusión',author='Lucila Vázquez Soriano')
doc.build(story,onFirstPage=footer,onLaterPages=footer)

def esc(s):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','≤':r'\ensuremath{\leq}','→':r'\ensuremath{\rightarrow}','−':r'\ensuremath{-}','±':r'\ensuremath{\pm}','×':r'\ensuremath{\times}','μ':r'\ensuremath{\mu}','σ':r'\ensuremath{\sigma}','Σ':r'\ensuremath{\Sigma}','⊙':r'\ensuremath{\odot}','′':r'\ensuremath{\prime}','″':r'\ensuremath{\prime\prime}','⁵':r'\textsuperscript{5}'}
    return ''.join(replacements.get(c,c) for c in s)
def texinline(s):
    parts=re.split(r'(https?://\S+)',s)
    return ''.join(r'\url{'+p+'}' if p.startswith('http') else esc(p) for p in parts)
tex=[r'''\documentclass[11pt,a4paper]{article}
\usepackage[spanish,es-nodecimaldot]{babel}
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\usepackage[margin=25mm]{geometry}
\usepackage{graphicx,longtable,array,booktabs,amsmath}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\titleformat{\section}{\sffamily\Large\bfseries}{}{0em}{}
\titleformat{\subsection}{\sffamily\large\bfseries}{}{0em}{}
\titleformat{\subsubsection}{\sffamily\normalsize\bfseries}{}{0em}{}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{1.2em}
\setlength{\emergencystretch}{3em}
\begin{document}
''']
first=True
for kind,val in blocks(text):
    if kind.startswith('h'):
        level=int(kind[1]); cmd=['','section','subsection','subsubsection'][level]
        if level==1 and not first: tex.append(r'\clearpage')
        first=False; tex.append('\\'+cmd+'*{'+esc(val)+'}\n'+r'\addcontentsline{toc}{'+cmd+'}{'+esc(val)+'}')
    elif kind=='p': tex.append(texinline(val)+'\n')
    elif kind=='table':
        n=len(val[0]); col=(1/n)-.027
        spec=''.join(r'>{\raggedright\arraybackslash}p{'+f'{col:.3f}'+r'\textwidth}' for _ in range(n))
        tex.append(r'{\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.25}'+'\n'+r'\begin{longtable}{'+spec+'}\n'+r'\toprule')
        tex.append(' & '.join(texinline(c) for c in val[0])+r' \\ \midrule\endhead')
        tex.extend(' & '.join(texinline(c) for c in row)+r' \\' for row in val[1:]); tex.append(r'\bottomrule\end{longtable}}')
    elif kind=='image':
        cap,path=val; tex.append(r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{'+path+'}\n'+r'\par\small '+esc(cap)+r'\end{figure}')
    else: tex.append(r'\begin{quote}\scriptsize\ttfamily\raggedright'+'\n'+r'\par '.join(esc(line).replace(' ',r'\ ') for line in val.splitlines())+'\n'+r'\end{quote}')
tex.append(r'\end{document}')
(OUT/'main.tex').write_text('\n\n'.join(tex),encoding='utf8')
with zipfile.ZipFile(OUT/'memoria_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    z.write(OUT/'main.tex','main.tex')
    for p in FIG.glob('*.png'): z.write(p,'figuras/'+p.name)
with zipfile.ZipFile(OUT/'evidencias_resultados.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in DATA.iterdir(): z.write(p,p.name)
report={'words':len(text.split()),'pages':len(PdfReader(OUT/'memoria_desde_metodologia.pdf').pages),'main_runs':len(flat),'cv_rows':len(cvs),'bootstrap_rows':len(cis),'cross_runs':len(crossrows),'tables':sum(1 for k,v in blocks(text) if k=='table'),'figures':len(list(FIG.glob('*.png'))),'verified':'Aggregate means, population std and accuracy from confusion matrices match original JSON.'}
(ROOT/'tmp/memoria/validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
