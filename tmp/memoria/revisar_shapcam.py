from pathlib import Path
from reportlab.pdfgen import canvas
from PIL import Image

root = Path('D:/GitHub/GastricCancer')
src = root / 'results/interpretability_shapcam'
out = root / 'tmp/memoria/shapcam_contactos.pdf'
c = canvas.Canvas(str(out), pagesize=(840, 740))
for resolution in (80, 120, 160):
    for experiment in ('baseline', 'cbam'):
        c.setFont('Helvetica-Bold', 16)
        c.drawString(20, 715, f'{experiment} - {resolution} px')
        for n in range(3):
            for col, cls in enumerate(('Normal', 'Abnormal')):
                p = src / f'shapcam_{experiment}_{resolution}px_{cls}_{n}.png'
                c.setFont('Helvetica', 10)
                c.drawString(20+col*410, 694-n*202, f'{cls}_{n}')
                c.drawImage(str(p), 20+col*410, 498-n*202, width=400, height=200,
                            preserveAspectRatio=True, anchor='c')
        p = src / f'shapcam_{experiment}_{resolution}px_escala.png'
        c.drawImage(str(p), 200, 1, width=440, height=88)
        c.showPage()
c.save()
for p in sorted(src.glob('*.png')):
    im=Image.open(p)
    print(p.name, im.size, im.info.get('Software',''))
