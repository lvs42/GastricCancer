"""Prepara una copia legible y un paquete de las secciones redactadas."""
from pathlib import Path
import hashlib
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/memoria/actualizacion_shapcam"
FIG = OUT / "figuras"
FIG.mkdir(exist_ok=True)
sources = sorted((ROOT / "results/interpretability_shapcam").glob("*.png"))
assert len(sources) == 42, len(sources)
for source in sources:
    dest = FIG / source.name
    shutil.copy2(source, dest)
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(dest.read_bytes()).digest()

CITES = {
    "lundberg2017shap": "[Lundberg y Lee, 2017](https://arxiv.org/abs/1705.07874)",
    "lundberg2018tree": "[Lundberg et al., 2018](https://arxiv.org/abs/1802.03888)",
    "catboostShap": "[Documentación de CatBoost](https://catboost.ai/docs/en/concepts/shap-values)",
}
sections = [OUT / "metodologia_shapcam.tex", OUT / "resultados_shapcam.tex"]
tex = "\n\n".join(p.read_text(encoding="utf-8") for p in sections)
labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
assert refs <= labels, refs - labels
for filename in re.findall(r"\\includegraphics\[[^]]+\]\{([^}]+)\}", tex):
    assert (OUT / filename).is_file(), filename
for keys in re.findall(r"\\cite\{([^}]+)\}", tex):
    assert set(keys.split(",")) <= set(CITES), keys
assert tex.count("{") == tex.count("}")
stack = []
for command, env in re.findall(r"\\(begin|end)\{([^}]+)\}", tex):
    if command == "begin":
        stack.append(env)
    else:
        assert stack and stack.pop() == env, env
assert not stack

refnames = {
    "eq:shap-aditividad": "(1)",
    "eq:shapcam-proyeccion": "(2)",
    "eq:shapcam-suma": "(3)",
    "tab:shapcam-probabilidades": "1",
}

def table_to_md(match):
    rows = []
    for line in match.group(1).splitlines():
        if " & " not in line:
            continue
        cells = [x.strip() for x in line.replace("\\\\", "").split("&")]
        rows.append("| " + " | ".join(cells) + " |")
        if len(rows) == 1:
            rows.append("| --- | --- | --- | --- |")
    assert len(rows) == 20, len(rows)
    return "\n\n" + "\n".join(rows) + "\n\n"

md = re.sub(r"\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}", table_to_md, tex, flags=re.S)
md = re.sub(r"\\(?:subsubsection|subsection)\{([^}]+)\}", r"## \1", md)
md = re.sub(r"\\paragraph\{([^}]+)\}", r"### \1", md)
md = re.sub(r"\\label\{[^}]+\}", "", md)
md = re.sub(r"\\ref\{([^}]+)\}", lambda m: refnames.get(m[1], m[1]), md)
md = re.sub(r"\\cite\{([^}]+)\}", lambda m: "; ".join(CITES[k] for k in m[1].split(",")), md)
md = re.sub(r"\\texttt\{([^}]+)\}", r"`\1`", md)
md = re.sub(r"\\caption\{([^}]+)\}", r"\n\n*\1*\n\n", md)
md = re.sub(r"\\includegraphics\[[^]]+\]\{([^}]+)\}", lambda m: "\n\n![" + Path(m[1]).stem + "](" + (OUT / m[1]).as_posix() + ")\n\n", md)
md = md.replace(r"\begin{equation}", "$$").replace(r"\end{equation}", "$$")
md = re.sub(r"\\(?:begin|end)\{(?:table|figure)\}(?:\[htbp\])?", "", md)
md = re.sub(r"\\(?:centering|smallskip|small|par)\b", "", md)
md = md.replace(r"\,\%", "%").replace("~", " ")
md = re.sub(r"\n{3,}", "\n\n", md)
preface = "# Secciones para la memoria: SHAP y proyección espacial\n\nRedacción basada en el código y las figuras disponibles el 10 de septiembre de 2026. Los archivos `.tex` contienen la versión para integrar en la memoria.\n\n"
(OUT / "secciones_shapcam.md").write_text(preface + md.strip() + "\n", encoding="utf-8")

archive = ROOT / "output/memoria/actualizacion_shapcam_overleaf.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len([n for n in z.namelist() if n.endswith(".png")]) == 42
print(f"Preparado: {archive}")
print("Verificado: 42 PNG idénticos a la fuente; 18 filas de resultados; referencias, citas y entornos LaTeX; integridad ZIP.")
print("No se ha compilado LaTeX: no hay un compilador TeX local disponible.")
