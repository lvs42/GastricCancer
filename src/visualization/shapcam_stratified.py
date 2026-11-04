"""
shapcam_stratified.py - analisis de interpretabilidad sobre muestra estratificada.

Amplia el analisis cualitativo de shap_cam.py en dos direcciones:

  1. Selecciona una muestra estratificada del conjunto de evaluacion en cuatro
     categorias: aciertos de alta confianza, aciertos de baja confianza, falsos
     positivos y falsos negativos.

  2. Calcula estadisticos agregados sobre los mapas de atribucion de cada
     categoria, de modo que las conclusiones no dependan de la inspeccion de
     imagenes individuales.

Los tres estadisticos calculados no requieren anotaciones de especialista:

  - Fraccion de masa positiva: proporcion de la atribucion total que empuja
    hacia la clase anomala. Caracteriza el sentido de la evidencia acumulada.

  - Concentracion espacial (indice de Gini sobre |CAM|): distingue una
    atribucion focalizada en pocas posiciones de otra repartida de forma
    homogenea.

  - Correlacion entre atribucion e intensidad local: contrasta si las
    posiciones de mayor contribucion hacia la clase anomala se corresponden con
    regiones de menor intensidad, esto es, con mayor densidad de tejido teñido.

Uso:
    python -m src.interpretability.shapcam_stratified --resolution 160 --seed 42
    python -m src.interpretability.shapcam_stratified --resolution 160 --seed 42 \
        --experiment cbam --n-por-categoria 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.preprocessing import load_image_rgb, load_normalization_stats
from src.features.extractors import build_backbone
from src.visualization.shap_cam import (
    build_cam,
    load_trained_cbam,
    overlay_cam,
    rebuild_classifier,
    shap_weights_for_image,
    spatial_feature_map,
)

# Umbrales de confianza. La confianza se define como max(p, 1-p), de modo que
# 0,5 corresponde a maxima indecision y 1,0 a maxima seguridad.
CONFIANZA_ALTA = 0.90
CONFIANZA_BAJA = 0.70

UMBRAL_DECISION = 0.5

CATEGORIAS = [
    "acierto_alta_confianza",
    "acierto_baja_confianza",
    "falso_positivo",
    "falso_negativo",
]

ETIQUETAS_CATEGORIA = {
    "acierto_alta_confianza": "Acierto, alta confianza",
    "acierto_baja_confianza": "Acierto, baja confianza",
    "falso_positivo": "Falso positivo",
    "falso_negativo": "Falso negativo",
}

project_root = Path(__file__).resolve().parents[2]
RESULTS_DIR = project_root / "results"
OUTPUT_DIR = RESULTS_DIR / "interpretabilidad_estratificada"


# ---------------------------------------------------------------------------
# Estratificacion del conjunto de evaluacion
# ---------------------------------------------------------------------------
def clasificar_en_categorias(y_true: np.ndarray, probabilidades: np.ndarray) -> dict:
    """Asigna cada observacion de evaluacion a una de las cuatro categorias.

    Las observaciones correctas con confianza intermedia (entre CONFIANZA_BAJA y
    CONFIANZA_ALTA) no pertenecen a ninguna categoria: se excluyen
    deliberadamente para que las dos categorias de acierto resulten claramente
    diferenciadas entre si.
    """
    y_pred = (probabilidades >= UMBRAL_DECISION).astype(np.int64)
    confianza = np.maximum(probabilidades, 1.0 - probabilidades)
    acierto = y_pred == y_true

    indices = {
        "acierto_alta_confianza": np.where(acierto & (confianza >= CONFIANZA_ALTA))[0],
        "acierto_baja_confianza": np.where(acierto & (confianza < CONFIANZA_BAJA))[0],
        "falso_positivo": np.where((y_true == 0) & (y_pred == 1))[0],
        "falso_negativo": np.where((y_true == 1) & (y_pred == 0))[0],
    }
    return indices, y_pred, confianza


def muestrear(indices: dict, n_por_categoria: int, semilla: int) -> dict:
    """Extrae una muestra aleatoria de cada categoria, de forma reproducible."""
    rng = np.random.default_rng(semilla)
    muestra = {}
    for categoria in CATEGORIAS:
        disponibles = indices[categoria]
        if len(disponibles) == 0:
            print(f"  [AVISO] La categoria '{categoria}' no tiene observaciones.")
            muestra[categoria] = np.array([], dtype=int)
            continue
        n = min(n_por_categoria, len(disponibles))
        if n < n_por_categoria:
            print(f"  [AVISO] La categoria '{categoria}' solo tiene {n} observaciones "
                  f"(se solicitaron {n_por_categoria}).")
        muestra[categoria] = rng.choice(disponibles, size=n, replace=False)
    return muestra


# ---------------------------------------------------------------------------
# Estadisticos agregados sobre los mapas
# ---------------------------------------------------------------------------
def fraccion_masa_positiva(cam: np.ndarray) -> float:
    """Proporcion de la atribucion total que empuja hacia la clase anomala."""
    masa_total = np.abs(cam).sum()
    if masa_total == 0:
        return 0.5
    return float(cam[cam > 0].sum() / masa_total)


def indice_gini(cam: np.ndarray) -> float:
    """Concentracion de la magnitud de la atribucion. 0 indica reparto uniforme
    entre posiciones y valores proximos a 1 indican concentracion en unas pocas."""
    valores = np.sort(np.abs(cam).ravel())
    n = valores.size
    suma = valores.sum()
    if suma == 0:
        return 0.0
    indices = np.arange(1, n + 1)
    return float((2.0 * (indices * valores).sum()) / (n * suma) - (n + 1.0) / n)


def _rangos(x: np.ndarray) -> np.ndarray:
    """Rangos de un vector, para el calculo de la correlacion de Spearman."""
    orden = np.argsort(x)
    rangos = np.empty_like(orden, dtype=float)
    rangos[orden] = np.arange(len(x), dtype=float)
    return rangos


def correlacion_intensidad(cam: np.ndarray, image_rgb: np.ndarray) -> float:
    """Correlacion de Spearman entre la atribucion de cada posicion y la
    intensidad media del tejido en esa misma posicion.

    Un valor negativo indica que las posiciones que mas contribuyen a la clase
    anomala se corresponden con regiones oscuras, esto es, con mayor densidad de
    tejido teñido.
    """
    alto, ancho = cam.shape
    # Se reduce la imagen a la rejilla del mapa promediando cada region.
    intensidad = cv2.resize(
        image_rgb.astype(np.float32).mean(axis=2) / 255.0,
        (ancho, alto),
        interpolation=cv2.INTER_AREA,
    )
    a, b = cam.ravel(), intensidad.ravel()
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(_rangos(a), _rangos(b))[0, 1])


# ---------------------------------------------------------------------------
# Analisis principal
# ---------------------------------------------------------------------------
def analizar(resolution: int, seed: int, experiment: str, n_por_categoria: int,
             semilla_muestreo: int, device: torch.device) -> None:
    model, paths, y, features, test_idx = rebuild_classifier(
        resolution, seed, experiment, device
    )

    y_test = y[test_idx]
    x_test = features[test_idx]

    print("\nCalculando predicciones sobre el conjunto de evaluacion...")
    probabilidades = model.predict_proba(x_test)[:, 1]

    indices, y_pred, confianza = clasificar_en_categorias(y_test, probabilidades)

    # ----- Distribucion de categorias en el conjunto completo -----
    print(f"\n{'=' * 78}")
    print(f"DISTRIBUCION DE CATEGORIAS EN EL CONJUNTO DE EVALUACION ({len(y_test)} parches)")
    print(f"{'=' * 78}")
    distribucion = {}
    for categoria in CATEGORIAS:
        n = len(indices[categoria])
        pct = 100.0 * n / len(y_test)
        distribucion[categoria] = {"n": int(n), "porcentaje": float(pct)}
        print(f"  {ETIQUETAS_CATEGORIA[categoria]:<28} {n:>7}  ({pct:>5.2f} %)")

    n_intermedios = len(y_test) - sum(len(indices[c]) for c in CATEGORIAS)
    print(f"  {'Acierto, confianza intermedia':<28} {n_intermedios:>7}  "
          f"({100.0 * n_intermedios / len(y_test):>5.2f} %)")

    # ----- Muestreo estratificado -----
    print(f"\nMuestreando hasta {n_por_categoria} parches por categoria "
          f"(semilla de muestreo: {semilla_muestreo})...")
    muestra = muestrear(indices, n_por_categoria, semilla_muestreo)

    # ----- Preparacion del extractor -----
    mean, std = load_normalization_stats(resolution)
    backbone = build_backbone(device)
    cbam = load_trained_cbam(resolution, seed, device) if experiment == "cbam" else None

    # ----- Calculo de mapas y estadisticos -----
    registros = []
    total = sum(len(v) for v in muestra.values())
    print(f"\nCalculando mapas de atribucion para {total} parches...")

    contador = 0
    for categoria in CATEGORIAS:
        for pos_local in muestra[categoria]:
            contador += 1
            idx_global = test_idx[pos_local]

            alpha, _ = shap_weights_for_image(model, features[idx_global])
            fmap = spatial_feature_map(paths[idx_global], backbone, mean, std, device, cbam)
            cam = build_cam(fmap, alpha)
            image_rgb = load_image_rgb(paths[idx_global])

            registros.append({
                "categoria": categoria,
                "indice_global": int(idx_global),
                "path": str(paths[idx_global]),
                "etiqueta_real": int(y_test[pos_local]),
                "prediccion": int(y_pred[pos_local]),
                "prob_abnormal": float(probabilidades[pos_local]),
                "confianza": float(confianza[pos_local]),
                "cam": cam,
                "image_rgb": image_rgb,
                "frac_masa_positiva": fraccion_masa_positiva(cam),
                "gini": indice_gini(cam),
                "corr_intensidad": correlacion_intensidad(cam, image_rgb),
                "rango_cam": float(cam.max() - cam.min()),
            })
            print(f"  [{contador}/{total}] {ETIQUETAS_CATEGORIA[categoria]:<28} "
                  f"P(Abn)={probabilidades[pos_local]:.3f}", end="\r")

    print(" " * 90, end="\r")

    # ----- Agregacion por categoria -----
    print(f"\n{'=' * 78}")
    print("ESTADISTICOS AGREGADOS POR CATEGORIA (media +- desviacion tipica)")
    print(f"{'=' * 78}")
    print(f"{'Categoria':<28} {'n':>4} {'Masa pos.':>13} {'Gini':>13} "
          f"{'Corr. intens.':>15}")
    print("-" * 78)

    agregados = {}
    for categoria in CATEGORIAS:
        subset = [r for r in registros if r["categoria"] == categoria]
        if not subset:
            continue

        def resumen(clave):
            valores = np.array([r[clave] for r in subset], dtype=float)
            valores = valores[~np.isnan(valores)]
            if valores.size == 0:
                return {"media": float("nan"), "std": float("nan")}
            return {"media": float(valores.mean()), "std": float(valores.std(ddof=1))
                    if valores.size > 1 else 0.0}

        agregados[categoria] = {
            "n_muestra": len(subset),
            "n_poblacion": distribucion[categoria]["n"],
            "frac_masa_positiva": resumen("frac_masa_positiva"),
            "gini": resumen("gini"),
            "corr_intensidad": resumen("corr_intensidad"),
            "rango_cam": resumen("rango_cam"),
            "prob_abnormal": resumen("prob_abnormal"),
        }

        a = agregados[categoria]
        print(f"{ETIQUETAS_CATEGORIA[categoria]:<28} {len(subset):>4} "
              f"{a['frac_masa_positiva']['media']:>6.3f} ± {a['frac_masa_positiva']['std']:<5.3f} "
              f"{a['gini']['media']:>6.3f} ± {a['gini']['std']:<5.3f} "
              f"{a['corr_intensidad']['media']:>7.3f} ± {a['corr_intensidad']['std']:<5.3f}")

    print("-" * 78)
    print("Masa pos.: fraccion de la atribucion total que empuja hacia la clase anomala.")
    print("Gini: concentracion espacial de la atribucion (0 uniforme, 1 concentrada).")
    print("Corr. intens.: correlacion de Spearman entre atribucion e intensidad local")
    print("               (negativa = mayor contribucion en regiones oscuras).")

    # ----- Figuras -----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vlim = max(float(np.abs(r["cam"]).max()) for r in registros)

    for categoria in CATEGORIAS:
        subset = [r for r in registros if r["categoria"] == categoria]
        if not subset:
            continue
        carpeta = OUTPUT_DIR / f"{experiment}_{resolution}px" / categoria
        carpeta.mkdir(parents=True, exist_ok=True)

        for n, r in enumerate(subset):
            overlay = overlay_cam(r["image_rgb"], r["cam"], vlim)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(r["image_rgb"])
            axes[0].set_title(f"Original - {'Anomalo' if r['etiqueta_real'] else 'Normal'}")
            axes[0].axis("off")
            axes[1].imshow(overlay)
            axes[1].set_title(f"SHAP-CAM   P(Abn)={r['prob_abnormal']:.2f}")
            axes[1].axis("off")
            fig.suptitle(ETIQUETAS_CATEGORIA[categoria], fontsize=11)
            fig.tight_layout()
            fig.savefig(carpeta / f"{categoria}_{n:02d}.png", dpi=150)
            plt.close(fig)

    # Barra de color compartida.
    fig, ax = plt.subplots(figsize=(6, 1.2))
    norm = matplotlib.colors.Normalize(vmin=-vlim, vmax=vlim)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.cm.bwr, norm=norm,
                                          orientation="horizontal")
    cb.set_label("Contribucion SHAP   (azul: Normal   |   rojo: Anomalo)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{experiment}_{resolution}px" / "escala.png", dpi=150)
    plt.close(fig)

    # ----- Volcado de resultados -----
    salida = {
        "experimento": experiment,
        "resolution_px": resolution,
        "semilla_modelo": seed,
        "semilla_muestreo": semilla_muestreo,
        "umbral_decision": UMBRAL_DECISION,
        "confianza_alta": CONFIANZA_ALTA,
        "confianza_baja": CONFIANZA_BAJA,
        "n_test": int(len(y_test)),
        "distribucion_categorias": distribucion,
        "escala_simetrica": float(vlim),
        "agregados_por_categoria": agregados,
        "parches": [
            {k: v for k, v in r.items() if k not in ("cam", "image_rgb")}
            for r in registros
        ],
    }
    ruta = RESULTS_DIR / f"interpretabilidad_estratificada_{experiment}_{resolution}px.json"
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(salida, fh, indent=2, ensure_ascii=False)

    print(f"\nFiguras guardadas en: {OUTPUT_DIR / f'{experiment}_{resolution}px'}")
    print(f"Resultados guardados en: {ruta}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analisis de interpretabilidad sobre muestra estratificada."
    )
    parser.add_argument("--resolution", type=int, required=True, choices=[80, 120, 160])
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla del modelo a explicar.")
    parser.add_argument("--experiment", choices=["baseline", "cbam"], default="baseline")
    parser.add_argument("--n-por-categoria", type=int, default=12,
                        help="Numero de parches a muestrear de cada categoria.")
    parser.add_argument("--semilla-muestreo", type=int, default=2026,
                        help="Semilla del muestreo, independiente de la del modelo.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"INTERPRETABILIDAD ESTRATIFICADA | {args.experiment} | "
          f"{args.resolution}px | semilla {args.seed}")
    print(f"Dispositivo: {device}")
    print("=" * 78)

    analizar(args.resolution, args.seed, args.experiment,
             args.n_por_categoria, args.semilla_muestreo, device)


if __name__ == "__main__":
    main()