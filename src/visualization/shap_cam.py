"""
shap_cam.py - mapas de atribucion espacial sobre el clasificador REAL (CatBoost).

Grad-CAM calcula un peso por canal promediando gradientes y combina linealmente
los mapas de canal con esos pesos. CatBoost no es diferenciable, asi que aqui el
gradiente se sustituye por el valor SHAP de cada caracteristica, que TreeSHAP
calcula de forma exacta y por imagen.

Como el global average pooling es lineal, la descomposicion espacial es EXACTA:

    (1/hw) * sum_{i,j} sum_c alpha_c * A_c(i,j)  ==  sum_c alpha_c * GAP(A_c)

es decir, el mapa reparte espacialmente la contribucion real del clasificador a
la prediccion de esa imagen concreta.

Ventajas frente a visualizar la atencion de CBAM:
  - No hay sigmoide, luego no hay saturacion: el mapa tiene rango dinamico real.
  - El mapa tiene signo: rojo empuja hacia Abnormal, azul hacia Normal.
  - Funciona tambien sobre la configuracion de referencia, que no incorpora
    ningun mecanismo de atencion y hasta ahora no admitia analisis.

Limitacion que NO resuelve: la resolucion espacial sigue siendo la del mapa del
backbone (~7x7 para entrada de 224px), asi que las imagenes siguen viendose
gruesas al ampliarlas.

Uso:
    python -m src.visualization.shap_cam --resolution 160 --seed 42 --experiment baseline
    python -m src.visualization.shap_cam --resolution 120 --seed 42 --experiment baseline
    .
    .
    .
    python -m src.visualization.shap_cam --resolution 80 --seed 42 --experiment cbam
    .
    .
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
from catboost import Pool

from src.data.loaders import load_inventory, split_train_test
from src.data.preprocessing import (
    load_image_rgb,
    load_normalization_stats,
    preprocess_batch,
)
from src.features.extractors import (
    BACKBONE_NAME,
    FEATURES_DIR,
    build_backbone,
    get_or_compute_features,
)
from src.models.classifiers import fit_final_model


TEST_SIZE = 0.2
NUM_CHANNELS = 1280
N_EXAMPLES_PER_CLASS = 3

project_root = Path(__file__).resolve().parents[2]
RESULTS_DIR = project_root / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "cbam_checkpoints"
OUTPUT_DIR = RESULTS_DIR / "interpretability_shapcam"

CLASS_NAMES = {0: "Normal", 1: "Abnormal"}


# ---------------------------------------------------------------------------
# Hiperparametros ya seleccionados en el experimento original
# ---------------------------------------------------------------------------
def load_best_params(resolution: int, seed: int, experiment: str) -> dict:
    """Lee del JSON de resultados los hiperparametros elegidos para esa semilla."""
    if experiment == "baseline":
        candidatos = [
            RESULTS_DIR / f"baseline_{resolution}px_results.json",
            RESULTS_DIR / f"track_b_{resolution}px_results.json",
        ]
    else:
        candidatos = [RESULTS_DIR / f"{experiment}_{resolution}px_results.json"]

    ruta = next((c for c in candidatos if c.exists()), None)
    if ruta is None:
        raise FileNotFoundError(
            f"No se encontro el JSON de resultados para '{experiment}' a {resolution}px. "
            f"Buscado en: {[c.name for c in candidatos]}"
        )

    with open(ruta, "r", encoding="utf-8") as fh:
        datos = json.load(fh)

    for entrada in datos["per_seed"]:
        if entrada["seed"] == seed:
            print(f"Hiperparametros ({ruta.name}): {entrada['best_params']}")
            return entrada["best_params"]

    raise ValueError(f"La semilla {seed} no aparece en {ruta.name}")


# ---------------------------------------------------------------------------
# Modulo CBAM entrenado (solo para --experiment cbam)
# ---------------------------------------------------------------------------
def load_trained_cbam(resolution: int, seed: int, device: torch.device):
    """Reconstruye el CBAM y le carga los pesos guardados por cbam_checkpoint.py."""
    from src.models.cbam import CBAM

    ruta = CHECKPOINTS_DIR / f"cbam_{resolution}px_seed{seed}.pt"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint {ruta.name}. Generalo primero con:\n"
            f"  python -m src.interpretability.cbam_checkpoint "
            f"--resolution {resolution} --seed {seed}"
        )

    cbam = CBAM(NUM_CHANNELS)
    cbam.load_state_dict(torch.load(ruta, map_location=device, weights_only=True))
    cbam = cbam.to(device).eval()
    for p in cbam.parameters():
        p.requires_grad = False
    return cbam


# ---------------------------------------------------------------------------
# PASO 1: recuperar el clasificador de la configuracion a explicar
# ---------------------------------------------------------------------------
def rebuild_classifier(resolution: int, seed: int, experiment: str, device: torch.device):
    """Reentrena el CatBoost con los hiperparametros ya seleccionados. Es barato:
    las features estan cacheadas y no se repite la busqueda en malla."""
    paths, y = load_inventory(resolution)

    if experiment == "baseline":
        features = get_or_compute_features(paths, resolution, device)
    else:
        cache = FEATURES_DIR / f"{BACKBONE_NAME}_{resolution}px_cbam_seed{seed}.npy"
        if not cache.exists():
            raise FileNotFoundError(
                f"No existe el cache de features CBAM {cache.name}. Ejecuta antes "
                f"el experimento cbam para esa resolucion."
            )
        features = np.load(cache)
        print(f"Features CBAM del cache: {cache.name} {features.shape}")

    train_idx, test_idx = split_train_test(y, seed=seed, test_size=TEST_SIZE)
    best_params = load_best_params(resolution, seed, experiment)

    print(f"Entrenando CatBoost sobre {len(train_idx)} muestras...")
    model = fit_final_model(features[train_idx], y[train_idx], best_params, seed=seed)
    print("Clasificador listo.")

    return model, paths, y, features, test_idx


# ---------------------------------------------------------------------------
# PASO 2: pesos SHAP por canal, para UNA imagen
# ---------------------------------------------------------------------------
def shap_weights_for_image(model, feature_vector: np.ndarray) -> tuple[np.ndarray, float]:
    """Devuelve (alpha, valor_esperado). alpha son los 1280 pesos SHAP.

    get_feature_importance devuelve (n_muestras, n_features + 1): la ultima
    columna es el valor esperado del modelo, no un canal.
    """
    pool = Pool(feature_vector.reshape(1, -1))
    shap = model.get_feature_importance(pool, type="ShapValues")
    alpha = shap[0, :NUM_CHANNELS].astype(np.float64)
    valor_esperado = float(shap[0, NUM_CHANNELS])
    return alpha, valor_esperado


def verificar_shap(model, feature_vector, alpha, valor_esperado, tol=1e-3) -> bool:
    """sum(alpha) + valor_esperado debe igualar la prediccion en log-odds."""
    raw = float(np.ravel(
        model.predict(feature_vector.reshape(1, -1), prediction_type="RawFormulaVal")
    )[0])
    reconstruido = float(alpha.sum() + valor_esperado)
    ok = abs(raw - reconstruido) < tol
    if not ok:
        print(f"  [AVISO] SHAP no cuadra: raw={raw:.6f} vs reconstruido={reconstruido:.6f}")
    return ok


# ---------------------------------------------------------------------------
# PASO 3: mapa espacial del backbone, antes del pooling
# ---------------------------------------------------------------------------
@torch.no_grad()
def spatial_feature_map(image_path, backbone, mean, std, device, cbam=None) -> np.ndarray:
    """Devuelve el mapa (1280, h, w) sobre el que opera el pooling."""
    batch = preprocess_batch([image_path], mean, std)
    tensor = torch.from_numpy(batch).to(device)

    fmap = backbone.forward_features(tensor)
    if cbam is not None:
        fmap = cbam(fmap)

    return fmap[0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# PASO 4: combinar mapa y pesos
# ---------------------------------------------------------------------------
def build_cam(feature_map: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """cam(i,j) = sum_c alpha_c * A_c(i,j) / (h*w). Conserva el signo."""
    c, h, w = feature_map.shape
    if c != alpha.shape[0]:
        raise ValueError(
            f"Desajuste de canales: el mapa tiene {c} y alpha tiene {alpha.shape[0]}. "
            "Revisa que no se haya colado la columna del valor esperado."
        )
    return np.tensordot(alpha, feature_map, axes=([0], [0])) / (h * w)


# ---------------------------------------------------------------------------
# PASO 5: superposicion con mapa divergente centrado en cero
# ---------------------------------------------------------------------------
def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, vlim: float,
                alpha_blend: float = 0.5) -> np.ndarray:
    """Superpone el CAM con escala simetrica [-vlim, +vlim]."""
    alto, ancho = image_rgb.shape[0], image_rgb.shape[1]
    cam_resized = cv2.resize(cam.astype(np.float32), (ancho, alto),
                             interpolation=cv2.INTER_CUBIC)

    if vlim <= 0:
        norm = np.full_like(cam_resized, 0.5)
    else:
        norm = np.clip((cam_resized + vlim) / (2.0 * vlim), 0.0, 1.0)

    heatmap = plt.cm.bwr(norm)[:, :, :3] * 255.0
    mezcla = (1 - alpha_blend) * image_rgb.astype(np.float32) + alpha_blend * heatmap
    return np.clip(mezcla, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# PASO 6: orquestacion
# ---------------------------------------------------------------------------
def visualize_shap_cam(resolution: int, seed: int, experiment: str, device: torch.device):
    model, paths, y, features, test_idx = rebuild_classifier(
        resolution, seed, experiment, device
    )

    mean, std = load_normalization_stats(resolution)
    backbone = build_backbone(device)
    cbam = load_trained_cbam(resolution, seed, device) if experiment == "cbam" else None

    # Ejemplos del conjunto de TEST: imagenes que el clasificador no vio al ajustarse.
    seleccion = []
    for etiqueta in (0, 1):
        seleccion.extend([i for i in test_idx if y[i] == etiqueta][:N_EXAMPLES_PER_CLASS])

    print(f"\nCalculando SHAP-CAM para {len(seleccion)} imagenes de test...")

    ejemplos = []
    for pos, idx in enumerate(seleccion, start=1):
        alpha, valor_esperado = shap_weights_for_image(model, features[idx])
        verificar_shap(model, features[idx], alpha, valor_esperado)

        fmap = spatial_feature_map(paths[idx], backbone, mean, std, device, cbam)
        cam = build_cam(fmap, alpha)

        # La suma del mapa debe igualar sum(alpha): confirma el factor 1/(h*w).
        if abs(cam.sum() - alpha.sum()) > 1e-6 * max(1.0, abs(alpha.sum())):
            print(f"  [AVISO] cam.sum()={cam.sum():.6f} != alpha.sum()={alpha.sum():.6f}")

        etiqueta = int(y[idx])
        prob = float(model.predict_proba(features[idx].reshape(1, -1))[0, 1])
        ejemplos.append({
            "path": paths[idx],
            "clase": CLASS_NAMES[etiqueta],
            "cam": cam,
            "prob": prob,
        })
        print(f"  [{pos}/{len(seleccion)}] {CLASS_NAMES[etiqueta]:9s} "
              f"P(Abn)={prob:.3f}  rango CAM=[{cam.min():+.4f}, {cam.max():+.4f}]")

    # Escala de color compartida entre todas las imagenes del lote.
    vlim = max(float(np.abs(e["cam"]).max()) for e in ejemplos)
    print(f"\nEscala simetrica compartida: [{-vlim:+.4f}, {vlim:+.4f}]")
    print(f"Rango dinamico total: {2 * vlim:.4f}")
    print("(comparar con el rango ~0,03-0,05 de los mapas de atencion de CBAM)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    contadores = {"Normal": 0, "Abnormal": 0}
    for e in ejemplos:
        image_rgb = load_image_rgb(e["path"])
        overlay = overlay_cam(image_rgb, e["cam"], vlim)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image_rgb)
        axes[0].set_title(f"Original - {e['clase']}")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"SHAP-CAM   P(Abn)={e['prob']:.2f}")
        axes[1].axis("off")

        n = contadores[e["clase"]]
        contadores[e["clase"]] += 1
        nombre = f"shapcam_{experiment}_{resolution}px_{e['clase']}_{n}.png"
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / nombre, dpi=150)
        plt.close(fig)

    # Barra de color de referencia, para acompanar a las figuras en la memoria.
    fig, ax = plt.subplots(figsize=(6, 1.2))
    norm = matplotlib.colors.Normalize(vmin=-vlim, vmax=vlim)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.cm.bwr, norm=norm,
                                          orientation="horizontal")
    cb.set_label("Contribucion SHAP   (azul: Normal   |   rojo: Abnormal)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"shapcam_{experiment}_{resolution}px_escala.png", dpi=150)
    plt.close(fig)

    print(f"\nSe han guardado {len(ejemplos)} visualizaciones y la barra de escala.")
    print(f"Directorio: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Mapas de atribucion SHAP-CAM.")
    parser.add_argument("--resolution", type=int, required=True, choices=[80, 120, 160])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", choices=["baseline", "cbam"], default="baseline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"SHAP-CAM | {args.experiment} | {args.resolution}px | semilla {args.seed}")
    print(f"Dispositivo: {device}")
    print("=" * 70)

    visualize_shap_cam(args.resolution, args.seed, args.experiment, device)


if __name__ == "__main__":
    main()