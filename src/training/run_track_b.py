""" python -m src.training.run_track_b --resolution 80
python -m src.training.run_track_b --resolution 120
python -m src.training.run_track_b --resolution 160 """

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.data.loaders import (
    VALID_RESOLUTIONS,
    load_inventory,
    split_train_test,
    validate_resolution,
)
from src.evaluation.metrics import aggregate_across_seeds, bootstrap_ci, compute_metrics
from src.features.extractors import BACKBONE_NAME, get_or_compute_features
from src.models.classifiers import PARAM_GRID, fit_final_model, tune_hyperparameters

# Cada semilla generará su propio split train/test
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

TEST_SIZE = 0.2 #20%


current_file = Path(__file__)

absolute_file_path = current_file.resolve()
project_root = absolute_file_path.parents[2]

RESULTS_DIR = project_root / "results"


def run_seed(features: np.ndarray, y: np.ndarray, seed: int) -> tuple[dict, dict]:
    """Un experimento completo para una semilla"""

    print(f"\n{'=' * 70}\nSEMILLA {seed}\n{'=' * 70}")

    # separar train y test
    train_idx, test_idx = split_train_test(y, seed=seed, test_size=TEST_SIZE)
    x_train, y_train = features[train_idx], y[train_idx]
    x_test, y_test = features[test_idx], y[test_idx]

    print(
        f"Train: {len(y_train)} (Abnormal {int(y_train.sum())}) | "
        f"Test: {len(y_test)} (Abnormal {int(y_test.sum())})"
    )

    
    print(f"\nTuning por CV (k=5) sobre el 80% de train...")
    #busqueda de los mejores hiperparámetros (solo sobre train(que dentro lleva el conjunto de validación))
    best_params, cv_info = tune_hyperparameters(x_train, y_train, seed=seed)

    print(
        f"\nMejores hiperparametros: {best_params}\n"
        f"CV balanced accuracy: {cv_info['best_cv_balanced_accuracy_mean']:.4f} "
        f"± {cv_info['best_cv_balanced_accuracy_std']:.4f}"
    )


    print("\nReentrenando modelo final sobre el 80% completo...")
    # Entrenameinto final del mdoelo con los mejores hiperparámetros encontrados + todo el conjunto de train
    model = fit_final_model(x_train, y_train, best_params, seed=seed)

    #y_pred = model.predict(x_test).astype(np.int64).ravel()
    y_pred = model.predict(x_test).astype(np.int64) # devuelve clase -> convertir a int -> (N,1 a N,0)
    #print("Forma original:", y_pred.shape)
    y_score = model.predict_proba(x_test)[:, 1] #devuelve la probabilidad de pertenecer a cada clase (caso binario, solo hace falta saebr una)

    # gurdamso todos los resultados
    metrics = compute_metrics(y_test, y_pred, y_score)
    metrics_ci = bootstrap_ci(y_test, y_pred, y_score, seed=seed)

    print(
        f"TEST -> accuracy {metrics['accuracy']:.4f} | "
        f"balanced acc {metrics['balanced_accuracy']:.4f} | "
        f"F1(Abnormal) {metrics['f1_abnormal']:.4f} | AUC {metrics['auc_roc']:.4f}"
    )

    seed_result = {
        "seed": seed,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "best_params": best_params,
        "cv": cv_info,
        "test_metrics": metrics,
        "test_metrics_bootstrap_ci95": metrics_ci,
    }
    # Solo los escalares se agregan entre semillas (la matriz de confusion no).
    flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    return seed_result, flat


def main() -> None:
    parser = argparse.ArgumentParser() #permite recibir parámetros por terminal

    parser.add_argument("--resolution", type=int, required=True, choices=VALID_RESOLUTIONS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-strict-counts", action="store_true")

    args = parser.parse_args()


    resolution = validate_resolution(args.resolution)



     
    if torch.cuda.is_available(): #usa gpu si se puede, sino cpu
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # Mostrar información del experimento.
    separator = "=" * 70

    print(separator)
    print(f"TRACK B — {BACKBONE_NAME} congelado + CatBoost")
    print(f"Resolución: {resolution}px")
    print(f"Dispositivo: {device}")
    print(separator)
    

    #  Inventario
    paths, y = load_inventory(resolution, strict_counts=not args.no_strict_counts)

    n_abn  = int((y == 1).sum())
    n_nor = int((y == 0).sum())

    print(f"Inventario {resolution}px: {len(paths)} imagenes ")
    print(f"Abnormal: {n_abn} ")
    print(f"Normal: {n_nor}")
    print()

     # si existe cache la carga, sino ejecutar tf_efficientnetv2_b0 para crearlo
    features = get_or_compute_features(paths, resolution, device, batch_size=args.batch_size)


    # Listas para guardar los resultados de todas las semillas.
    per_seed_results: list[dict] = []
    per_seed_flat: list[dict] = []

    # repetimos el experimento completo por cada semilla
    for seed in SEEDS:
        result, flat = run_seed(features, y, seed)
        per_seed_results.append(result)
        per_seed_flat.append(flat)

    aggregated = aggregate_across_seeds(per_seed_flat)
    

    ############
    #RESULTADOS
    output = {
        "track": "B",
        "description": "Baseline corregido sin fuga: EfficientNetV2B0 (timm, congelado) + CatBoost.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "resolution_px": resolution,
            "backbone": BACKBONE_NAME,
            "backbone_frozen": True,
            "pooling": "global_average_pooling",
            "classifier": "CatBoost",
            "input_size_px": 224,
            "color_space": "RGB (cv2 BGR -> RGB)",
            "normalization": f"configs/normalization_stats.yaml :: gashisdb_{resolution}",
            "class_weights": "frecuencia inversa (sin submuestreo)",
            "test_size": TEST_SIZE,
            "cv_folds": 5,
            "seeds": list(SEEDS),
            "param_grid": PARAM_GRID,
            "n_images": int(len(y)),
            "n_abnormal": n_abn,
            "n_normal": n_nor,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "aggregated_across_seeds": aggregated,
        "per_seed": per_seed_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"track_b_{resolution}px_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}\nRESUMEN — Track B @ {resolution}px ({len(SEEDS)} semillas)\n{'=' * 70}")
    for name in ["accuracy", "balanced_accuracy", "precision_abnormal", "recall_abnormal",
                 "f1_abnormal", "f1_macro", "auc_roc", "specificity"]:
        stat = aggregated[name]
        print(f"  {name:22s} {stat['mean']:.4f} ± {stat['std']:.4f}")
    print(f"\nHiperparametros por semilla:")
    for r in per_seed_results:
        print(f"  seed {r['seed']}: {r['best_params']}")
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
