""" python -m src.training.run_track_b --resolution 80
python -m src.training.run_track_b --resolution 120
python -m src.training.run_track_b --resolution 160 """

from __future__ import annotations

import argparse
import gc
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import multiprocessing
import traceback
from queue import Empty

import time


def _current_rss_gb() -> float:
    """RAM residente actual del proceso (para ver si se acumula memoria entre semillas)."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 ** 2)  # kB -> GB
    return float("nan")

from src.data.loaders import (
    VALID_RESOLUTIONS,
    load_inventory,
    split_train_test,
    validate_resolution,
)
from src.evaluation.metrics import aggregate_across_seeds, bootstrap_ci, compute_metrics
from src.features.extractors import BACKBONE_NAME, get_or_compute_features, get_or_compute_augmented_features, K_AUGMENTED_VIEWS
from src.models.classifiers import PARAM_GRID, fit_final_model, tune_hyperparameters, fit_final_model_augmentation, tune_hyperparameters_augmentation

# Cada semilla generará su propio split train/test
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

TEST_SIZE = 0.2 #20%
SEED_TIMEOUT_SECONDS = 6 * 60 * 60 #margen de seis horas por semilla


current_file = Path(__file__)

absolute_file_path = current_file.resolve()
project_root = absolute_file_path.parents[2]

RESULTS_DIR = project_root / "results"


def run_seed(features: np.ndarray, y: np.ndarray, seed: int, aug_features = None) -> tuple[dict, dict]:
    """Un experimento completo para una semilla"""

    print(f"\n{'=' * 70}\nSEMILLA {seed}\n{'=' * 70}")
    print(f"RAM residente al empezar la semilla: {_current_rss_gb():.2f} GB")

    # separar train y test
    train_idx, test_idx = split_train_test(y, seed=seed, test_size=TEST_SIZE)
    x_train = features[train_idx]
    y_train = y[train_idx]

    x_test = features[test_idx]
    y_test = y[test_idx]

    print(
        f"Train: {len(y_train)} (Abnormal {int(y_train.sum())}) | "
        f"Test: {len(y_test)} (Abnormal {int(y_test.sum())})"
    )

    if aug_features is None:
    
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

    else:
        aug_features_train = aug_features[train_idx]
        print(f"\nTuning por CV (k=5) + daat AUGMENTATION sobre el 80% de train...")

        best_params, cv_info = tune_hyperparameters_augmentation(x_train, y_train,aug_features_train, seed=seed)

        print(
            f"\nMejores hiperparametros: {best_params}\n"
            f"CV balanced accuracy: {cv_info['best_cv_balanced_accuracy_mean']:.4f} "
            f"± {cv_info['best_cv_balanced_accuracy_std']:.4f}"
        )


        print("\nReentrenando modelo final sobre el 80% completo + vistas aumentadas")
        model = fit_final_model_augmentation(x_train, y_train, aug_features_train, best_params, seed=seed)

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

    # liberar explicitamente antes de pasar a la siguiente semilla: CatBoost en GPU no siempre
    # libera sus buffers internos solo por refcounting entre llamadas sucesivas a .fit()
    del x_train, y_train, x_test, y_test, model, y_pred, y_score
    if aug_features is not None:
        del aug_features_train
    gc.collect()
    print(f"RAM residente al terminar la semilla: {_current_rss_gb():.2f} GB")

    return seed_result, flat



def _run_seed_worker(resolution, batch_size, experiment, k, seed, strict_counts, result_queue):
    """Ejecuta una semilla completa como un proceso independiente"""

    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        paths, y = load_inventory(resolution, strict_counts=strict_counts)
        features = get_or_compute_features(paths, resolution, device, batch_size)

        aug_features = None

        if experiment == "augmentation":
            aug_features = get_or_compute_augmented_features(paths, resolution, device, k, batch_size)
        seed_result, flat = run_seed(features, y, seed, aug_features=aug_features)

        result_queue.put(("OK", seed_result, flat,))

    except Exception as error:
        error_message = (f"{type(error).__name__}: {error}")
        error_traceback = traceback.format_exc()

        result_queue.put(("ERROR", error_message, error_traceback,))


def run_seeds_in_separate_processes(resolution, batch_size, experiment, k, strict_counts):
    """Ejecuta cada semilla en un proceso independiente"""

    context = multiprocessing.get_context("spawn")


    per_seed_results = []
    per_seed_flat = []

    for seed in SEEDS:
        print()
        print("=" * 70)
        print(f"CREANDO PROCESO PARA LA SEMILLA {seed}")
        print("=" * 70)

        result_queue = context.Queue()

        process = context.Process(target=_run_seed_worker, args=(
                                                                    resolution,
                                                                    batch_size,
                                                                    experiment,
                                                                    k,
                                                                    seed,
                                                                    strict_counts,
                                                                    result_queue,
                                                                ),
        )

        process.start()

        start_time = time.monotonic()
        message = None

        while message is None:

            try:
                message = result_queue.get(timeout=5)

            except Empty:
                elapsed_time = (time.monotonic() - start_time)

                if not process.is_alive():
                    process.join()

                    exit_code = process.exitcode

                    result_queue.close()
                    result_queue.join_thread()
                    process.close()

                    raise RuntimeError(
                        f"La semilla {seed} terminó sin "
                        f"devolver resultados. "
                        f"Exitcode: {exit_code}. "
                        f"Puede haber sido terminada "
                        f"por falta de memoria."
                    )

                if elapsed_time > SEED_TIMEOUT_SECONDS:
                    process.terminate()
                    process.join()

                    result_queue.close()
                    result_queue.join_thread()
                    process.close()

                    raise RuntimeError(f"La semilla número {seed} superó el límite de segundos especificado")

        process.join()

        exit_code = process.exitcode

        result_queue.close()
        result_queue.join_thread()
        process.close()

        status = message[0]

        if status == "ERROR":
            error_message = message[1]
            error_traceback = message[2]

            raise RuntimeError(
                f"La semilla número {seed} produjo un error.\n"
                f"Exitcode: {exit_code}\n"
                f"{error_message}\n\n"
                f"{error_traceback}" #que pasooo
            )

        if status != "OK":
            raise RuntimeError(f"Estado desconocido recibido para la semilla {seed}: {status}")


        seed_result = message[1]
        flat = message[2]

        per_seed_results.append(seed_result)
        per_seed_flat.append(flat)

        print(
            f"\nProceso de la semilla {seed} finalizado. "
            f"MEMORIA LIBERADA"
        )

    return per_seed_results, per_seed_flat



##############################################################################

###### MAIN #########################################################

##############################################################################

def main() -> None:
    parser = argparse.ArgumentParser() #permite recibir parámetros por terminal

    parser.add_argument("--resolution", type=int, required=True, choices=VALID_RESOLUTIONS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-strict-counts", action="store_true")
    parser.add_argument("--experiment",choices=["baseline","augmentation",],default="baseline")
    parser.add_argument("--k",type=int,default=K_AUGMENTED_VIEWS)

    args = parser.parse_args()


    resolution = validate_resolution(args.resolution)



     
    if torch.cuda.is_available(): #usa gpu si se puede, sino cpu
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # Mostrar información del experimento.
    separator = "=" * 70

    print(separator)
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
    #features = get_or_compute_features(paths, resolution, device, batch_size=args.batch_size)

    strict_counts = not args.no_strict_counts
    per_seed_results, per_seed_flat = (
        run_seeds_in_separate_processes(
            resolution=resolution,
            batch_size=args.batch_size,
            experiment=args.experiment,
            k=args.k,
            strict_counts=strict_counts
        )
    )

    
    aggregated = aggregate_across_seeds(
        per_seed_flat
    )

    """
    aug_features = None
    if args.experiment == "augmentation":
        aug_features = get_or_compute_augmented_features(
            paths, resolution, device, k=args.k, batch_size=args.batch_size
        )

    # Listas para guardar los resultados de todas las semillas.
    per_seed_results = []
    per_seed_flat = []

    # repetimos el experimento completo por cada semilla
    for seed in SEEDS:
        result, flat = run_seed(features, y, seed, aug_features=aug_features)
        per_seed_results.append(result)
        per_seed_flat.append(flat)

    aggregated = aggregate_across_seeds(per_seed_flat)
    """

    ############
    #RESULTADOS
    if args.experiment == "baseline":
        track_name = "Baseline"

        description = (
            "Baseline: "
            "EfficientNetV2B0 + CatBoost."
        )

        output_filename = (
            f"baseline_{resolution}px_results.json"
        )

        summary_name = "Baseline"

    else:
        track_name = "Augmentation"

        description = (
            "Experimento con data augmentation: "
            "EfficientNetV2B0 congelado + CatBoost. "
            "Las vistas aumentadas se usan únicamente en el entrenamiento de cada fold."
        )

        output_filename = (
            f"augmentation_{resolution}px_results.json"
        )

        summary_name = "Augmentation"

    output = {
        "track": track_name,
        "description": description,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "experiment": args.experiment,
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
            "augmentation_enabled": (args.experiment == "augmentation"),
            "augmentation_views_per_image": (
                args.k
                if args.experiment == "augmentation"
                else 0
            ),
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
    out_path = (RESULTS_DIR / output_filename)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}\n RESUMEN — {summary_name} @ {resolution}px ({len(SEEDS)} semillas)\n {'=' * 70}")
    
    for name in ["accuracy", "balanced_accuracy", "precision_abnormal", "recall_abnormal",
                 "f1_abnormal", "f1_macro", "auc_roc", "specificity"]:
        stat = aggregated[name]
        print(f"  {name:22s} {stat['mean']:.4f} ± {stat['std']:.4f}")
    print(f"\nHiperparametros por semilla:")
    for r in per_seed_results:
        print(f"  seed {r['seed']}: {r['best_params']}")
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
