"""
Entrena el modelo (backbone congelado + CatBoost) SOLO con datos de UNA
resolucion, y lo evalua directamente sobre las imagenes de OTRA resolucion,
sin reentrenar nada para la resolucion de evaluacion.

Ejemplo:

    entrenar con 80px
            |
            v
        CatBoost
            |
            v
    evaluar con 120px

Esto permite comprobar si lo aprendido a una escala de parche
se mantiene cuando cambia la resolucion.
"""

import argparse
import json

from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from src.data.loaders import ( VALID_RESOLUTIONS,load_inventory,split_train_test)
from src.evaluation.metrics import ( aggregate_across_seeds,compute_metrics)
from src.features.extractors import get_or_compute_features
from src.models.classifiers import fit_final_model


SEEDS = ( 42, 43, 44, 45, 46)
TEST_SIZE = 0.2


current_file = Path(__file__)
project_root = current_file.resolve().parents[2]

RESULTS_DIR = project_root / "results"


def load_best_params_per_seed(train_resolution):
    """ Carga los hiperparametros que ya fueron seleccionados durante el baseline para una resolucion """

    filename = ( f"baseline_{train_resolution}px_results.json")

    baseline_path = RESULTS_DIR / filename



    with open(baseline_path, "r", encoding="utf-8") as file:

        baseline_results = json.load(file)


    per_seed = baseline_results[ "per_seed"]




    best_params_per_seed = {}


    for seed_result in per_seed:

        seed = seed_result["seed"]

        best_params = seed_result["best_params"]


        best_params_per_seed[seed] = best_params


    return best_params_per_seed


def run_cross_size_pair(train_resolution, eval_resolution, device):
    """
    Entrena CatBoost utilizando una resolucion

    Despues evalua el MISMO modelo sobre todas las imagenesde otra resolucion
    """

    print()
    print("=" * 70)

    print(f"{train_resolution}px -> {eval_resolution}px")

    print("=" * 70)



    paths_train_res, y_train_res = load_inventory(train_resolution)


    features_train_res = get_or_compute_features(paths_train_res, train_resolution, device)



    paths_eval_res, y_eval_res = load_inventory( eval_resolution)

    features_eval_res = get_or_compute_features(paths_eval_res,  eval_resolution, device)


    best_params_per_seed = load_best_params_per_seed(train_resolution)


    # resultados completos
    per_seed_results = []


    #Solo metricas numericas para agregar
    per_seed_flat = []



    for seed in SEEDS:

        print()
        print(f"Semilla: {seed}")

        train_idx, test_idx = split_train_test(y_train_res, seed=seed, test_size=TEST_SIZE)


        x_train = features_train_res[ train_idx]


        y_train = y_train_res[ train_idx]

        params = best_params_per_seed[ seed]


        model = fit_final_model(x_train, y_train, params, seed=seed)


        #prediccion de clases
        y_pred = model.predict(features_eval_res)
        y_pred = y_pred.astype(np.int64)


        #probabilidades
        probabilities = model.predict_proba(features_eval_res)

        #Probabilidad de la clase positiva -abnormal
        y_score = probabilities[:, 1]


        metrics = compute_metrics(y_eval_res, y_pred,  y_score)


        print(f"Accuracy: {metrics['accuracy']:.4f}")

        print(f"Balanced accuracy: "f"{metrics['balanced_accuracy']:.4f}")


        seed_result = {"seed": seed, "best_params": params, "metrics": metrics}


        per_seed_results.append(seed_result)

        scalar_metrics = {}


        for metric_name, metric_value in metrics.items():

            if np.isscalar(metric_value):
                scalar_metrics[metric_name] = float(metric_value)


        per_seed_flat.append(scalar_metrics)



        del model


    aggregated = aggregate_across_seeds(per_seed_flat)



    result = {
        "train_resolution": train_resolution,
        "eval_resolution": eval_resolution,
        "aggregated_across_seeds": aggregated,
        "per_seed": per_seed_results
    }


    return result


def main():

    parser = argparse.ArgumentParser( description=("Generalizacion cross-size entre resoluciones de GasHisSDB"))

    args = parser.parse_args()



    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")


    print(f"Dispositivo: {device}")


    
    resolution_pairs = permutations(VALID_RESOLUTIONS, 2)
    resolution_pairs = list(resolution_pairs)


    print()
    print("Combinaciones cross-size:")


    for pair in resolution_pairs:
        print(pair)



    all_results = []



    for train_resolution, eval_resolution in resolution_pairs:

        pair_result = run_cross_size_pair(train_resolution, eval_resolution, device)


        all_results.append(pair_result)


   
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


    output_path = RESULTS_DIR / ("cross_size_results.json")


    with open( output_path,  "w", encoding="utf-8") as file:

        json.dump(all_results, file, indent=4)



    print()
    print(f"Resultados guardados en: {output_path}")




    summary = {}



    for resolution in VALID_RESOLUTIONS:
        summary[resolution] = {}


    for result in all_results:

        train_resolution = result["train_resolution"]
        eval_resolution = result["eval_resolution"]
        aggregated = result["aggregated_across_seeds"]


        mean_accuracy = aggregated["accuracy"]["mean"]


        summary[train_resolution][eval_resolution] = mean_accuracy


    print()
    print("=" * 70)
    print( "RESUMEN CROSS-SIZE")
    print("=" * 70)


    print("Train/Eval", end="\t")


    for eval_resolution in VALID_RESOLUTIONS:
        print(f"{eval_resolution}px", end="\t")


    print()

    for train_resolution in VALID_RESOLUTIONS:
        print(f"{train_resolution}px", end="\t\t")


        for eval_resolution in VALID_RESOLUTIONS:
            

            if train_resolution == eval_resolution:
                print("-", end="\t")

                continue


            accuracy = summary[train_resolution][eval_resolution]


            print(f"{accuracy:.4f}", end="\t")


        print()



if __name__ == "__main__":

    main()