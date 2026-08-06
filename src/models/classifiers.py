"""Balanceo de datos, probar configuraciones de CatBoost y entrenar el modelo finla"""

from __future__ import annotations

from itertools import product

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold



PARAM_GRID: dict[str, list] = {
    "iterations": [300, 500, 700], #número de árboles que construye CatBoost
    "learning_rate": [0.05, 0.1, 0.2],
    "depth": [4, 6, 8],
}

N_FOLDS = 5

# el grid = 27 combinaciones de configuraciones x 5 folds x 5 semillas = 675 entrenamientos


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Calcula pesos para compensar el desbalance
    Equivocarse en una imagen abnormal penaliza más que equivocarse en una normal"""

    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    return {int(c): float(w) for c, w in zip(classes, weights)}

######### 
# Probar tmb balanceo de clases por oversampling
#########

def catboost_task_type() -> str:
    """Usa GPU para CatBoost si hay una disponible -> más rápido"""
    return "GPU" if torch.cuda.is_available() else "CPU"


def build_classifier(params: dict, class_weights: dict[int, float], seed: int) -> CatBoostClassifier:
    """Crea una instancia de CatBoost con unos hiperparametros y pesos de clase dados"""
    return CatBoostClassifier(
        #iterations=500,
        #learning_rate=0.1,
        #depth=6,
        **params, #params = {"iterations": 500, "learning_rate": 0.1, "depth": 6,}
        task_type=catboost_task_type(),
        # CatBoost espera los pesos en orden de clase ascendente (0 = Normal y 1 = Abnormal).
        class_weights=[class_weights[0], class_weights[1]],
        loss_function="Logloss",
        random_seed=seed,
        verbose=False, #evita prints que hacen ir más lenta
        allow_writing_files=False,
    )


def tune_hyperparameters(x_train: np.ndarray, y_train: np.ndarray, seed: int,) -> tuple[dict, dict]:
    """Selecciona hiperparametros sobre train, sin evaluar co ntest"""

    class_weights = compute_class_weights(y_train) #Calcula los pesos usando el conjunto de entrenamiento
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed) # divide en 5 particiones para validación cruzada

    folds = list(skf.split(x_train, y_train)) # 5 folds

    grid = []

    for iterations in [300, 500, 700]: #creamos las 27 combianciones de configuraciones
        for learning_rate in [0.05, 0.1, 0.2]:
            for depth in [4, 6, 8]:
                params = {
                    "iterations": iterations,
                    "learning_rate": learning_rate,
                    "depth": depth,
                }

                grid.append(params)
     
    results: list[dict] = []

    for i, params in enumerate(grid, start=1): #evaluar cada configuracioens, recorremso las 27
        fold_scores: list[float] = []
        for train_fold, val_fold in folds:
            model = build_classifier(params, class_weights, seed)
            model.fit(x_train[train_fold], y_train[train_fold]) #entrena con 4 folds
            y_pred = model.predict(x_train[val_fold]) #predice con 1 fold restante
            
            #metrica balanced accuracy porque por ahora el data set se encuentra desbaalnceado
            fold_scores.append(balanced_accuracy_score(y_train[val_fold], y_pred))#guarda el accuracy balanceado resultante del fold restante


        #almacenamos resultados de cada configuracion
        entry = {
            "params": params,
            "cv_balanced_accuracy_mean": float(np.mean(fold_scores)),
            "cv_balanced_accuracy_std": float(np.std(fold_scores)),
            "cv_fold_scores": [float(s) for s in fold_scores],
        }

        
        results.append(entry)
        print(
            f"  [{i}/{len(grid)}] {params} -> "
            f"CV balanced acc {entry['cv_balanced_accuracy_mean']:.4f} "
            f"± {entry['cv_balanced_accuracy_std']:.4f}"
        )

    best = max(results, key=lambda r: r["cv_balanced_accuracy_mean"]) #mejor accuracy medio
    cv_info = {
        "n_folds": N_FOLDS,
        "selection_metric": "balanced_accuracy",
        "n_configurations": len(grid),
        "best_params": best["params"],
        "best_cv_balanced_accuracy_mean": best["cv_balanced_accuracy_mean"],
        "best_cv_balanced_accuracy_std": best["cv_balanced_accuracy_std"],
        "best_cv_fold_scores": best["cv_fold_scores"],
        "all_results": results,
    }
    return best["params"], cv_info


def fit_final_model(x_train: np.ndarray, y_train: np.ndarray, params: dict, seed: int,) -> CatBoostClassifier:
    """Una vez elegidos los mejores parámetros reentrena el modelo final utilizando todo el 80% de train"""
    class_weights = compute_class_weights(y_train)

    model = build_classifier(params, class_weights, seed)
    model.fit(x_train, y_train)
    
    return model







############ DATA AUGMENTATION
# las vistas aumentadas SOLO al lado de entrenamiento de cada fold


def tune_hyperparameters_augmentation( x_train: np.ndarray, y_train: np.ndarray,aug_features_train: np.ndarray,  seed: int,) -> tuple[dict, dict]:
    """el lado FIT se amplia con las vistas aumentadas de las imagenes de ese fold
    de fit (val fold usa solo originales"""

    #class_weights = compute_class_weights(y_train) #Calcula los pesos usando el conjunto de entrenamiento
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed) # divide en 5 particiones para validación cruzada

    folds = list(skf.split(x_train, y_train)) # 5 folds

    grid = []

    for iterations in [300, 500, 700]: #creamos las 27 combianciones de configuraciones
        for learning_rate in [0.05, 0.1, 0.2]:
            for depth in [4, 6, 8]:
                params = {
                    "iterations": iterations,
                    "learning_rate": learning_rate,
                    "depth": depth,
                }

                grid.append(params)

     
    results = []

    k = aug_features_train.shape[1]
    number_of_features = aug_features_train.shape[2]

    for i, params in enumerate(grid, start=1): #evaluar cada configuracioens, recorremso las 27
        fold_scores: list[float] = []
        for train_fold, val_fold in folds:


            normal = x_train[train_fold]
            augmented = aug_features_train[train_fold]
            augmented = augmented.reshape(-1, number_of_features)
            
            x_fit = np.concatenate([normal, augmented,],axis=0)

            labels = y_train[train_fold]
            augmented_labels = np.repeat(labels, k)

            y_fit = np.concatenate([labels, augmented_labels,],axis=0)



            x_validation = x_train[val_fold]
            y_validation = y_train[val_fold]

            class_weights = compute_class_weights(y_fit)

            model = build_classifier(params, class_weights, seed)
            model.fit(x_fit, y_fit)
            y_pred = model.predict(x_train[val_fold]) #predice con 1 fold restante
            
            #metrica balanced accuracy porque por ahora el data set se encuentra desbaalnceado
            fold_scores.append(balanced_accuracy_score(y_train[val_fold], y_pred))#guarda el accuracy balanceado resultante del fold restante


        #almacenamos resultados de cada configuracion
        entry = {
            "params": params,
            "cv_balanced_accuracy_mean": float(np.mean(fold_scores)),
            "cv_balanced_accuracy_std": float(np.std(fold_scores)),
            "cv_fold_scores": [float(s) for s in fold_scores],
        }

        
        results.append(entry)
        print(
            f"  [{i}/{len(grid)}] {params} -> "
            f"CV balanced acc {entry['cv_balanced_accuracy_mean']:.4f} "
            f"± {entry['cv_balanced_accuracy_std']:.4f}"
        )

    best = max(results, key=lambda r: r["cv_balanced_accuracy_mean"]) #mejor accuracy medio
    cv_info = {
        "n_folds": N_FOLDS,
        "selection_metric": "balanced_accuracy",
        "n_configurations": len(grid),
        "best_params": best["params"],
        "best_cv_balanced_accuracy_mean": best["cv_balanced_accuracy_mean"],
        "best_cv_balanced_accuracy_std": best["cv_balanced_accuracy_std"],
        "best_cv_fold_scores": best["cv_fold_scores"],
        "all_results": results,
    }
    return best["params"], cv_info


def fit_final_model_augmentation(x_train: np.ndarray, y_train: np.ndarray, aug_features_train: np.ndarray, params: dict, seed: int,) -> CatBoostClassifier:
    """entrena sobre el 80% completo + sus vistas aumentadas"""
    k = aug_features_train.shape[1]
    number_of_features = aug_features_train.shape[2]

    augmented = aug_features_train.reshape(-1, number_of_features)
    augmented_labels = np.repeat(y_train, k)

    x_fit = np.concatenate([x_train, augmented], axis=0)
    y_fit = np.concatenate([y_train, augmented_labels], axis=0)


    class_weights = compute_class_weights(y_fit)
    
    model = build_classifier(params, class_weights, seed)
    #print(x_fit.shape)
    model.fit(x_fit, y_fit)
        
    return model
