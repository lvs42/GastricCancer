"""calculo de métricas, la matriz de confusión, intervalos de confianza y resultados medios 
entre semillas de la clasificación binaria """

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

POS_LABEL = 1

N_BOOTSTRAP = 1000


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Calcula el bloque completo de metricas de test"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        #porcentaje total de aciertos
        "accuracy": float(accuracy_score(y_true, y_pred)),
        
        #promedia el rendimiento entre ambas clases (mejor cuando desbalanceo)
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),

        # De todas las imágenes que el modelo clasificó como Abnormal, cuantas eran realmente Abnormal?
        "precision_abnormal": float(precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        
        #De todas las imágenes realmente Abnormal, cuantas detectó el modelo?
        "recall_abnormal": float(recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        
        #  combina presición y recall. alta solo cuando ambas son buenas
        "f1_abnormal": float(f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),

        # promedia ambas clases con el mismo peso, sin que la mayoritaria domine.
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),

        # AUC probabilidad
        "auc_roc": float(roc_auc_score(y_true, y_score)),

        # De todas las imágenes realmente normales, cuantas identificó correctamente como normales?
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "labels": ["Normal(0)", "Abnormal(1)"],
        },
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, seed: int, n_resamples: int = N_BOOTSTRAP, alpha: float = 0.05,) -> dict:
    """Intervalos de confianza del 95%
    Boostrap repetido 100 veces con reemplazo
    (no vuevlve a predecir, reemeustrea las predicciones ya obtendias)"""

    rng = np.random.default_rng(seed)
    n = len(y_true)

    metrics: dict[str, list[float]] = {
        "accuracy": [], "balanced_accuracy": [], "f1_abnormal": [], "auc_roc": [],
    }

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]

        # Si solo uan clase hace indefinidos AUC y balanced accuracy.
        if len(np.unique(yt)) < 2:
            continue

        yp, ys = y_pred[idx], y_score[idx]


        metrics["accuracy"].append(accuracy_score(yt, yp))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        metrics["f1_abnormal"].append(f1_score(yt, yp, pos_label=POS_LABEL, zero_division=0))
        metrics["auc_roc"].append(roc_auc_score(yt, ys))

    #percentiles 2.5 abajo 97.5 arriba para un itnerval ode 95%
    lo_q = 100 * (alpha / 2)
    hi_q = 100 * (1 - alpha / 2)

    confidence_intervals = {}

    for name, vals in metrics.items():
        lower_percentile = np.percentile(vals, lo_q)
        upper_percentile = np.percentile(vals, hi_q)

        lower_limit = float(lower_percentile)
        upper_limit = float(upper_percentile)

        number_of_valid_resamples = len(vals)

        metric_confidence_interval = {"ci95_low": lower_limit,  "ci95_high": upper_limit, "n_valid_resamples": number_of_valid_resamples}

        confidence_intervals[name] = metric_confidence_interval


    return confidence_intervals


def aggregate_across_seeds(per_seed: list[dict]) -> dict:
    """Recibir resultados de las semillas y caclula media std y values"""

    scalar_keys = []
    first_seed_results = per_seed[0]

    for key, value in first_seed_results.items():
        is_integer = isinstance(value, int)
        is_float = isinstance(value, float)
        is_scalar = (is_integer or is_float)

        if is_scalar:
            scalar_keys.append(key)


    summary = {}

    for key in scalar_keys: #para cada metrica calula mean std y value
        key_values = []

        for run in per_seed:
            value = run[key]

            float_value = float(value)
            key_values.append(float_value)


        mean_value = np.mean(key_values)
        mean_value = float(mean_value)

        standard_deviation = np.std(key_values)
        standard_deviation = float(standard_deviation)

        key_summary = {
            "mean": mean_value,
            "std": standard_deviation,
            "values": key_values
        }

        summary[key] = key_summary

    return summary
