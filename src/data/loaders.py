"""Carga del inventario de GasHisSDB y split estratificado sin fuga (Track B)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split



current_file = Path(__file__)

absolute_file_path = current_file.resolve()

project_root = absolute_file_path.parents[2]
dataset_directory = project_root / "dataset"
gashissdb_directory = dataset_directory / "GasHisSDB"

DATASET_ROOT = gashissdb_directory / "GasHisSDB"


# Unicas resoluciones validas. Son la MISMA fuente de WSIs recortada a distinta escala.
VALID_RESOLUTIONS: tuple[int, ...] = (80, 120, 160)

# asigna etiquetas binarias. Abnormal = clase positiva = 1. Normal = clase 0
CLASS_TO_LABEL: dict[str, int] = {"Normal": 0, "Abnormal": 1}


# Recuentos de comprobacion
EXPECTED_COUNTS: dict[int, dict[str, int]] = {
    80: {"Abnormal": 59151, "Normal": 87500},
    120: {"Abnormal": 24801, "Normal": 40460},
    160: {"Abnormal": 13124, "Normal": 20160},
}


def validate_resolution(resolution: int) -> int:
    """Valida que la resolucion pedida sea una sola de las tres soportadas. si se combinasen parches del mismo
      wsi podrían acabar tanto en el conjunto de etst como en el de train"""

    if isinstance(resolution, (list, tuple, set)):
        raise ValueError(
            f"Solo se admite UNA resolucion por ejecucion, recibido: {resolution}. "
            "Las tres resoluciones son la misma fuente de WSIs a distinta escala; "
            "combinarlas invalidaria el experimento."
        )
    resolution = int(resolution)
    if resolution not in VALID_RESOLUTIONS:
        raise ValueError(f"Resolucion {resolution} invalida.")
    return resolution


def load_inventory(resolution: int, strict_counts: bool = True) -> tuple[list[Path], np.ndarray]:
    """Lista todas las imagenes de una unica resolucion con su etiqueta binaria."""
    resolution = validate_resolution(resolution)
    res_dir = DATASET_ROOT / str(resolution) #construccion de la carpeta
    if not res_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de la resolucion: {res_dir}")

    paths: list[Path] = [] #rutas de las imágenes
    labels: list[int] = [] #nombre de las imagenes
    for class_name, label in sorted(CLASS_TO_LABEL.items()):
        class_files = sorted((res_dir / class_name).glob("*.png")) #busca archivos png
        if not class_files:
            raise FileNotFoundError(f"Sin imagenes en {res_dir / class_name}")
        paths.extend(class_files)
        labels.extend([label] * len(class_files))

    y = np.asarray(labels, dtype=np.int64)

    # Comprobacion de integridad contra los recuentos verificados en el EDA.
    if strict_counts and resolution in EXPECTED_COUNTS:
        for class_name, expected in EXPECTED_COUNTS[resolution].items():
            found = int((y == CLASS_TO_LABEL[class_name]).sum())
            if found != expected:
                raise ValueError(
                    f"Inventario inconsistente en {resolution}px / {class_name}: "
                    f"{found} imagenes, se esperaban {expected} (eda.ipynb seccion 2). "
                    "El dataset puede estar incompleto."
                )

    #todas las rutas deben colgar de la MISMA resolucion.
    assert all(p.parents[1].name == str(resolution) for p in paths), (
        "Se han mezclado imagenes de mas de una resolucion en el mismo inventario."
    )

    return paths, y


def split_train_test(
    y: np.ndarray, seed: int, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Split estratificado fijo train/test."""
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, #para divisiones de cuantos k folds en validación cruzada
        test_size=test_size,
        stratify=y,
        random_state=seed,
        shuffle=True,
    )
    return train_idx, test_idx
