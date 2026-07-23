from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

# tf_efficientnetv2_b0 se entreno a 224x224 (pero usa 192 para train??) https://huggingface.co/timm/tf_efficientnetv2_b0.in1k
# Los parches de GasHisSDB son de 80/120/160 px -> se hace upsampling.
INPUT_SIZE = 224

current_file = Path(__file__)

absolute_file_path = current_file.resolve()
project_root = absolute_file_path.parents[2]

configs_directory = project_root / "configs"

CONFIG_PATH = configs_directory / "normalization_stats.yaml"



def load_normalization_stats(resolution: int,) -> tuple[np.ndarray, np.ndarray]:
    """Carga la media y desviación RGB de una resolución."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        stats = yaml.safe_load(file)


    if f"gashisdb_{resolution}" not in stats:
        raise KeyError(
            f"No hay estadísticas de normalización para la resolución "
            f"{resolution} en {CONFIG_PATH}. "
            f"Claves disponibles: {sorted(stats)}"
        )

    entry = stats[f"gashisdb_{resolution}"]

    if int(entry["resolution"]) != int(resolution):
        raise ValueError(
            f"La sección 'gashisdb_{resolution}' declara "
            f"resolution={entry['resolution']}, "
            f"pero se ha solicitado la resolución {resolution}."
        )


    mean = np.asarray(entry["mean"], dtype=np.float32,)
    std = np.asarray(entry["std"], dtype=np.float32,)

    if mean.shape != (3,):
        raise ValueError(
            f"La media debe tener tres canales RGB. "
            f"Forma recibida: {mean.shape}"
        )

    if std.shape != (3,):
        raise ValueError(
            f"La desviación debe tener tres canales RGB. "
            f"Forma recibida: {std.shape}"
        )

    return mean, std


def load_image_rgb(path: Path) -> np.ndarray:
    """Lee un PNG y devuelve los pixeles en RGB."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR) # imread devuelve bgr NO rgb!!

    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) #ordena correctamete los canales de colores para ser RGB


def preprocess_batch(paths: list[Path], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """procesa un lote de imágenes, no procesa todo el dataset de golpe"""

    batch = np.empty((len(paths), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)#reserva memoria para el lote en formato NHWC
    #NHWC = tamaño de lote, altura, ancho, canales rgb =3

    for i, path in enumerate(paths):
        rgb = load_image_rgb(path)
        # redimensiona imagenes a tamaño de tf_efficientnetv2_b0
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR) #calcula los nuevos píxeles interpolando los píxeles originales
        #resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        
        #normaliza imagen (convierte pixles a [0,1])
        batch[i] = resized.astype(np.float32) / 255.0

    
    batch = (batch - mean) / std
    # Pythorch espera NCHW 
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2)) #NHWC -> NCHW
