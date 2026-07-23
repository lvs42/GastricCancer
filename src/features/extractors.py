"""crea EfficientNetV2B0 y la utiliza para convertir cada imagen en un vector de 1.280 características"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import timm
import torch

from src.data.preprocessing import load_normalization_stats, preprocess_batch


# se utilizan los pesos ImageNet originales de TensorFlow portados a PyTorch
BACKBONE_NAME = "tf_efficientnetv2_b0" # modelo proviene de tensorflow


current_file = Path(__file__)

absolute_file_path = current_file.resolve()
project_root = absolute_file_path.parents[2]

results_directory = project_root / "results"

FEATURES_DIR = results_directory / "features"


def build_backbone(device: torch.device) -> torch.nn.Module:

    model = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0, global_pool="avg")
    # se utilizan los pesos ImageNet originales de TensorFlow portados a PyTorch -> ya ha aprendido a dectectar: bordes, texturas,formas, colores y patrones complejos
    #num_classes = 0 -> elimina la clasificación final (el modelo devolvia clasificacion entre gato, perro, etc)
    # num classes + pooling hace que se devuelva vector de características

    # se cogela para que no se modifiquen los pesos, es decir que no aprenda con GasHisSDB
    for param in model.parameters():
        param.requires_grad = False

    return model.eval().to(device) #directamente modo evaluación, no se entrena la red neuronal
# to device par escoger GPU o CPU

#solo predecir tiene ventajas: no se necesitan calcular o guardar gradientes ni realizar la backpropagation


#####


@torch.inference_mode() #inference para marcar a hace predicciones
def extract_features(paths: list[Path], resolution: int, device: torch.device, batch_size: int = 256,) -> np.ndarray:

    """Recibe todas las rutas y devuelve todas las características
    Carga las estadísticas de la resolución especificada en os parámetros.
    Preprocesa las imágenes por lotes"""

    mean, std = load_normalization_stats(resolution)
    model = build_backbone(device)

    chunks: list[np.ndarray] = []

    for start in range(0, len(paths), batch_size):
    
        batch_paths = paths[start : start + batch_size]
        batch = preprocess_batch(batch_paths, mean, std)
        # conviertimos el array de numpy en un tensor de PyTorch
        tensor = torch.from_numpy(batch).to(device, non_blocking=True)

        #tensor.shape #(256, 3, 224, 224)

        feats = model(tensor) #pasamos las imágenes por el modelo
        #feats.shape #(256, 1280)

        #volver a formato array numpy para futuro catboost
        features = feats.float().cpu().numpy()
        chunks.append(features)

        done = min(start + batch_size, len(paths))
        print(f"  features {done}/{len(paths)}", end="\r", flush=True)

    print()
    return np.concatenate(chunks, axis=0) #devovlemos los lotes concatenados


def get_or_compute_features(paths: list[Path], resolution: int, device: torch.device, batch_size: int = 256,) -> np.ndarray:
    """Devuelve las features de una resolucion, reutilizando el cache para evitar repetir trabajo.
    Las features solo se calculan una vez por resolución"""

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FEATURES_DIR / f"{BACKBONE_NAME}_{resolution}px.npy"

    if cache_path.exists(): #si ya existe el archivo 
        features = np.load(cache_path) #Carga las características previamente calculadas

        # El cache solo es valido si cubre exactamente el inventario actual.
        if features.shape[0] != len(paths):
            raise ValueError(
                f"El cache {cache_path} tiene {features.shape[0]} filas pero el inventario "
                f"de {resolution}px tiene {len(paths)} imagenes. Borra el cache y reextrae."
            )
        print(f"Features cargadas del cache: {cache_path} {features.shape}")
        return features

    print(f"Extrayendo features de {len(paths)} imagenes a {resolution}px ({BACKBONE_NAME})...")
    features = extract_features(paths, resolution, device, batch_size)
    np.save(cache_path, features)
    print(f"Features guardadas en {cache_path} {features.shape}")
    return features
