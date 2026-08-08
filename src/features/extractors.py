"""crea EfficientNetV2B0 y la utiliza para convertir cada imagen en un vector de 1.280 características"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import timm
import torch
import cv2

from src.data.preprocessing import load_normalization_stats, preprocess_batch, INPUT_SIZE
from src.data.data_augmentation import generate_augmented_batch


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

    chunks = []

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

    # Libera la VRAM cacheada por PyTorch -- si no, CatBoost (GPU) ve menos memoria
    # libre de la real y puede quedarse sin memoria al usar depths mas altos.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.concatenate(chunks, axis=0) #devovlemos los lotes concatenados


def get_or_compute_features(paths: list[Path], resolution: int, device: torch.device, batch_size: int = 256,) -> np.ndarray:
    """Devuelve las features de una resolucion, reutilizando el cache para evitar repetir trabajo.
    Las features solo se calculan una vez por resolución"""

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FEATURES_DIR / f"{BACKBONE_NAME}_{resolution}px.npy"

    if cache_path.exists(): #si ya existe el archivo
        # mmap: no carga el array entero en RAM, el SO pagina bajo demanda y puede liberarlo si hace falta memoria
        features = np.load(cache_path, mmap_mode="r") #Carga las características previamente calculadas

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
    del features #libera la copia en RAM: se relee mapeada a disco para no duplicar el array completo
    features = np.load(cache_path, mmap_mode="r")
    print(f"Features guardadas en {cache_path} {features.shape}")
    return features












############ DATA AUGMENTATION
#extraccion de features aumentadas


AUGMENTATION_SEED = 123   #cualquiera que no sea la misma seed de particiones

K_AUGMENTED_VIEWS = 3

#mover a processing.py(?)
def _preprocess_augmented_array(image_rgb: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """procesa una imagen hecha array"""
    image = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    image = image.astype(np.float32) / 255.0
    image_norm = (image - mean) / std

    
    # HWC -> CHW    012 201
    image = image_norm.transpose(2,0,1) #puede que no reorganizar en memoria

    image = np.ascontiguousarray(image) #aseguramos que se puede convertir a tensor de pytorch
    return image


@torch.inference_mode()
def extract_augmented_features(paths: list[Path], resolution: int, device: torch.device,
                                k: int = K_AUGMENTED_VIEWS, batch_size: int = 256,) -> np.ndarray:
    """ extract_features() pero para las vistas aumentadas
    cambio: devuelve un array (N, k, 1280) (antes N, 1280)"""
    mean, std = load_normalization_stats(resolution)
    model = build_backbone(device)

    rng = np.random.default_rng(AUGMENTATION_SEED)
    chunks = []
    
    for start in range(0, len(paths), batch_size):#Recorre paths en bloques de batch_size
        
        batch_paths = paths[start : start + batch_size]
        aug_batch = generate_augmented_batch(batch_paths, rng, k)

        preprocessed_images = []

        for views in aug_batch:
            preprocessed_views = []

            for view in views:
                image_array = _preprocess_augmented_array(view, mean, std)

                preprocessed_views.append(image_array)

            preprocessed_images.append(preprocessed_views)

        preprocessed_batch = np.stack(preprocessed_images, axis=0)

        number_of_images = preprocessed_batch.shape[0]

        number_of_views = preprocessed_batch.shape[1]
        flattened_batch = preprocessed_batch.reshape(number_of_images * number_of_views, 3, INPUT_SIZE, INPUT_SIZE)

        # conviertimos el array de numpy en un tensor de PyTorch
        tensor = torch.from_numpy(flattened_batch).to(device, non_blocking=True)
    
        #tensor.shape
    
        feats = model(tensor) #pasamos las imágenes por el modelo
        #feats.shape
    
        #volver a formato array numpy para futuro catboost
        features = feats.float().cpu().numpy()
        number_of_features = features.shape[1]

        features = features.reshape(number_of_images, number_of_views, number_of_features)


        chunks.append(features)
    
        done = min(start + batch_size, len(paths))
        print(f"  features {done}/{len(paths)}", end="\r", flush=True)
    
    print()
    
        
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    features = np.concatenate(chunks, axis=0)
    
    return features #devovlemos los lotes concatenados

    


def get_or_compute_augmented_features(paths: list[Path], resolution: int, device: torch.device, 
                                      k: int = K_AUGMENTED_VIEWS, batch_size: int = 256,) -> np.ndarray:
    """Devuelve las features de las vistas aumentadas.

    Reutiliza el cache si ya existe y tiene la forma correcta (numero_imagenes, k, 1280)"""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FEATURES_DIR / f"{BACKBONE_NAME}_{resolution}px_aug{k}.npy"


    shape = (len(paths), k, 1280)
    
    if cache_path.exists(): #si ya existe el archivo
        # mmap: no carga el array entero en RAM, el SO pagina bajo demanda y puede liberarlo si hace falta memoria
        features = np.load(cache_path, mmap_mode="r") #Carga las características previamente calculadas

         # El cache solo es valido si cubre exactamente el inventario actual.
        if features.shape != shape:
            raise ValueError(
                f"El cache {cache_path} tiene {features.shape[0]} filas pero el inventario "
                f"de {resolution}px tiene {len(paths)} imagenes. Borra el cache y reextrae."
            )
        print(f"Features cargadas del cache: {cache_path} {features.shape}")
        return features

    print(f"Extrayendo features de {len(paths)} imagenes a {resolution}px ({BACKBONE_NAME})...")
    features = extract_augmented_features(paths, resolution, device, k, batch_size)
    np.save(cache_path, features)
    del features #libera la copia en RAM: se relee mapeada a disco para no duplicar el array completo
    features = np.load(cache_path, mmap_mode="r")
    print(f"Features guardadas en {cache_path} {features.shape}")

    return features
