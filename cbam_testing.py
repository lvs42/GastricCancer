
import torch
from src.data.loaders import load_inventory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths, y = load_inventory(80, strict_counts=False)

# 15 abnormal y 15 normal
idx_abnormal = [i for i, label in enumerate(y) if label == 1][:15]
idx_normal = [i for i, label in enumerate(y) if label == 0][:15]
idx_prueba = idx_abnormal + idx_normal

paths_prueba = [paths[i] for i in idx_prueba]
y_prueba = y[idx_prueba]

print(f"{len(paths_prueba)} imagenes de prueba, clases: {set(y_prueba.tolist())}")


from src.features.extractors import build_backbone
from src.data.preprocessing import load_normalization_stats
from src.features.cbam_extractors import get_raw_feature_maps

mean, std = load_normalization_stats(80)
backbone = build_backbone(device)

mapas = get_raw_feature_maps(backbone, paths_prueba[:4], mean, std, device)
print("Forma del mapa:", mapas.shape)   # esperado: (4, 1280, 7, 7) aprox.




from src.features.cbam_extractors import train_cbam
import src.features.cbam_extractors as cbam_mod
cbam_mod.MAX_EPOCHS = 2   # solo para que la prueba sea rapida

cbam_entrenado = train_cbam(paths_prueba, y_prueba, 80, device, seed=42, batch_size=8)
print("Entrenamiento terminado sin errores.")




from src.features.cbam_extractors import extract_cbam_features

features = extract_cbam_features(paths_prueba, 80, device, backbone, cbam_entrenado, batch_size=8)
print("Forma de las features:", features.shape)   # esperado: (30, 1280)

import numpy as np
from src.features.cbam_extractors import get_or_compute_cbam_features

train_idx_prueba = np.arange(20)   # las primeras 20 de las 30 como "train" ficticio
features_completas = get_or_compute_cbam_features(
    paths_prueba, y_prueba, train_idx_prueba, 80, device, seed=42, batch_size=8
)
print(features_completas.shape)   # esperado: (30, 1280)

# Ejecutalo una SEGUNDA vez y comprueba que esta vez carga del cache
# (deberia imprimir "Cache CBAM encontrado", no volver a entrenar)
features_otra_vez = get_or_compute_cbam_features(
    paths_prueba, y_prueba, train_idx_prueba, 80, device, seed=42, batch_size=8
)
assert np.array_equal(features_completas, features_otra_vez)
print("Cache funciona correctamente.")