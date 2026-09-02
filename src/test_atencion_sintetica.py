"""
test_atencion_sintetica.py - diagnostico rapido: comprueba si el patron de
"esquinas calientes" visto en las visualizaciones reales es un artefacto
posicional (independiente del contenido de la imagen) o responde de verdad
al contenido histologico.

Genera una imagen completamente uniforme (sin ninguna estructura real) y
calcula su mapa de atencion espacial de CBAM con el mismo pipeline exacto
que usan las imagenes reales (mismo resize, misma normalizacion). Si el
patron de esquinas persiste incluso sin contenido, confirma que es un
artefacto de la arquitectura (el kernel 7x7 de SpatialAttention sobre un
mapa de solo ~7x7 hace que las celdas de esquina tengan su campo receptivo
dominado por padding, no por datos reales) - no algo aprendido sobre tejido.

Script de un solo uso para depurar esta hipotesis - no forma parte del
pipeline. Puedes borrarlo o moverlo a una carpeta de scripts de debug una
vez tengas la respuesta.
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.preprocessing import INPUT_SIZE, load_normalization_stats
from src.features.extractors import build_backbone
from src.visualization.visualize import load_trained_cbam

RESOLUTION = 160
SEED = 42


def make_synthetic_image(resolution: int, value: int) -> np.ndarray:
    """Imagen RGB completamente uniforme, sin ninguna estructura."""
    return np.full((resolution, resolution, 3), value, dtype=np.uint8)


def compute_attention_from_array(
    image_rgb: np.ndarray, backbone, cbam, mean: np.ndarray, std: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Igual que compute_spatial_attention_map de visualize.py, pero parte
    de un array ya en memoria en vez de una ruta a disco - mismo pipeline
    de preprocesado exacto (resize -> escalar a [0,1] -> normalizar ->
    NHWC->NCHW), para que la comparacion con las imagenes reales sea justa."""
    resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    scaled = resized.astype(np.float32) / 255.0
    normalized = (scaled - mean) / std
    chw = normalized.transpose(2, 0, 1)
    batch = np.ascontiguousarray(chw[np.newaxis, ...])  # (1, 3, H, W)

    tensor = torch.from_numpy(batch).float().to(device)

    with torch.no_grad():
        feature_map = backbone.forward_features(tensor)
        channel_weights = cbam.channel_attention(feature_map)
        x_channel = feature_map * channel_weights
        spatial_weights = cbam.spatial_attention(x_channel)

    return spatial_weights[0, 0].cpu().numpy()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = build_backbone(device)
    cbam = load_trained_cbam(RESOLUTION, SEED, device)
    mean, std = load_normalization_stats(RESOLUTION)

    # Tres imagenes uniformes de valores distintos, por si el nivel de gris
    # concreto importara (no deberia, pero es gratis comprobarlo).
    valores_prueba = [0, 128, 255]

    fig, axes = plt.subplots(1, len(valores_prueba), figsize=(4 * len(valores_prueba), 4))

    for ax, valor in zip(axes, valores_prueba):
        imagen = make_synthetic_image(RESOLUTION, valor)
        mapa = compute_attention_from_array(imagen, backbone, cbam, mean, std, device)

        print(f"\n=== Imagen uniforme, valor={valor} ===")
        print(np.round(mapa, 4))
        print(f"Min: {mapa.min():.4f}  Max: {mapa.max():.4f}  Rango: {mapa.max() - mapa.min():.4f}")

        im = ax.imshow(mapa, cmap="jet")
        ax.set_title(f"Valor uniforme = {valor}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    output_path = "test_atencion_sintetica.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nGuardado: {output_path}")


if __name__ == "__main__":
    main()