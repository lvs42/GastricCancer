"""
Transformaciones de data augmentation sobre las imágenes de GasHisSDB.
flips + rotaciones de 90/180/270
rotar o reflejar la imagen no cambia su interpretacion clinica
"""


from pathlib import Path

import numpy as np

from src.data.preprocessing import load_image_rgb



def flip_h(image):
    flipped = np.fliplr(image)
    return flipped

def flip_v(image):
    flipped = np.flipud(image)
    return flipped

def rotate(image, degree):
    k=0
    match degree:
        case 90:
            k = 1
        case 180:
            k = 2
        case 270:
            k = 3

    rotated = np.rot90(image, k = k)

    return rotated

def flip_and_rotate(image, dir, degree):
    k=0
    match degree:
        case 90:
            k = 1
        case 180:
            k = 2
        case 270:
            k = 3

    if dir == "v":
        image = flip_v(image) 
    elif dir == "h":
        image = flip_h(image)

    rotated = np.rot90(image, k = k)

    return rotated

def identity(image):
    return image

D4_TRANSFORMS = [
    identity,
    flip_v,
    flip_h,
    #rotate,
    #flip_and_rotate,
    lambda image: rotate(image, 90),
    lambda image: rotate(image, 180),
    lambda image: rotate(image, 270),
    lambda image: flip_and_rotate(image, "h", 90),
    lambda image: flip_and_rotate(image, "v", 90),
]



def sample_augmentations(rng: np.random.Generator, k: int) -> list:
    """Elige k transformaciones D4 distintas (excluye la identidad, indice 0)."""

    n = len(D4_TRANSFORMS ) - 1

    indices = rng.choice(n, size = k, replace = False)
    
    selected_transforms = []

    for index in indices:
        real_index = index + 1

        selected_transform = D4_TRANSFORMS[real_index]

        selected_transforms.append(selected_transform)

    return selected_transforms



def augment_image(image: np.ndarray, transform) -> np.ndarray:
    """Aplica una transformacion D4 a una imagen RGB ya cargada."""
    aumentada = transform(image) #augmented_image = flip_h/rotate/...(image_rgb)

    return aumentada



def generate_augmented_views(path: Path, rng: np.random.Generator, k: int) -> list[np.ndarray]:
    """Carga una imagen y devuelve k versiones aumentadas en RGB, sin resize
    ni normalizar todavia (eso lo hace preprocess_batch despues, igual que
    con las imagenes originales)."""
    image = load_image_rgb(path)

    transformations = sample_augmentations(rng, k)

    list_aumentadas = []

    for t in transformations:
        transformed = augment_image(image, t)
        list_aumentadas.append(transformed)
    
    return list_aumentadas



def generate_augmented_batch(paths: list[Path], rng: np.random.Generator, k: int) -> np.ndarray:
    """Devuelve un array con las vistas aumentadas de cada imagen del lote,
    listo para pasar por preprocess_batch."""
    batch = []
    for path in paths:
        #image = load_image_rgb(path)

        versions = generate_augmented_views(path, rng, k)
        batch.append(versions)

    batch_array = np.stack(batch, axis = 0)

    return batch_array