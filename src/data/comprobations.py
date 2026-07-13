import csv
import random
from pathlib import Path

DATASET_ROOT = Path("dataset/GasHisSDB/GasHisSDB")
RESOLUTIONS = ["80", "120", "160"]
CLASSES = ["Abnormal", "Normal"]


def main():
    random.seed(42)

    if not DATASET_ROOT.exists():
        print(f"ERROR: no existe la ruta {DATASET_ROOT}. Revisa el bind mount de Docker.")
        return [], 0

    total_images = 0
    list_images = []
    
    for res in RESOLUTIONS:
        res_folder = DATASET_ROOT / res
        for classs in CLASSES:
            class_folder = res_folder/ classs
            images = sorted(class_folder.glob("*.png"))
            list_images.extend(images)
            total_images += len(images)
    print(f"Total de imágenes encontradas: {total_images}")
    
    if total_images != 245196:
        print("El total no coincide exactamente con el README. ")
    return list_images, total_images



if __name__ == "__main__":
    list_images, total_images = main()

    print("Ejemplo primera imagen:", list_images[0] if list_images else "No hay imágenes")
    print("Total:", total_images)
