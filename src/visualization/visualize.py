"""
Genera imagenes con el mapa de atencion espacial de CBAM superpuesto sobre parches de ejemplo

Solo visualizamos Spatial Attention
"""

from pathlib import Path

import cv2

import matplotlib.pyplot as plt

import numpy as np
import torch

from src.data.loaders import load_inventory
from src.data.preprocessing import (load_image_rgb, load_normalization_stats, preprocess_batch)
from src.features.extractors import build_backbone

from src.models.cbam import CBAM


current_file = Path(__file__)

project_root = current_file.resolve().parents[2]

CHECKPOINTS_DIR = (project_root / "results" / "cbam_checkpoints")
OUTPUT_DIR = (project_root / "results" / "interpretability")


NUM_CHANNELS = 1280
N_EXAMPLES_PER_CLASS = 3



def load_trained_cbam(resolution, seed, device):
    """ Carag un cbam ya entrenado (carga los pesos) listo para eval"""


    cbam = CBAM(NUM_CHANNELS)

    checkpoint_name = (f"cbam_{resolution}px_seed{seed}.pt")

    checkpoint_path = CHECKPOINTS_DIR / checkpoint_name


    #cargamos los pesso guardados
    saved_weights = torch.load(checkpoint_path, map_location = device)

    
    cbam.load_state_dict(saved_weights) #le introducimos lso pesos cargados al cbam creado

    cbam = cbam.to(device)

    cbam.eval()




    for parameter in cbam.parameters():
        parameter.requires_grad = False #congelamos los parametros


    return cbam






def compute_spatial_attention_map(image_path, backbone, cbam, mean, std, device):

    image_rgb = load_image_rgb(image_path) #imagen directa del dataset, sin preprocesamiento



    batch = preprocess_batch([image_path],  mean, std) #la preprocesamos


    tensor = torch.from_numpy(batch)
    tensor = tensor.to(device)



    with torch.no_grad():


        feature_map = backbone.forward_features(tensor)

        channel_weights = cbam.channel_attention(feature_map)
        x_channel = feature_map * channel_weights #aplicamos pesso al mapa


        spatial_weights = cbam.spatial_attention(x_channel)




    attention_map = spatial_weights[0] #(1, 1, H, W) -> (1, H, W)

    attention_map = attention_map[0] # + eliminar dimension de canal




    attention_map = attention_map.cpu()

    attention_map = attention_map.numpy()


    #  devolvemos la imagen original + mapa de atencion
    return (image_rgb, attention_map)







def overlay_attention_on_image(image_rgb, attention_map, vmin, vmax, alpha=0.5):
    """Convierte el mapa de atencion en un mapa de calor del tamaño de la imagen original y mezcla ambos"""



    image_height = image_rgb.shape[0]
    image_width = image_rgb.shape[1]



    #redimensioanmos el mapa a ltamaño de la imagen original
    resized_attention = cv2.resize(attention_map, (image_width, image_height), interpolation = cv2.INTER_CUBIC) #recomiendan INTER_CUBIC or INTER_LINEAR para agrandar la iamgen, y INTER_AREA si se hace más pequeña - proabr





    
    #min_value = resized_attention.min()
    #max_value = resized_attention.max()

    #rango total
    #value_range = max_value - min_value
    value_range = vmax - vmin #escala global

   
    if value_range:
        normalized_attention = (resized_attention - vmin) / value_range

        normalized_attention = np.clip(normalized_attention, 0.0, 1.0)

    else: #caso no poder dividir entre cero (si mimso tamaño)
        normalized_attention = np.zeros_like(resized_attention)




    # Creamos mapa de color
    heatmap = plt.cm.jet(normalized_attention) #nso da el alto, ancho y lso 4 canales de la imagen (rgb y alpha)

    heatmap_rgb = heatmap[:, :, :3] #asi nso qedamos solo con lso canales rgb (con valroes entre 0 y 1)
    heatmap_rgb = heatmap_rgb * 255



    heatmap_rgb = heatmap_rgb.astype(np.uint8) #lo cambiamos para poder hacer operacioness


    # Fusionamos con la imagen

    image_float = image_rgb.astype(np.float32)

    heatmap_float = heatmap_rgb.astype(np.float32)



    overlay = (((1 - alpha) * image_float) + (alpha * heatmap_float))

    overlay = overlay.astype(np.uint8)


    return overlay




def visualize_examples(resolution, seed, device):
    """ Genera varias visualizaciones para las dos clases """

    backbone = build_backbone(device)

    cbam = load_trained_cbam(resolution, seed, device)




    mean, std = load_normalization_stats(resolution)
    paths, y = load_inventory(resolution)


    #priemros ejemplos de cada clase
    normal_paths = []

    abnormal_paths = []


    for i in range(len(paths)):

        path = paths[i]
        label = y[i]


        if (label == 0 and len(normal_paths) < N_EXAMPLES_PER_CLASS): # y todavía no tenemos n ejemplos por clase normales

            normal_paths.append(path)


        if (label == 1 and len(abnormal_paths) < N_EXAMPLES_PER_CLASS): #por clase abnormal
            abnormal_paths.append(path)

        #si ya tenemos duifucientes
        if (len(normal_paths) >= N_EXAMPLES_PER_CLASS and len(abnormal_paths) >= N_EXAMPLES_PER_CLASS): #suficientes de ambso tipos

            break






    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)




    ejemplos = []

    for index, image_path in enumerate(normal_paths):

        image_rgb, attention_map = compute_spatial_attention_map(image_path,  backbone, cbam, mean, std, device)

        ejemplos.append((image_rgb, attention_map, "Normal", index))
        
        """
        overlay = overlay_attention_on_image(image_rgb, attention_map) # genera el overlay



        fig, axes = plt.subplots(1,  2, figsize=(8, 4))

        axes[0].imshow(image_rgb)

        axes[0].set_title("Original - Normal")
        axes[0].axis("off")


        axes[1].imshow(overlay)
        axes[1].set_title("Atencion espacial")

        axes[1].axis( "off")



        filename = (f"atencion_{resolution}px_Normal_{index}.png")
        output_path = OUTPUT_DIR / filename



        plt.tight_layout()


        plt.savefig(output_path, dpi=150)





        plt.close(fig) # "libera" la memoria cerrrarla
"""

   

    for index, image_path in enumerate(abnormal_paths):

        image_rgb, attention_map = compute_spatial_attention_map(image_path,  backbone, cbam, mean, std, device)

        ejemplos.append((image_rgb, attention_map, "Abnormal", index))
        
        """
        overlay = overlay_attention_on_image(image_rgb, attention_map) # genera el overlay




        fig, axes = plt.subplots(1,  2, figsize=(8, 4))

        axes[0].imshow(image_rgb)

        axes[0].set_title("Original - Abnormal")
        axes[0].axis("off")


        axes[1].imshow(overlay)
        axes[1].set_title("Atencion espacial")

        axes[1].axis( "off")



        filename = (f"atencion_{resolution}px_Abnormal_{index}.png")
        output_path = OUTPUT_DIR / filename



        plt.tight_layout()


        plt.savefig(output_path, dpi=150)





        plt.close(fig) # "libera" la memoria cerrrarla
"""
    for image_rgb, attention_map, clase, index in ejemplos:
        print(f"{clase}_{index}: min={attention_map.min():.4f} max={attention_map.max():.4f} rango={attention_map.max()-attention_map.min():.4f}")

    # Escala global: minimo y maximo entre TODOS los mapas de este lote.
    global_min = min(attention_map.min() for _, attention_map, _, _ in ejemplos)
    global_max = max(attention_map.max() for _, attention_map, _, _ in ejemplos)
    print(f"Escala de color compartida: [{global_min:.4f}, {global_max:.4f}]")

    for image_rgb, attention_map, clase, index in ejemplos:
        overlay = overlay_attention_on_image(image_rgb, attention_map, vmin=global_min, vmax=global_max)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(image_rgb)
        axes[0].set_title(f"Original - {clase}")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Atencion espacial")
        axes[1].axis("off")

        filename = f"atencion_{resolution}px_{clase}_{index}.png"
        output_path = OUTPUT_DIR / filename

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    total_images = len(normal_paths) + len(abnormal_paths)


    print()
    print(f"Se han guardado {total_images} visualizaciones.")

    print(f"Directorio: {OUTPUT_DIR}")




if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser(
        description="Visualiza la atencion espacial de CBAM."
    )
    parser.add_argument(
        "--resolution",
        type = int,
        required = True,
        choices = [80,120,160]
    )


    parser.add_argument("--seed", type = int, default = 42)


    args = parser.parse_args()




    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    visualize_examples(args.resolution, args.seed, device)