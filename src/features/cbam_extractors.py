"""
extraccion de features con atencion

El backbone congelado

CBAM se entrena una vez para cada semilla utilizando train de esa semilla

Tras entrenarse CBAM se congela y genera las features finales
"""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import math

from sklearn.model_selection import train_test_split
from src.data.preprocessing import (load_normalization_stats, preprocess_batch)
from src.features.extractors import (build_backbone, BACKBONE_NAME, FEATURES_DIR)

from src.models.classifiers import compute_class_weights

from src.models.cbam import CBAM





NUM_CHANNELS = 1280
INTERNAL_VAL_FRACTION = 0.1
MAX_EPOCHS = 15

PATIENCE = 3 #numero de epocas sin mejorar para ordenar el early stopping

LEARNING_RATE = 1e-3



def get_raw_feature_maps(backbone, batch_paths, mean, std, device):
    """ Devuelve el mapa de caracteristicas del backbone antes del pooling """

    batch = preprocess_batch(batch_paths, mean, std)

    tensor = torch.from_numpy(batch) #  convertimos el arrar de numpy en un tensor de pytorch
    tensor = tensor.to(device, non_blocking=True) # como siemrpe cpu o gpu

#congeladooo
    with torch.no_grad(): #backboen congelado -> no calcular gradientes -> toorch n ograd:

        feature_maps = backbone.forward_features(tensor)


    return feature_maps




def train_cbam(paths_train, y_train, resolution, device, seed, batch_size=64):
    """
    Entrena CBAM utilizando solamente el train de una semilla

    Cuenta con un conjunto de validacion -> para early stopping.
    """


  

    (paths_fit, paths_val_interno, y_fit, y_val_interno) = train_test_split(
                                                            paths_train,
                                                            y_train,
                                                            test_size = INTERNAL_VAL_FRACTION,
                                                            random_state = seed,
                                                            stratify = y_train
                                                        )



    mean, std = load_normalization_stats(resolution)


    backbone = build_backbone(device)

    cbam = CBAM(NUM_CHANNELS)


    cbam = cbam.to(device, non_blocking=True)



    head = nn.Linear(in_features = NUM_CHANNELS, out_features = 2) #1280 features de entrada y 2 posibles calses como respeusta abnormal y normal

    head = head.to(device, non_blocking=True)

    #Optimizacion
    trainable_parameters = (list(cbam.parameters()) + list(head.parameters()))



    #creamos el clasificador
    optimizer = torch.optim.Adam(trainable_parameters, lr= LEARNING_RATE)




    class_weights = compute_class_weights(y_fit)

    weights_list = [class_weights[0], class_weights[1]]
    weights_tensor = torch.tensor(weights_list, dtype = torch.float32, device = device)



    crit = nn.CrossEntropyLoss(weight=weights_tensor)


    

    # Early stopping - menor perdida -> mejor modelo.
    best_val_loss = math.inf

    epochs_without_improvement = 0 #cantidad de epocas que han pasado sin haber encontrado mejora

    best_cbam = [] #mejores pesos encontrados por cbam


    #TRAIN

    for epoch in range(MAX_EPOCHS):

        cbam.train()
        head.train()



        for start in range(0, len(paths_fit), batch_size):

            end = batch_size + start #cuando acaba cada lote


            batch_paths = paths_fit[start:end]

            batch_labels = y_fit[start:end]

            # etiquetas a tensor (crossentropy las necesita enteras)
            labels_tensor = torch.tensor(batch_labels,  dtype = torch.long,  device = device)



            feature_map = get_raw_feature_maps(backbone, batch_paths, mean, std, device)

            # Pasar el mapa por el modul ode atencion
            attended = cbam(feature_map)

            pooled = torch.nn.functional.adaptive_avg_pool2d(attended, 1) # (batch, 1280, H, W) -> (batch, 1280, 1, 1)

            pooled = pooled.flatten(1) # -> (batch, 1280)



            logits = head(pooled) # pasamso las features por head temporal


            # Calcula la perdida comparando los pesos antes de convertirlso en probabilidad con la salida original
            loss = crit(logits, labels_tensor)




            optimizer.zero_grad() #eliminamos lso gradientes de la iteracion anterior

            #calculamso los nuevos gradietnes
            loss.backward() 


            optimizer.step() #actualizamos los pesos



        #### Validacion

        cbam.eval()
        head.eval()

        # vamos a acumular la perdida total.
        val_loss_sum = 0.0

        val_samples = 0



        with torch.no_grad(): #val sin gradientes


            for start in range(0, len(paths_val_interno), batch_size):
                end = batch_size + start


                batch_paths = paths_val_interno[start:end]

                batch_labels = y_val_interno[start:end]

                # etiquetas a tensor (crossentropy las necesita enteras)
                labels_tensor = torch.tensor(batch_labels,  dtype = torch.long,  device = device)



                feature_map = get_raw_feature_maps(backbone, batch_paths, mean, std, device)

                # Pasar el mapa por el modul ode atencion
                attended = cbam(feature_map)

                pooled = torch.nn.functional.adaptive_avg_pool2d(attended, 1) # (batch, 1280, H, W) -> (batch, 1280, 1, 1)

                pooled = pooled.flatten(1) # -> (batch, 1280)


                logits = head(pooled) # pasamso las features por head temporal


                # Calcula la perdida comparando los pesos antes de convertirlso en probabilidad con la salida original
                loss = crit(logits, labels_tensor)


                #cantidad imagenes en e llote
                current_batch_size = len(batch_labels)


                # loss.item() = media
                val_loss_sum += loss.item() * current_batch_size

                val_samples += current_batch_size



        val_loss =  val_loss_sum / val_samples


        print(f"Epoch {epoch + 1}/{MAX_EPOCHS} - val_loss: {val_loss:.4f}")


        # Comprobacion ealy stopping
        if val_loss < best_val_loss: #si la nueva es mejor quedate con la nueva

            best_val_loss = val_loss

            best_cbam = copy.deepcopy(cbam.state_dict()) # hace falta deepcopy para copiar tmb los tensores


            epochs_without_improvement = 0 #reini9cio del contador


        else:  # si no ha mejorado +1 al contador
            epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE: #detenemos el entrenamiento

                print("Early stopping activado")

                break


    # recuperamos los pesos del mejor cbam
    cbam.load_state_dict(best_cbam)

    #ya no mas entrnear + congelamos sus parametros
    cbam.eval()

    for parameter in cbam.parameters():
        parameter.requires_grad = False


    return cbam



#############
# Extraeccion de features usando CBAM entrenado


@torch.inference_mode()
def extract_cbam_features(paths, resolution, device, backbone, cbam, batch_size=256):
    """ Extrae las features finales utilizando """

    mean, std = load_normalization_stats(resolution)

    feature_chunks = []


    for start in range(0, len(paths), batch_size):
        end = batch_size + start #cuando acaba cada lote


        batch_paths = paths[start:end]

        batch = preprocess_batch(batch_paths, mean, std)


        tensor = torch.from_numpy(batch)
        tensor = tensor.to(device)


        feature_map = backbone.forward_features(tensor)

        # Pasar el mapa por el modul ode atencion
        attended = cbam(feature_map)

        pooled = torch.nn.functional.adaptive_avg_pool2d(attended, 1) # (batch, 1280, H, W) -> (batch, 1280, 1, 1)

        pooled = pooled.flatten(1) # -> (batch, 1280)


        pooled = pooled.cpu()

        pooled_numpy = pooled.numpy()

        feature_chunks.append(pooled_numpy)



    # Unir lotes - buscamos (N, 1280) - tnemos varios 256, 1280
    features = np.concatenate(feature_chunks, axis=0) 






    del backbone
    del cbam

    if torch.cuda.is_available(): #si cuda esta disponible
        torch.cuda.empty_cache() #vacia memoria cache


    return features



def get_or_compute_cbam_features(paths, y, train_idx, resolution, device, seed, batch_size=256):
    cache_path = FEATURES_DIR / ( f"{BACKBONE_NAME}_{resolution}px_cbam_seed{seed}.npy")


    if cache_path.exists():

        cached_features = np.load(cache_path)

        expected_shape = (len(paths), NUM_CHANNELS)


        if cached_features.shape == expected_shape: #si en la cache existe ya guardaras una features con la forma correcta
            print(f"Cache CBAM encontrado: {cache_path}")

            return cached_features


        print("El cache CBAM existe pero no tiene al forma correcta")



    paths_train = []

    for i in train_idx:
        paths_train.append(paths[i]) 


    y_train_local = y[train_idx]


    backbone = build_backbone(device)


    cbam_entrenado = train_cbam(paths_train, y_train_local, resolution, device, seed)

    #generamos las features para todos lso conjuntos de datos (no solo train)
    features = extract_cbam_features(paths, resolution, device, backbone, cbam_entrenado, batch_size)


    #almacenamos
    np.save(cache_path, features)


    print(f"Features CBAM guardadas en: {cache_path}")


    return features