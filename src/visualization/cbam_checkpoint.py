"""
Entrena CBAM UNA vez para una resolucion y una semilla concretas

Despues guarda sus pesos para reutilziarlos más adelante
"""

import argparse

from pathlib import Path
import torch

from src.data.loaders import (load_inventory, split_train_test)
from src.features.cbam_extractors import train_cbam



TEST_SIZE = 0.2


current_file = Path(__file__)
project_root = current_file.resolve().parents[2]

CHECKPOINTS_DIR = (project_root / "results" / "cbam_checkpoints")




def main():

    parser = argparse.ArgumentParser(description="Entrena y guarda un CBAM para visualizacion")


    parser.add_argument(
        "--resolution",
        type = int,
        required = True, #
        choices = [80,120,160]
    )


    parser.add_argument("--seed", type = int, default = 42)


    args = parser.parse_args()

    #si cuda esta diponible la usamos
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    print(f"Dispositivo: {device}")


    paths, y = load_inventory(args.resolution)



    # repliocamos el mismo split que en el training
    train_idx, test_idx = split_train_test(y, seed = args.seed, test_size = TEST_SIZE)



    paths_train = []

    for i in train_idx:

        path = paths[i]
        paths_train.append(path)


    #labels
    y_train = y[train_idx]



    # entrenar cbam
    cbam = train_cbam(paths_train, y_train,  args.resolution, device, seed = args.seed)


    # creamos la carpeta dodnde guardaremso los pesos si todavia no exiwste
    CHECKPOINTS_DIR.mkdir(parents = True, exist_ok = True)

    checkpoint_name = (f"cbam_{args.resolution}px_seed{args.seed}.pt") #por ej. cbam_80px_seed42.pt
    checkpoint_path = CHECKPOINTS_DIR / checkpoint_name #la ruta en sí



    #recuepramso lso pesos
    cbam_weights = cbam.state_dict()

    torch.save(cbam_weights, checkpoint_path)


    print()

    print("Checkpoint CBAM guardado en:", checkpoint_path)







if __name__ == "__main__":

    main()