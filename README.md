# GastricCancer

Clasificación binaria (*Normal* / *Abnormal*) de parches histopatológicos de cáncer gástrico del conjunto [GasHisSDB](https://doi.org/10.1016/j.compbiomed.2021.105207), con un extractor convolucional preentrenado (**EfficientNetV2-B0**) y un clasificador de *gradient boosting* (**CatBoost**).

Código del Trabajo Fin de Máster *«Clasificación de cáncer gástrico en imágenes histopatológicas mediante aprendizaje profundo»*, Máster Universitario en Ciencia de Datos, Universidad de Alicante (2026).

<img width="1253" height="713" alt="image" src="https://github.com/user-attachments/assets/0d5d7733-40c0-4e58-9d49-3f5f6ede7801" />

<!-- <p align="center">
  <img src="img/pipeline.png" alt="Pipeline del sistema" width="750">
</p> -->

## Resultados principales

Exactitud media sobre el conjunto de prueba (5 semillas):

| Configuración | 80 × 80 | 120 × 120 | 160 × 160 |
|---|:---:|:---:|:---:|
| EfficientNetV2-B0 + CatBoost | 89,64 % | 92,24 % | 93,24 % |
| + aumento de datos (D4) | 90,76 % | 93,19 % | 94,38 % |
| + CBAM | **91,80 %** | **93,46 %** | **94,70 %** |

- El aumento de datos mejora en torno a un punto en todos los tamaños y semillas.
- CBAM da el mejor resultado y gana más cuanto más pequeño es el parche.
- La transferencia entre tamaños es asimétrica: entrenar en 80 px y evaluar en 160 px conserva un 85,28 %, pero en la dirección contraria baja al 67,38 %.

## Método

1. **Preprocesado.** Los parches se redimensionan a 224 × 224 y se normalizan.
2. **Extracción.** EfficientNetV2-B0 preentrenado en ImageNet, con los pesos congelados, produce un mapa de 7 × 7 × 1280 que se resume con *Global Average Pooling* en un vector de 1280 dimensiones.
3. **Clasificación.** CatBoost con pesos de clase. Los hiperparámetros se eligen con validación cruzada estratificada de 5 pliegues sobre el 80 % de entrenamiento.
4. **Variantes evaluadas por separado:**
   - **Aumento de datos:** rotaciones y reflexiones (simetrías D4), solo en entrenamiento.
   - **CBAM:** módulo de atención de canal y espacial entre el extractor y el GAP.
5. **Análisis:** transferencia entre tamaños de parche e interpretabilidad (mapas de atención de CBAM y proyecciones SHAP).

Todas las configuraciones comparten particiones y semillas, así que las comparaciones son pareadas.

## Estructura del repositorio

```
GastricCancer/
├── src/            # Extracción de características, entrenamiento y evaluación
├── results/        # Métricas de cada experimento (JSON)
├── img/            # Figuras del README
├── requirements.txt
└── README.md
```

## Instalación

```bash
git clone https://github.com/lvs42/GastricCancer.git
cd GastricCancer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datos

El conjunto GasHisSDB no se incluye en el repositorio. Descárgalo desde su fuente original y colócalo así:

```
data/GasHisSDB/
├── 80/   ├── Normal/   └── Abnormal/
├── 120/  ├── Normal/   └── Abnormal/
└── 160/  ├── Normal/   └── Abnormal/
```

## Uso

```bash
# 1. Extraer características
python src/<script_extraccion>.py --size 160

# 2. Entrenar y evaluar CatBoost
python src/<script_entrenamiento>.py --size 160 --seed 0
```

Los resultados se guardan en `results/`.

## Limitaciones

- GasHisSDB no indica de qué imagen o paciente procede cada parche, así que las particiones son a nivel de parche. Por eso las métricas podrían ser optimistas para pacientes no vistos.
- No hay validación sobre un conjunto externo.
- La interpretabilidad es exploratoria: la atención espacial de CBAM trabaja cerca de la saturación y SHAP no vincula las decisiones a estructuras histológicas concretas.

## Cita

```bibtex
@mastersthesis{vazquez2026gastric,
  author = {Vázquez Soriano, Lucila},
  title  = {Clasificación de cáncer gástrico en imágenes histopatológicas mediante aprendizaje profundo},
  school = {Universidad de Alicante},
  year   = {2026},
  type   = {Trabajo Fin de Máster}
}
```

Dataset: Hu, W. *et al.* GasHisSDB: A new gastric histopathology image dataset for computer aided diagnosis of gastric cancer. *Computers in Biology and Medicine*, 2022.


## Licencia

Distribuido bajo licencia MIT. Ver [`LICENSE`](LICENSE).
