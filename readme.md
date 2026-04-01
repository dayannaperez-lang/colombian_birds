# 🐦 Identificación de Cantos de Aves Colombianas con RNN

## Descripción y Contextualización del Problema

Colombia es el país con mayor diversidad de aves del mundo, con más de 1.900 especies registradas que representan cerca del 20% del total mundial. Esta riqueza biológica convierte al territorio colombiano en un escenario de alto valor para la investigación ornitológica y la conservación de ecosistemas. Sin embargo, el monitoreo tradicional de especies depende de la presencia de expertos en campo, lo que implica altos costos logísticos, limitaciones geográficas y una capacidad de cobertura reducida frente a la extensión del territorio nacional.

El canto de las aves es una de las señales biológicas más características y constantes de cada especie. A diferencia de la identificación visual, la identificación acústica permite el monitoreo remoto, continuo y no invasivo de poblaciones aviares, convirtiéndose en una herramienta especialmente útil para estudios de biodiversidad y detección de especies en zonas de difícil acceso.

### Formulación del Problema de Machine Learning

El problema se formula como una tarea de **clasificación multiclase supervisada**: dado un fragmento de audio que contiene el canto de un ave, el modelo debe predecir a cuál de las 29 especies registradas en el dataset pertenece dicha grabación.

Se busca aprender una función `f: X → y` donde:

- `X ∈ R^(224×224×3)` representa la entrada: un espectrograma de Mel procesado como imagen de tres canales.
- `y ∈ {1, 2, ..., 29}` es la especie predicha.

---

## Descripción de la Base de Datos

Los datos fueron obtenidos de **[Xeno-canto](https://xeno-canto.org/)**, una plataforma colaborativa de acceso abierto que reúne grabaciones de cantos de aves de todo el mundo realizadas por ornitólogos y aficionados. Para este proyecto se filtraron grabaciones correspondientes a **29 especies de aves presentes en Colombia**.

| Característica | Valor |
|---|---|
| Fuente | Xeno-canto (xeno-canto.org) |
| Número de muestras | 1.101 |
| Número de clases | 29 especies |
| Formato original | MP3 |
| Duración de segmentos | 5 segundos |
| Representación | Espectrograma de Mel (224×224×3) |

### Preprocesamiento

Cada archivo de audio fue procesado mediante el siguiente pipeline:

1. **Segmentación**: cada grabación se divide en fragmentos de 5 segundos. Los segmentos más cortos se rellenan con ceros (padding).
2. **Espectrograma de Mel**: cada segmento se transforma en un espectrograma de Mel con 128 bandas de frecuencia usando `librosa`.
3. **Redimensionamiento**: el espectrograma se escala a 224×224 píxeles usando `cv2.resize`.
4. **Normalización**: los valores se normalizan al rango [0, 1].
5. **Canales**: el espectrograma se replica en 3 canales para ser compatible con la CNN.
6. **Selección de chunk**: por cada audio se selecciona el segmento con mayor energía (mayor desviación estándar).

---

## Diseño Experimental

### Etapas del Proyecto

1. **Recolección y preparación de datos**: descarga de audios desde Xeno-canto, segmentación y transformación a espectrogramas de Mel.
2. **Construcción del modelo**: diseño de la arquitectura CNN + LSTM.
3. **Entrenamiento**: ajuste de pesos con el conjunto de entrenamiento.
4. **Evaluación**: medición del desempeño sobre el conjunto de validación.

### División de Datos

| Conjunto | Proporción | Muestras |
|---|---|---|
| Entrenamiento | 80% | ~880 |
| Validación | 20% | ~221 |

La división se realizó de forma estratificada para garantizar representación proporcional de cada especie en ambos conjuntos.

### Métricas de Evaluación

- **Accuracy**: proporción de predicciones correctas sobre el total de muestras.
- **Precision**: capacidad del modelo de no clasificar como positiva una muestra negativa.
- **Recall**: capacidad del modelo de encontrar todas las muestras positivas de cada clase.
- **F1-score**: media armónica entre precision y recall, especialmente útil ante posible desbalance de clases.
- **Matriz de confusión**: visualización de los errores de clasificación entre especies.

---

## Experimentos

### Arquitectura: CNN + LSTM

Se adoptó una arquitectura híbrida que combina redes convolucionales y recurrentes. La CNN extrae patrones espaciales y frecuenciales del espectrograma, mientras que la LSTM modela la evolución temporal del canto.

```
Input (224×224×3)
      ↓
Conv2D(32) → ReLU → MaxPool    # → (32, 112, 112)
Conv2D(64) → ReLU → MaxPool    # → (64, 56, 56)
Conv2D(128) → ReLU → MaxPool   # → (128, 28, 28)
      ↓
Reshape → 28 timesteps × (128×28) features
      ↓
LSTM(hidden=128, layers=2, dropout=0.3)
      ↓
Linear(128→64) → ReLU → Dropout(0.4)
      ↓
Linear(64→29)  # clasificación final
```

### Código de Entrenamiento

```python
# Preprocesamiento de audio
def audio_to_mel_chunks(path, segment_duration=5, n_mels=128):
    y, sr = librosa.load(path, sr=None)
    segment_length = int(sr * segment_duration)
    mels = []
    for start in range(0, len(y), segment_length):
        segment = y[start:start+segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
        mels.append(librosa.power_to_db(mel, ref=np.max))
    return mels, sr

# Modelo
class BirdCNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(128*28, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        f = self.cnn(x)
        B, C, H, W = f.shape
        seq = f.permute(0,3,1,2).reshape(B, W, C*H)
        _, (h, _) = self.lstm(seq)
        return self.classifier(h[-1])

# Hiperparámetros
EPOCHS     = 40
LR         = 1e-3
BATCH_SIZE = 16
OPTIMIZER  = Adam (weight_decay=1e-4)
SCHEDULER  = StepLR (step_size=10, gamma=0.5)
```

### Configuración del Experimento

| Hiperparámetro | Valor |
|---|---|
| Épocas | 40 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Optimizador | Adam |
| Dropout | 0.3 / 0.4 |
| Hidden size LSTM | 128 |
| Capas LSTM | 2 |

---

## Conclusiones

| Métrica | Valor |
|---|---|
| Accuracy validación | 0.93 |
| F1-score macro | 0.84|


### Discusión

La arquitectura CNN + LSTM demostró ser una elección adecuada para este problema por dos razones principales. Primero, la representación del audio como espectrograma de Mel permite aprovechar las capacidades de las redes convolucionales para extraer patrones frecuenciales característicos de cada especie. Segundo, la LSTM captura la dinámica temporal del canto, que es el rasgo más discriminativo entre especies con perfiles frecuenciales similares.

Con un dataset de 1.101 muestras y 29 clases, el modelo enfrenta el reto del **desbalance de datos** y la **escasez de muestras por clase**.
El modelo alcanzó un accuracy del 93%, con F1-score perfecto en la mayoría de las 29 especies. Las clases 6, 17 y 25 obtuvieron F1-score de 0.00, pero las tres cuentan con solo 2 muestras en validación, lo que hace que cualquier error colapse las métricas. Las clases 5, 8 y 16 presentaron métricas más bajas con mayor soporte, lo que indica dificultad real del modelo, posiblemente por similitud acústica entre especies. El principal factor limitante es el desbalance de clases.

---

## Requisitos

```bash
pip install librosa opencv-python torch scikit-learn matplotlib seaborn numpy
```

---

*Datos obtenidos de [Xeno-canto](https://xeno-canto.org/)*