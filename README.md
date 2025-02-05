# Clasificador KNN en Python

Este proyecto implementa un clasificador K-Nearest Neighbors (KNN) en Python, basado en el video "Clasificador KNN | Machine Learning | Aprendizaje Automático | Python" de Victor Romero.

## Descripción

El algoritmo KNN es un método de clasificación supervisada que se utiliza para clasificar un punto de datos basado en la cercanía de sus vecinos más cercanos. Este proyecto incluye una implementación básica del algoritmo KNN utilizando Python y bibliotecas como `pandas`, `numpy` y `collections`.

## Librerías Utilizadas

- `pandas`: Manejo, análisis y procesamiento de datos.
- `collections`: Cálculo de frecuencias.
- `numpy`: Manejo de matrices y arreglos multidimensionales.

## Funciones Principales

### AlgKNN
Clase que implementa el algoritmo KNN.

- `__init__(self, k)`: Inicializa el clasificador con el número de vecinos cercanos (k).
- `learn(self, Q, C)`: Método de aprendizaje que recibe los datos de entrenamiento y sus respectivas clases.
- `clasf(self, P)`: Método de clasificación que recibe los datos de prueba y devuelve las clases predichas.

### euclidiana(x, y)
Función que calcula la distancia euclidiana entre dos puntos.

## Uso

1. Asegúrate de tener las bibliotecas necesarias instaladas:
    ```bash
    pip install pandas numpy
    ```

2. Coloca tus datos de entrenamiento y prueba en las variables correspondientes dentro del script.

3. Ejecuta el script para entrenar el clasificador y clasificar nuevos puntos de datos.

## Ejemplos

El proyecto incluye ejemplos de cómo inicializar el clasificador KNN, entrenarlo con datos de ejemplo y clasificar nuevos puntos de datos. Consulta el código fuente para más detalles.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request para discutir cualquier cambio que te gustaría hacer.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

## Referencias

> Basado a la codificacion del video Clasificador KNN | Machine Learning | Aprendizaje Automático | Python
 Autor: Victor Romero
 Link : https://www.youtube.com/watch?v=qv0rb_A0f3M&t=1098s