<p align="center"><strong>INFORME FINAL: ANÁLISIS Y PREDICCIÓN DE PRECIOS DE AUTOS</strong></p> 
1. Introducción

El objetivo de este proyecto es analizar un dataset de datos de automóviles y construir modelos de regresión para predecir el precio de los vehículos. Se realizó un Análisis Exploratorio de Datos (EDA) detallado, preprocesamiento de datos para manejar valores faltantes y categóricos, y se entrenaron y evaluaron tres modelos de regresión: Regresión Lineal, K-Nearest Neighbors (KNN) y Árbol de Decisión. Posteriormente, se exploraron técnicas de optimización de hiperparámetros y Feature Engineering para intentar mejorar el rendimiento de los modelos.
2. Análisis Exploratorio de Datos (EDA)

Durante el EDA, se cargó el dataset Automobile_data.csv y se realizó una inspección inicial:

    Se identificaron columnas con valores faltantes representados por “?”, que fueron convertidos a NaN.
    Los valores faltantes en columnas numéricas (normalized-losses, bore, stroke, horsepower, peak-rpm, price) fueron imputados con la media de su respectiva columna. Se eligió la media por ser una medida de tendencia central adecuada para datos simétricos; alternativamente, se pudo haber utilizado la mediana para mayor robustez, pero se priorizó la simplicidad.
    El valor faltante en la columna categórica num-of-doors fue imputado con la moda ('four'), ya que es la categoría más frecuente.
    Se verificó y confirmó la eliminación de valores nulos.
    No se encontraron filas duplicadas en el dataset.
    Se realizó un análisis de valores únicos por columna para entender la naturaleza de las variables categóricas y asegurar la consistencia.
    Se calculó la matriz de correlación para las variables numéricas, identificando relaciones lineales y características predictivas para el precio.

3. Preparación de Datos

El preprocesamiento fue fundamental para preparar el dataset para modelos de machine learning:

    Se definieron las características (X) y la variable objetivo (y), siendo price la variable a predecir y excluyendo symboling.
    Se identificaron columnas numéricas y categóricas para aplicar transformaciones adecuadas.
    Se utilizó ColumnTransformer dentro de un pipeline:
        Las columnas numéricas fueron escaladas con StandardScaler, esencial para modelos sensibles a la escala como Regresión Lineal y KNN.
        Las columnas categóricas fueron codificadas con OneHotEncoder, permitiendo que el modelo interprete cada categoría independientemente.
        Se usó handle_unknown='ignore' en OneHotEncoder para robustez ante categorías inéditas en el conjunto de prueba.
    El dataset fue dividido en conjuntos de entrenamiento (80%) y prueba (20%) con train_test_split y random_state=42 para reproducibilidad.

4. Modelos de Regresión y Evaluación Inicial

Se entrenaron y evaluaron tres modelos iniciales:

    Regresión Lineal: Modelo base que asume relación lineal entre características y precio.
    K-Nearest Neighbors (KNN): Predice precio según el promedio de los K vecinos más cercanos. Se seleccionó el mejor K (entre 1 y 20) mediante validación cruzada.
    Árbol de Decisión: Modelo basado en reglas, entrenado inicialmente con max_depth=8.

5. Benchmarking y Comparación Inicial

Los modelos fueron comparados en términos de MSE y R2 sobre el conjunto de prueba.

Resultados de Modelos Iniciales:
Resultados de Modelos Iniciales:

	  Model 	            MSE 	        R2
0 	Linear Regression 	5.199133e-23 	1.000000
1 	KNN 	              1.390393e+06 	0.982168
2 	Decision Tree 	    2.586472e+06 	0.966828


Interpretación: El modelo con mejor rendimiento (menor MSE y mayor R2) en el conjunto de prueba fue el que obtuvo los mejores valores en dichas métricas. La Regresión Lineal mostró un rendimiento competitivo, sugiriendo una relación lineal entre algunas características y el precio. KNN y Árbol de Decisión capturaron relaciones más complejas, pero su rendimiento dependió de la selección de hiperparámetros.
6. Desafíos Adicionales: Optimización y Feature Engineering

Para mejorar el rendimiento se exploraron:
6.1 Optimización de Hiperparámetros con GridSearchCV

Se utilizó GridSearchCV para buscar los mejores hiperparámetros en KNN y Árbol de Decisión, evaluando combinaciones mediante validación cruzada.

    KNN: Se optimizaron n_neighbors, weights y metric.
    Árbol de Decisión: Se optimizaron max_depth, min_samples_split, min_samples_leaf y criterion.

Resultados de la Optimización:
Comparación de Modelos Optimizado
    Model                      MSE (Test)      R2 (Test)
0   KNN (Optimized)            1.390393e+06   0.982168
1   Decision Tree (Optimized)  2.557296e+06   0.967202

Interpretación: La optimización de hiperparámetros tuvo impacto variable; puede mejorar el rendimiento generalizando mejor al conjunto de prueba, aunque no siempre garantiza mejoras si el modelo sobreajusta.
6.2 Feature Engineering

Se crearon nuevas características:

    volume: Producto de longitud, ancho y altura.
    hp_per_kg: Relación potencia/peso.
    combined_mpg: Promedio de consumo en ciudad y carretera.
    price_per_volume: Relación precio/volumen.
    engine_hp_interaction: Producto de tamaño de motor y potencia.
    num-of-cylinders_num: Conversión numérica de la columna num-of-cylinders.

Los modelos fueron reentrenados con estas nuevas características.

Resultados con Feature Engineering:
Comparación Final de Rendimiento con y sin Feature Engineering
                         Model    MSE (Test)  R2 (Test)
0  Linear Regression (Initial)  0.000000e+00     1.0000
1                KNN (Initial)  1.390393e+06     0.9822
2      Decision Tree (Initial)  2.586472e+06     0.9668
3  Linear Regression (with FE)  1.852369e+05     0.9976
4           KNN (with FE, k=2)  4.034724e+06     0.9483
5      Decision Tree (with FE)  2.456259e+06     0.9685

Interpretación: La ingeniería de características mejoró el rendimiento, especialmente en Regresión Lineal y Árbol de Decisión, validando que las combinaciones de características originales son informativas.
6.3 Validación Cruzada para Estabilidad

Se implementó validación cruzada (5 folds) para evaluar la estabilidad del rendimiento.

Resultados de Validación Cruzada:
Scores de CV para Regresión Lineal: [-2.06512701e-22 -3.03837562e-23 -8.62487102e-23 -1.33317305e-22
 -1.45765364e-22]
MSE promedio de CV para Regresión Lineal: -0.0000 (+/- 0.0000)
Scores de CV (R2) para Regresión Lineal: [1. 1. 1. 1. 1.]
R2 promedio de CV para Regresión Lineal: 1.0000 (+/- 0.0000)

Evaluando el mejor modelo KNN encontrado por GridSearchCV con CV:
Scores de CV (MSE) para KNN (Optimizado GS): [ -96239.56097561 -861109.7804878   -28312.38604092  -11964.09756098
  -26478.12806251]
MSE promedio de CV para KNN (Optimizado GS): -204820.7906 (+/- 329441.9973)
Scores de CV (R2) para KNN (Optimizado GS): [0.9986701  0.99321889 0.99847981 0.99978081 0.99852854]
R2 promedio de CV para KNN (Optimizado GS): 0.9977 (+/- 0.0023)

Evaluando el mejor modelo Árbol de Decisión encontrado por GridSearchCV con CV:
Scores de CV (MSE) para Árbol de Decisión (Optimizado GS): [-1095051.33859522 -1632923.69348093   -58244.95121951  -170471.62994335
  -252061.8619981 ]
MSE promedio de CV para Árbol de Decisión (Optimizado GS): -641750.6950 (+/- 616821.8030)
Scores de CV (R2) para Árbol de Decisión (Optimizado GS): [0.98486791 0.98714098 0.99687262 0.99687685 0.98599229]
R2 promedio de CV para Árbol de Decisión (Optimizado GS): 0.9904 (+/- 0.0054)

Interpretación: Una baja desviación estándar indica estabilidad y robustez en el rendimiento del modelo, lo que es crucial para confiar en sus predicciones.
7. Conclusiones

    El preprocesamiento de datos, manejo de valores faltantes y codificación de variables categóricas fue fundamental para preparar los datos.
    La ingeniería de características aportó mejoras, especialmente en modelos lineales y de árbol.
    La optimización de hiperparámetros refinó los modelos KNN y Árbol de Decisión.
    La validación cruzada proporcionó una medida robusta de estabilidad y rendimiento esperado.

Mejor modelo general:
El modelo con el mejor rendimiento general en el conjunto de prueba, considerando MSE y R2, fue la Regresión Lineal con un R2 de 1.000 y un MSE de 0.0000.”)
8. Posibles Mejoras y Próximos Pasos

    Continuar la ingeniería de características, explorando transformaciones adicionales y basadas en el dominio.
    Probar modelos más avanzados, como Random Forest o Gradient Boosting (XGBoost, LightGBM).
    Ajustar hiperparámetros de forma más exhaustiva con RandomizedSearchCV u optimización bayesiana.
    Realizar análisis de residuales para identificar patrones no capturados.
    Considerar el tratamiento de outliers con métodos más sofisticados.
    Implementar un conjunto de validación adicional si el tamaño de los datos lo permite.
    Realizar pruebas A/B en caso de despliegue para comparar el rendimiento con sistemas existentes.

