# Time Series Auto

## Objetivo

Este proyecto construye un pipeline de forecasting semanal para series temporales agregadas por `type`, generando variables explicativas a partir del histórico de demanda y entrenando un modelo global de `CatBoostRegressor`.

El flujo principal está dividido en dos scripts:

- **`src/create_processed_for_training.py`**
  - lee el dataset procesado
  - construye la tabla de modelado con features
  - guarda el dataset final para entrenamiento

- **`src/train_and_evaluate.py`**
  - carga la tabla de modelado
  - separa train, validation y test en el tiempo
  - entrena el modelo
  - evalúa baselines y CatBoost
  - guarda métricas, predicciones, modelo y feature importance

---

## Estructura de features

La ingeniería de variables está implementada en:

- **`src/utils/utils_feature_engineering.py`**

Estas features se generan sobre datos semanales usando la columna temporal `week` y la serie identificada por `type`.

### 1. Features de rezago

Capturan el comportamiento histórico reciente y estacional de cada serie.

Se generan lags en semanas como:

- `lag_0`
- `lag_1`
- `lag_2`
- `lag_3`
- `lag_4`
- `lag_8`
- `lag_13`
- `lag_26`
- `lag_51`
- `lag_52`
- `lag_53`

#### Interpretación

- **`lag_0`**: valor actual de la semana origen
- **`lag_1`**: demanda de la semana anterior
- **`lag_4` / `lag_8` / `lag_13`**: memoria de corto y medio plazo
- **`lag_52`**: referencia anual aproximada

---

### 2. Features rolling

Resumen estadístico sobre ventanas móviles históricas.

Para varias ventanas (`3`, `4`, `5`, `8`, `13`, `26`, `52`) se calculan:

- `rolling_mean_*`
- `rolling_median_*`
- `rolling_std_*`
- `rolling_min_*`
- `rolling_max_*`
- `rolling_iqr_*`

#### Interpretación

Estas variables ayudan a modelar:

- **nivel medio de demanda**
- **volatilidad**
- **rango típico de variación**
- **comportamiento estable o errático**

---

### 3. Features EWM

Se generan medias móviles exponenciales:

- `ewm_mean_4`
- `ewm_mean_8`
- `ewm_mean_13`
- `ewm_mean_26`

#### Interpretación

Dan más peso a las observaciones recientes que una media rolling clásica.

Son útiles cuando la serie cambia de nivel y conviene reaccionar más rápido al último comportamiento observado.

---

### 4. Features de tendencia y momentum

Se construyen variables que comparan semanas o promedios entre sí:

- `momentum_diff_1`
- `momentum_ratio_1`
- `momentum_diff_4`
- `momentum_ratio_4`
- `momentum_mean_4_vs_13`
- `momentum_ratio_4_vs_13`
- `momentum_mean_8_vs_26`
- `momentum_ratio_8_vs_26`
- `rolling_slope_4`
- `rolling_slope_8`
- `rolling_slope_13`

#### Interpretación

Estas features intentan capturar:

- **aceleración o desaceleración de la demanda**
- **cambios de tendencia**
- **dirección reciente de la serie**

---

### 5. Features de estacionalidad y calendario

Se generan componentes temporales explícitos:

- `year`
- `week_of_year`
- `month`
- `quarter`
- `time_index`

Y componentes de Fourier:

- `fourier_sin_1`, `fourier_cos_1`
- `fourier_sin_2`, `fourier_cos_2`
- `fourier_sin_3`, `fourier_cos_3`

Además:

- `last_year_mean_3`
- `last_year_median_3`
- `last_year_ratio_3`

#### Interpretación

Estas variables permiten representar:

- **patrones anuales**
- **efectos de semana del año**
- **estacionalidad suave**
- **comparación con el mismo período del año anterior**

---

### 6. Features de intermitencia

Pensadas para series con muchos ceros o demanda irregular.

Se generan:

- `is_nonzero_lag_0`
- `nonzero_rate_4`
- `nonzero_rate_8`
- `nonzero_rate_13`
- `nonzero_rate_26`
- `nonzero_rate_52`

#### Interpretación

Estas variables ayudan a distinguir entre:

- series de demanda frecuente
- series intermitentes
- series con alta probabilidad de cero

---

### 7. Features de spikes y anomalías

El pipeline crea variables para detectar picos de demanda:

- `spike_prior_median_13`
- `spike_prior_mad_13`
- `robust_z_spike_13`
- `is_spike`
- `time_since_last_spike`
- `spike_rate_8`
- `spike_rate_13`
- `spike_rate_26`

#### Interpretación

Estas features permiten modelar:

- eventos anómalos
- distancia al último pico relevante
- frecuencia reciente de picos

En el resultado actual, `time_since_last_spike` aparece como la feature más importante, lo que indica que la recurrencia de picos tiene mucho peso predictivo.

---

### 8. Variable objetivo y escalado

Se construyen también variables relacionadas con el target:

- `target`
- `scale_factor`
- `log_scale_factor`
- `target_scaled`

#### Interpretación

- **`target`**: demanda futura a predecir
- **`scale_factor`**: escala dinámica de la serie basada en historial reciente
- **`target_scaled`**: target normalizado para estabilizar el entrenamiento

El modelo se entrena sobre `target_scaled`, y luego las predicciones se reescalan al nivel original.

---

### 9. Features escaladas

Muchas features de demanda se transforman también a una versión escalada mediante `scale_factor`.

Ejemplos:

- `rolling_std_52_scaled`
- `ewm_mean_13_scaled`
- `lag_8_scaled`
- `rolling_mean_26_scaled`
- `rolling_slope_13_scaled`

#### Objetivo

Esto ayuda a que series con niveles muy distintos puedan compartir un mismo modelo global sin que las series grandes dominen numéricamente a las pequeñas.

---

## Selección de features para entrenamiento

La selección de columnas finales se hace con la función `build_feature_columns(...)`.

### Se excluyen del entrenamiento

No se usan como input directo:

- `value`
- `target`
- `target_scaled`
- `week`
- `sample_weight`

### Variables categóricas

Las variables tratadas como categóricas son:

- `type`
- `week_of_year`
- `month`
- `quarter`

Estas columnas se convierten a string para compatibilidad con CatBoost.

### Uso de features escaladas

Cuando `use_scaled_demand_features=True`, el pipeline prioriza las versiones escaladas de las variables de demanda y excluye las versiones originales no escaladas equivalentes.

Esto hace más robusto el modelo global cuando conviven series de distinta magnitud.

---

## Cómo se usan en el entrenamiento

La lógica de entrenamiento está en:

- **`src/utils/utils_model_training.py`**

### 1. Split temporal

La tabla de modelado se divide en:

- **train**
- **validation**
- **test**

usando ventanas temporales semanales, sin mezclar futuro con pasado.

Por defecto:

- `test_size_weeks = 8`
- `valid_size_weeks = 4`

---

### 2. Pesos temporales

Se genera `sample_weight` con decaimiento temporal.

#### Objetivo

Dar más peso a las observaciones más recientes, ya que suelen ser más representativas para forecasting operativo.

---

### 3. Entrenamiento con CatBoost

Se usa un modelo global:

- **`CatBoostRegressor`**

Configuración base:

- loss: `RMSE`
- eval metric: `RMSE`
- early stopping activado
- uso de features categóricas
- entrenamiento sobre `target_scaled`

Luego las predicciones se reescalan multiplicando por `scale_factor`.

---

### 4. Evaluación

Se comparan tres enfoques:

- `baseline_previous_week`
- `baseline_mean_last_4_weeks`
- `catboost_global`

Métricas reportadas:

- `mae`
- `rmse`
- `wape`
- `bias`

En la ejecución actual, `catboost_global` supera a ambos baselines.

---

## Feature importance

El script `src/train_and_evaluate.py` también guarda la importancia de variables en:

- **`results/lecta_feature_importance.csv`**

Según el resultado actual, las variables con mayor peso incluyen:

- `time_since_last_spike`
- `rolling_std_52_scaled`
- `ewm_mean_13_scaled`
- `fourier_sin_2`
- `week_of_year`
- `ewm_mean_26_scaled`

### Lectura rápida del resultado

Esto sugiere que el modelo está apoyándose sobre todo en:

- **señales de anomalías o recurrencia de picos**
- **volatilidad histórica**
- **nivel reciente suavizado**
- **estacionalidad anual**

---

## Archivos de salida del pipeline

### Generación de dataset de entrenamiento

- `data/Lecta/processed_for_training/lecta_processed_for_training.csv`

### Entrenamiento y evaluación

- `data/Lecta/interim/lecta_train.csv`
- `data/Lecta/interim/lecta_valid.csv`
- `data/Lecta/interim/lecta_test.csv`
- `data/Lecta/interim/lecta_metrics.csv`
- `data/Lecta/interim/lecta_test_predictions.csv`
- `data/Lecta/interim/lecta_catboost_model.cbm`
- `data/Lecta/interim/lecta_feature_metadata.json`
- `results/lecta_feature_importance.csv`

---

## Resumen

El proyecto construye un modelo global de series temporales semanales usando features que representan:

- **histórico reciente**
- **nivel y dispersión**
- **tendencia y momentum**
- **estacionalidad**
- **intermitencia**
- **spikes o anomalías**
- **escalado por serie**

Estas variables se usan para entrenar un `CatBoostRegressor` sobre un target escalado, con evaluación temporal y comparación contra baselines simples.
