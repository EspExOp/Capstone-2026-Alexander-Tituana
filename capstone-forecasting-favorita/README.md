# Forecasting causal de demanda — Corporación Favorita

Pipeline global de pronóstico de demanda para las 1.782 series diarias tienda-familia de
Corporación Favorita (Ecuador, 2013–2017), con validación temporal causal, interpretabilidad
SHAP y traducción a decisiones de inventario mediante un modelo Newsvendor.

Anexo de código del proyecto Capstone *"Forecasting de demanda con modelos de gradient
boosting (XGBoost y LightGBM), variables exógenas y validación causal: implicaciones para la
gestión de inventario en Corporación Favorita"*.

**Autor:** Alexander David Tituaña Casa · Maestría en Inteligencia de Negocios y Ciencia de Datos

---

## Resultados principales

Evaluación recursiva sobre el holdout (1–15 de agosto de 2017), no utilizado en ninguna
decisión de modelado:

| Modelo | RMSLE | MAE | WAPE |
|---|---|---|---|
| Seasonal Naive | 0,6124 | 98,69 | 21,22% |
| **XGBoost (Optuna)** | **0,4058** | **82,54** | **17,75%** |
| LightGBM (estándar) | 0,4099 | 82,82 | 17,81% |

Reducción frente al baseline: **−33,74% en RMSLE** y **−16,37% en MAE y WAPE**.

Simulación Newsvendor con fractil empírico del percentil 95: safety stock −35,25%
(321.359 → 208.084 unidades), faltantes −32,83%, excedentes −15,06%. La cobertura alcanzada
fue 92,41%, por debajo del 95% objetivo.

Ningún algoritmo dominó todos los regímenes temporales: LightGBM obtuvo el mejor RMSLE en la
ventana de estrés petrolero de enero de 2016 (0,6430 frente a 1,0222 de XGBoost).

---

## Diseño causal

El pipeline preserva el orden temporal y la disponibilidad real de la información:

- **`shift(1)` en toda variable derivada del target.** Rezagos, medias móviles, desviaciones y
  estadísticos expansivos se calculan dentro de cada serie con desplazamiento de un día, de
  modo que la venta del día objetivo nunca participa en sus propias features. Verificado por
  pruebas unitarias (`run_causality_tests`).
- **WTI congelado** en el último valor conocido antes de cada origen de pronóstico. El precio
  contemporáneo futuro queda excluido de las features.
- **Pronóstico recursivo.** Cada predicción se incorpora a la matriz de historia y alimenta
  los rezagos del día siguiente. No se imputan ventas reales dentro del horizonte.
- **`transactions` excluida** por no existir en el conjunto de prueba.
- Entrenamiento sobre `log1p(sales)` y predicciones vía `expm1` con recorte en cero.

### Ventanas de evaluación

| Ventana | Período | Función |
|---|---|---|
| Stress WTI | 2016-01-01 → 01-15 | Robustez durante el colapso petrolero |
| T1 | 2016-10-01 → 10-16 | Optimización de hiperparámetros |
| T2 | 2017-03-01 → 03-16 | Optimización de hiperparámetros |
| T3 | 2017-06-01 → 06-16 | Optimización de hiperparámetros |
| B1 | 2017-06-30 → 07-15 | Selección de familia y calibración de inventario |
| B2 | 2017-07-16 → 07-31 | Selección de familia y calibración de inventario |
| Holdout | 2017-08-01 → 08-15 | Evaluación externa, consultada una sola vez |

Tres decisiones separadas: familia de modelo en B1–B2, hiperparámetros en T1–T3, evaluación
en el holdout. Las ventanas de tuning son anteriores a las de selección y al holdout.

---

## Estructura

```
.
├─ capstone_pipeline_vCPU.ipynb          Notebook completo, con salidas de ejecución
├─ requirements.txt
├─ data/                                 Vacío: descargar los CSV de Kaggle
└─ Outputs v3 causal param comparison/   Caché de la búsqueda Optuna sobre T1-T3
   ├─ pipeline_metadata.json
   └─ tables/
      ├─ xgb_default_vs_optuna_windows.csv
      ├─ xgb_default_vs_optuna_summary.csv
      └─ xgb_optuna_trials.csv
```

El notebook se entrega **con las salidas guardadas**: todos los resultados pueden consultarse
sin ejecutarlo.

---

## Datos

Dataset público *Store Sales — Time Series Forecasting* (Corporación Favorita):
https://www.kaggle.com/competitions/store-sales-time-series-forecasting

Descargar los seis archivos en `data/`: `train.csv`, `test.csv`, `stores.csv`, `oil.csv`,
`holidays_events.csv`, `transactions.csv`. No se versionan en el repositorio por tamaño y por
ser de acceso público.

Serie WTI complementaria: Federal Reserve Economic Data (FRED), https://fred.stlouisfed.org

---

## Ejecución

```bash
pip install -r requirements.txt
jupyter lab capstone_pipeline_vCPU.ipynb
```

Antes de ejecutar, ajustar la ruta en la primera celda de código:

```python
DATA_DIR = Path(r'C:\Users\David\Documents\Capston\data')   # ← cambiar a la ruta local
```

`PROJECT_DIR` se deriva como `DATA_DIR.parent`, de modo que la carpeta
`Outputs v3 causal param comparison/` debe quedar al mismo nivel que `data/`.

La ejecución completa toma varias horas en CPU. Semilla fija `SEED = 42`.

### Banderas relevantes

| Bandera | Valor | Efecto |
|---|---|---|
| `RUN_TUNING` | `False` | No re-ejecuta la búsqueda Optuna |
| `REUSE_XGB_TUNING` | `True` | Lee los hiperparámetros de la caché en `TUNING_CACHE_DIR` |
| `RUN_BACKTEST` | `True` | Evalúa las cuatro ventanas de backtest |
| `RUN_MODEL_COMPARISON` | `True` | Compara XGBoost, LightGBM y Naive |

La búsqueda Optuna se ejecutó una vez sobre T1–T3 con 10 ensayos TPE y sus resultados se
conservaron en `Outputs v3 causal param comparison/`, de modo que reejecutar secciones previas
del pipeline no repite la optimización. Poner `RUN_TUNING = True` y `REUSE_XGB_TUNING = False`
vuelve a lanzar la búsqueda; los hiperparámetros resultantes pueden diferir de los reportados.

---

## Hiperparámetros seleccionados

Ensayo 6 de 10, con RMSLE medio de 0,4196 sobre T1–T3:

```json
{
  "n_estimators": 609,
  "max_depth": 6,
  "learning_rate": 0.14108340900523528,
  "subsample": 0.9212964881763901,
  "colsample_bytree": 0.9788246295474662,
  "min_child_weight": 18,
  "reg_alpha": 0.03728942986384152,
  "reg_lambda": 4.869640941520899
}
```

La optimización redujo el RMSLE medio de T1–T3 de 0,4417 a 0,4196 (−5,0%), pero la mejora no
fue homogénea: en B1–B2 la configuración estándar obtuvo un RMSLE medio de 0,3869 frente a
0,3940 de la optimizada. Esta sensibilidad al régimen temporal se documenta en el informe.

---

## Entorno

Python 3.13.5 · XGBoost 3.4.1 · LightGBM 4.7.0

Existe una variante del pipeline preparada para XGBoost sobre CUDA. **No se incluye en este
repositorio**: produce hiperparámetros y métricas distintos a los reportados en el documento,
por lo que el anexo se limita a la versión que generó los resultados publicados.

---

## Limitaciones

- Datos de 2013–2017; la aplicación operativa requiere validación con información reciente.
- Los precios unitarios (USD 5–15) y la tasa de mantenimiento (25% anual) son supuestos de
  sensibilidad: el dataset no contiene costos ni márgenes por SKU. Las cifras monetarias son
  escenarios, no ahorros contables.
- La cobertura del 92,41% queda por debajo del objetivo del 95%; el fractil de calibración no
  garantiza cobertura fuera de muestra.
- El análisis SHAP describe asociaciones predictivas, no relaciones causales.
- El holdout de agosto de 2017 ya fue consumido y no debe reutilizarse para nuevos ajustes.

---

## Licencia

Uso académico. El dataset pertenece a Corporación Favorita y se distribuye bajo los términos
de la competencia de Kaggle.
