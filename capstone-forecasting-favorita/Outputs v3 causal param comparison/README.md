# Caché de la búsqueda Optuna (T1–T3)

Artefactos de la búsqueda de hiperparámetros ejecutada sobre las ventanas T1, T2 y T3. Con
`REUSE_XGB_TUNING = True`, la sección 8 del notebook los lee en lugar de repetir la
optimización.

| Archivo | Uso en el pipeline |
|---|---|
| `pipeline_metadata.json` | Se lee la clave `xgb_tuned_params` |
| `tables/xgb_default_vs_optuna_windows.csv` | Métricas Base vs. Optuna por ventana |
| `tables/xgb_default_vs_optuna_summary.csv` | Medias de ambas configuraciones |
| `tables/xgb_optuna_trials.csv` | Los 10 ensayos de la búsqueda |

Si falta alguno de los tres primeros, la celda lanza `FileNotFoundError`.

Los hiperparámetros corresponden al ensayo 6 de `xgb_optuna_trials.csv`, con valor objetivo
0,4196141434201315 — el mínimo de la búsqueda y coincidente con `mean_rmsle` de la fila
`Optuna` del resumen.
