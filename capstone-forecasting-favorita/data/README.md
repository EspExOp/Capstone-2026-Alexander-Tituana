# Datos

Los archivos de datos no se versionan: son públicos y superan el tamaño razonable
para un repositorio.

Descargar de la competencia de Kaggle
[Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
y colocar en esta carpeta:

| Archivo | Registros | Contenido |
|---|---|---|
| `train.csv` | 3.000.888 | Ventas diarias por tienda y familia, 2013-01-01 a 2017-08-15 |
| `test.csv` | 28.512 | Horizonte de 16 días, 2017-08-16 a 2017-08-31 |
| `stores.csv` | 54 | Metadatos de tienda: tipo, ciudad, estado, cluster |
| `oil.csv` | 1.218 | Precio diario del petróleo WTI |
| `holidays_events.csv` | 350 | Calendario de feriados y eventos de Ecuador |
| `transactions.csv` | 83.488 | Transacciones diarias por tienda (solo diagnóstico) |

El notebook valida la presencia de los seis archivos en su primera celda de carga.
