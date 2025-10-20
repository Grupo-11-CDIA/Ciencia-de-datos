# Proyecto Integrado: Análisis y Modelado de Ventas en un Petshop

Resumen
-------
Este repositorio contiene el proyecto final del grupo 11 para las materias "Ciencia de Datos II" y "Estadística y Exploración de Datos II". A partir de un dataset de ventas de una tienda de mascotas (petshop), se aplican técnicas de limpieza, análisis estadístico y modelos predictivos (regresión lineal, regresión múltiple y logística) con el objetivo de extraer información útil para la toma de decisiones comerciales.

Ejes temáticos / Unidades conceptuales
-------------------------------------
El proyecto integra los contenidos de ambas materias del cuatrimestre:
- Ciencia de Datos II: Numpy, Pandas, Introducción a Machine Learning, correlación, regresión lineal, regresión lineal múltiple y regresión logística.
- Estadística y Exploración de Datos II: estimación de parámetros, contraste de hipótesis, ANOVA, correlación, regresión lineal y regresión logística.

Problemática / Caso de estudio
------------------------------
Dado un conjunto de datos de ventas de un petshop (productos, categorías, precios, fechas, cantidades vendidas, etc.), se busca:
- Obtener información descriptiva y estadística relevante.
- Validar hipótesis acerca de relaciones entre variables.
- Construir modelos predictivos que permitan inferir montos/volúmenes de venta y apoyar decisiones operativas y estratégicas.

Fundamentación / Hipótesis
--------------------------
El proyecto parte de la necesidad de aplicar de forma integrada conocimientos estadísticos y de ciencia de datos en un contexto real. Hipótesis de trabajo (ejemplos):
- El precio y la categoría del producto influyen significativamente en el monto vendido.
- Existen diferencias significativas en los montos de venta según el tipo de mascota (perro, gato, etc.).
- Un conjunto reducido de variables explicativas permite predecir el monto de venta con precisión razonable mediante regresión múltiple.

Objetivo general
----------------
Analizar y modelar el comportamiento de las ventas del petshop mediante herramientas estadísticas y técnicas de ciencia de datos para identificar factores que influyen en las ventas y desarrollar modelos predictivos útiles para la toma de decisiones.

Objetivos específicos
---------------------
1. Explorar y depurar el dataset: detectar nulos, duplicados, inconsistencias y atípicos.
2. Realizar análisis descriptivo numérico y categórico (medidas de tendencia y visualizaciones).
3. Evaluar normalidad y homocedasticidad (Shapiro-Wilk, Levene, QQ-plots).
4. Aplicar ANOVA (una vía y dos vías) para contrastar diferencias en montos por categoría y tipo de mascota.
5. Analizar correlaciones (Pearson / Spearman) y matrices de correlación.
6. Ajustar modelos de regresión lineal y múltiple; validar supuestos y evaluar rendimiento.
7. Interpretar resultados y proponer recomendaciones comerciales.

Cronograma (resumen)
--------------------
Semana 1
- Importar y limpiar dataset (Pandas, Numpy).  
Semana 2
- Análisis descriptivo y pruebas de normalidad/homocedasticidad (Scipy, Statsmodels).  
Semana 3
- ANOVA y análisis de correlación (Statsmodels, Seaborn).  
Semana 4
- Modelos de regresión (Statsmodels, Scikit-Learn); validación y diagnóstico.  
Semana 5
- Interpretación de resultados y redacción de conclusiones.  
Semana 6
- Integración, entrega final y exposición.

Estructura sugerida del repositorio
-----------------------------------
- data/                -> datasets (raw/processed)
- notebooks/           -> notebooks con análisis paso a paso (exploratorio, pruebas, modelos)
- src/                 -> scripts y módulos reutilizables
- reports/             -> gráficos, tablas y el informe final
- requirements.txt     -> dependencias del proyecto
- README.md            -> este documento

Requisitos / Instalación
------------------------
Recomendado: Python 3.9+ y virtualenv o conda.

1. Clonar el repositorio:
   git clone https://github.com/Grupo-11-CDIA/Ciencia-de-datos.git

2. Crear y activar entorno (ejemplo con venv):
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\.venv\Scripts\activate    # Windows

3. Instalar dependencias:
   pip install -r requirements.txt

Dependencias típicas
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- scipy
- jupyterlab / notebook

Uso / Reproducción
------------------
- Abrir los notebooks en `notebooks/` con Jupyter o ejecutar en Google Colab (se incluyen enlaces a Colab si se desea).
- Flujo de trabajo recomendado:
  1. data/ -> raw: colocar el CSV original (por ejemplo, dataset desde Kaggle).
  2. Ejecutar notebook de preprocesamiento para generar data/processed.
  3. Ejecutar notebook de EDA (estadísticas descriptivas y visualizaciones).
  4. Ejecutar notebooks de pruebas estadísticas (Shapiro-Wilk, Levene, ANOVA).
  5. Ejecutar notebooks de modelado (regresión lineal, múltiple y logística) y scripts de evaluación.

Metodología (resumen)
---------------------
1. Limpieza y transformación: imputación o eliminación de nulos, tratamiento de outliers, normalización/encodings categóricos.
2. Análisis exploratorio: estadísticas descriptivas, visualizaciones, análisis por grupos.
3. Pruebas estadísticas: normalidad (Shapiro-Wilk), homogeneidad de varianzas (Levene), ANOVA.
4. Correlación: coeficientes Pearson/Spearman y matriz de correlación.
5. Modelado: selección de variables, ajuste de modelos (Statsmodels/Scikit-Learn), evaluación por R², MAE, RMSE y validación cruzada.
6. Interpretación y recomendaciones comerciales.

Resultados esperados
--------------------
- Un set de notebooks que documente todo el proceso (EDA, pruebas, modelos).
- Visualizaciones clave (distribuciones, boxplots, heatmaps).
- Modelos explicativos y predictivos con interpretación de coeficientes y recomendaciones accionables para el petshop.

Buenas prácticas y reproducibilidad
----------------------------------
- Mantener los notebooks reproducibles (celdas que cargan datos y generan outputs).
- Versionar los datos procesados en data/processed o describir claramente cómo generarlos desde raw.
- Usar archivos con dependencias (requirements.txt) y proporcionar instrucciones para ejecutarlo en Colab.

Fuentes y referencias
---------------------
- Dataset (origen): Kaggle.
- Documentación de las librerías utilizadas: Pandas, Numpy, Scikit-learn, Statsmodels, Scipy, Seaborn.
