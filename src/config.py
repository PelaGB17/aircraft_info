"""
Módulo de configuración global del sistema
"""

from pathlib import Path

# Directorios de datos
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
VISUALIZATION_DIR = DATA_DIR / "visualizations"
PRIORITY_VIZ_DIR = VISUALIZATION_DIR / "priority"
FUZZY_VIZ_DIR = VISUALIZATION_DIR / "fuzzy_evaluation"
BAYESIAN_VIZ_DIR = VISUALIZATION_DIR / "bayesian"
MONITORING_VIZ_DIR = VISUALIZATION_DIR / "monitoring"
PREPROCESSING_VIZ_DIR = VISUALIZATION_DIR / "preprocessing"
RECOMMENDATIONS_VIZ_DIR = VISUALIZATION_DIR / "recommendations"

# Archivos específicos
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.parquet"
PRIORITY_OUTPUT_FILE = RESULTS_DIR / "priority_data.parquet"
FUZZY_OUTPUT_FILE = RESULTS_DIR / "fuzzy_evaluation_data.parquet"
BAYESIAN_OUTPUT_FILE = RESULTS_DIR / "bayesian_evaluation_data.parquet"
RECOMMENDATIONS_OUTPUT_FILE = RESULTS_DIR / "recommendations_data.parquet"
MONITORING_OUTPUT_FILE = RESULTS_DIR / "monitoring_data.parquet"

# Configuración de logging
LOG_DIR = Path("logs")

# Parámetros generales del sistema
DASHBOARD_PORT = 8050
MAX_DISTANCE_THRESHOLD = 20  # Distancia máxima en km para considerar aeronaves relevantes
MIN_SPEED_THRESHOLD = 150    # Velocidad mínima en knots para evaluar riesgo

if __name__ == "__main__":
    # Ejemplo de uso: imprimir configuraciones
    print(f"Directorio de datos: {DATA_DIR}")
    print(f"Archivo procesado: {PROCESSED_DATA_FILE}")
    print(f"Directorio de logs: {LOG_DIR}")
