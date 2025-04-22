"""
Script principal para ejecutar el flujo completo del sistema de detección de colisiones
"""
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, LOG_DIR, VISUALIZATION_DIR, PREPROCESSING_VIZ_DIR, PRIORITY_VIZ_DIR, FUZZY_VIZ_DIR, BAYESIAN_VIZ_DIR, MONITORING_VIZ_DIR, RECOMMENDATIONS_VIZ_DIR
from src.config import RAW_DATA_DIR, PROCESSED_DATA_FILE, PRIORITY_OUTPUT_FILE, FUZZY_OUTPUT_FILE, BAYESIAN_OUTPUT_FILE, RECOMMENDATIONS_OUTPUT_FILE, MONITORING_OUTPUT_FILE
from src.preprocessing import process_raw_data
from src.priority import process_priority
from src.fuzzy_evaluation import process_fuzzy_evaluation
from src.bayesian import process_bayesian_evaluation
from src.monitoring import process_monitoring
from src.recommendations import process_recommendations
from src.utils.logger import configure_logger

logger = configure_logger("main")

def create_required_directories():
    """
    Crea todos los directorios necesarios para la ejecución del sistema.
    """
    # Lista de directorios a crear
    directories = [
        RAW_DATA_DIR, 
        PROCESSED_DATA_DIR, 
        RESULTS_DIR, 
        LOG_DIR,
        VISUALIZATION_DIR,
        PREPROCESSING_VIZ_DIR,
        PRIORITY_VIZ_DIR,
        FUZZY_VIZ_DIR,
        BAYESIAN_VIZ_DIR,
        MONITORING_VIZ_DIR,
        RECOMMENDATIONS_VIZ_DIR,
    ]
    
    # Crear cada directorio si no existe
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio asegurado: {directory}")

def main():
    """
    Ejecuta el flujo completo del sistema de detección de colisiones.
    """
    try:
        logger.info("Iniciando ejecución del sistema de detección de colisiones")

        create_required_directories()

        # 1. Preprocesamiento de datos brutos ADS-B
        logger.info("Paso 1: Preprocesamiento de datos brutos")
        process_raw_data(Path(RAW_DATA_DIR), Path(PROCESSED_DATA_DIR))

        # 2. Clasificación por prioridad
        logger.info("Paso 2: Clasificación por prioridad")
        process_priority(PROCESSED_DATA_FILE, PRIORITY_OUTPUT_FILE)

        # 3. Evaluación borrosa para aeronaves de prioridad 1
        logger.info("Paso 3: Evaluación borrosa")
        process_fuzzy_evaluation(PRIORITY_OUTPUT_FILE, FUZZY_OUTPUT_FILE)

        # 4. Cálculo bayesiano para probabilidad de colisión
        logger.info("Paso 4: Evaluación bayesiana")
        process_bayesian_evaluation(FUZZY_OUTPUT_FILE, BAYESIAN_OUTPUT_FILE)

        # 5. Monitorización pasiva para prioridades 3, 4 y 5
        logger.info("Paso 5: Monitorización pasiva")
        process_monitoring(PRIORITY_OUTPUT_FILE, MONITORING_OUTPUT_FILE)

        # 6. Generación de recomendaciones basadas en análisis previos
        logger.info("Paso 6: Generación de recomendaciones")
        process_recommendations(BAYESIAN_OUTPUT_FILE, RECOMMENDATIONS_OUTPUT_FILE)

    except Exception as e:
        logger.critical(f"Fallo en la ejecución del sistema: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
