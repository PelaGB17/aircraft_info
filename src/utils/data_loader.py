"""
Módulo para cargar y gestionar los datos ADS-B desde archivos CSV
"""

import pandas as pd
from pathlib import Path
from src.utils.logger import configure_logger

logger = configure_logger("data_loader")

def load_raw_data(input_dir: Path) -> pd.DataFrame:
    """
    Carga todos los archivos CSV de la carpeta especificada y los combina en un DataFrame.
    
    Args:
        input_dir (Path): Ruta al directorio que contiene los archivos CSV.
        
    Returns:
        pd.DataFrame: DataFrame combinado con todos los datos cargados.
    """
    try:
        logger.info(f"Cargando datos desde el directorio: {input_dir}")
        
        # Verificar si el directorio existe
        if not input_dir.exists():
            logger.error(f"El directorio {input_dir} no existe.")
            raise FileNotFoundError(f"El directorio {input_dir} no existe.")
        
        # Buscar archivos CSV en el directorio
        csv_files = list(input_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No se encontraron archivos CSV en el directorio {input_dir}.")
            return pd.DataFrame()  # Retornar un DataFrame vacío
        
        logger.info(f"Archivos encontrados: {[file.name for file in csv_files]}")
        
        # Cargar y combinar todos los archivos CSV
        dataframes = [pd.read_csv(file) for file in csv_files]
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        logger.info(f"Datos cargados exitosamente. Total de registros: {len(combined_df)}")
        
        return combined_df
    
    except Exception as e:
        logger.error(f"Error al cargar datos: {str(e)}")
        raise

def save_data(df: pd.DataFrame, output_file: Path) -> None:
    """
    Guarda un DataFrame en un archivo CSV.
    
    Args:
        df (pd.DataFrame): DataFrame a guardar.
        output_file (Path): Ruta del archivo de salida.
        
    Returns:
        None
    """
    try:
        logger.info(f"Guardando datos en el archivo: {output_file}")
        
        # Guardar el DataFrame en formato CSV
        df.to_csv(output_file, index=False)
        
        logger.info("Datos guardados exitosamente.")
    
    except Exception as e:
        logger.error(f"Error al guardar datos: {str(e)}")
        raise

if __name__ == "__main__":
    # Ejemplo de uso
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    try:
        raw_data = load_raw_data(Path(RAW_DATA_DIR))
        
        if not raw_data.empty:
            output_file = Path(PROCESSED_DATA_DIR) / "combined_raw_data.csv"
            save_data(raw_data, output_file)
    except Exception as e:
        logger.critical(f"Fallo en la carga o guardado de datos: {str(e)}")
