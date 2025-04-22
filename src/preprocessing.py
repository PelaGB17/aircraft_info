"""
Módulo para preprocesamiento de datos ADS-B
"""
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
import numpy as np
from src.config import PROCESSED_DATA_DIR, PREPROCESSING_VIZ_DIR
from pathlib import Path
from src.utils.logger import configure_logger
from src.utils.data_loader import load_raw_data

logger = configure_logger("preprocessing")

# Mapeo de nombres de columnas de los archivos originales a los nombres estándar usados en el código
COLUMN_MAPPING = {
    'alt_geom': 'altgeom',
    'addr': 'hex',
    'gs': 'speed',
    'track': 'track',
    'category': 'category',
    'lat': 'lat',
    'lon': 'lon',
    'timestamp': 'timestamp',
    'flight': 'flight'
}

# Configuración de columnas esenciales
ESSENTIAL_COLUMNS = [
    'timestamp', 'lat', 'lon', 'altgeom', 'speed', 
    'track', 'category', 'hex', 'flight'
]

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra las columnas según el diccionario COLUMN_MAPPING"""
    logger.info("Renombrando columnas según mapeo definido")
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    return df

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y filtra datos esenciales"""
    logger.info("Validando estructura de datos")
    
    # Verificar columnas esenciales
    missing_cols = [col for col in ESSENTIAL_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.error(f"Columnas esenciales faltantes: {missing_cols}")
        raise ValueError("Datos incompletos - columnas esenciales faltantes")
    
    # Filtrar datos inválidos
    initial_count = len(df)
    df = df[~df['lat'].isna() & ~df['lon'].isna()]
    df = df[(df['lat'].between(-90, 90)) & (df['lon'].between(-180, 180))]
    
    # Filtrar coordenadas con valor 0.0
    df = df[(df['lat'] != 0.0) & (df['lon'] != 0.0)]
    
    logger.info(f"Datos filtrados: {initial_count} -> {len(df)} registros")
    return df

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza formatos y tipos de datos"""
    logger.info("Normalizando datos")
    
    # Conversión de tipos
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['flight'] = df['flight'].str.strip().replace('', np.nan)
    
    # Normalización de altitudes
    df['altgeom'] = df['altgeom'].clip(lower=0)
    
    # Codificación de categorías
    category_map = {
        0: 'No info',
        1: 'Light',
        2: 'Medium',
        3: 'Heavy'
    }
    df['category'] = df['category'].map(category_map).fillna('Unknown')
    
    return df

def visualize_aircraft_distribution(df, OUTPUT_DIR: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Diagrama de dispersión con transparencia para ver concentraciones
    scatter = ax.scatter(df['lon'], df['lat'], 
                         c=df['altgeom'], cmap='viridis', 
                         alpha=0.6, s=30, edgecolor='w', linewidth=0.5)
    
    # Añadir mapa base
    ctx.add_basemap(ax, crs='EPSG:4326')
    
    # Configuraciones
    plt.colorbar(scatter, label='Altitud (pies)')
    ax.set_title('Distribución espacial de aeronaves')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spatial_distribution.png")
    plt.close()
    
    logger.info(f"Visualización de distribución espacial generada")

def process_raw_data(input_dir: Path, output_dir: Path) -> None:
    """Procesa todos los archivos RAW y guarda resultados"""
    try:
        logger.info(f"Iniciando preprocesamiento desde: {input_dir}")
        
        # Asegurar que el directorio de salida existe
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = load_raw_data(input_dir)
        
        # Etapa de renombrado
        df = rename_columns(df)
        
        df = validate_data(df)
        df = normalize_data(df)
        
        # Guardar datos procesados - crear un nombre de archivo específico
        output_file = output_dir / "processed_data.parquet"
        df.to_parquet(output_file)

        # Asegurar que el directorio de visualizaciones existe
        Path(PREPROCESSING_VIZ_DIR).mkdir(parents=True, exist_ok=True)
        
        # Generar visualizaciones
        visualize_aircraft_distribution(df, PREPROCESSING_VIZ_DIR)

        logger.info(f"Datos procesados guardados en: {output_file}")
        
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {str(e)}")
        raise

if __name__ == "__main__":
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    try:
        process_raw_data(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    except Exception as e:
        logger.critical(f"Fallo en el preprocesamiento: {str(e)}")
        exit(1)
