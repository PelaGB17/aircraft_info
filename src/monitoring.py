"""
Módulo para monitorización pasiva de aeronaves (prioridades 2 y 3)
"""
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from src.utils.logger import configure_logger
from src.config import MONITORING_VIZ_DIR

logger = configure_logger("monitoring")

def monitor_aircraft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la monitorización pasiva de aeronaves con prioridad 3, 4 y 5.
    """
    logger.info("Iniciando monitorización pasiva de aeronaves")

    # Crear columna 'status' con valor por defecto para todas las aeronaves
    df['status'] = 'no monitoreada'  # Valor por defecto

    # Filtrar aeronaves de prioridad 3, 4 y 5 para monitorizar
    monitored_df = df[df['priority'].isin(['priority3', 'priority4', 'priority5'])].copy()

    if monitored_df.empty:
        logger.warning("No hay aeronaves de prioridad 3, 4 y 5 para monitorizar")
        return df

    # Calcular estado para aeronaves monitoreadas
    monitored_indices = monitored_df.index
    df.loc[monitored_indices, 'status'] = monitored_df.apply(
        lambda row: evaluate_status(row['distance'], row['speed']),
        axis=1
    )
    
    # Registrar estadísticas básicas
    priority_3_count = (df['priority'] == 'priority3').sum()
    priority_4_count = (df['priority'] == 'priority4').sum()
    priority_5_count = (df['priority'] == 'priority5').sum()
    
    logger.info(f"Aeronaves monitorizadas - Prioridad 3: {priority_3_count}, Prioridad 4: {priority_4_count}, Prioridad 5: {priority_5_count}")
    
    return df


def evaluate_status(distance: float, speed: float) -> str:
    """
    Evalúa el estado de una aeronave basada en su distancia y velocidad.
    
    Args:
        distance (float): Distancia entre la aeronave y un punto de referencia (en km).
        speed (float): Velocidad de la aeronave (en knots).
        
    Returns:
        str: Estado evaluado ("estable", "potencial riesgo", etc.).
    """
    if distance < 10 and speed > 200:
        return "potencial riesgo"
    elif distance < 20:
        return "precaución"
    else:
        return "estable"
    
# Código para visualización de mapa de estados
def plot_status_map(df, RESULTS_DIR: str) -> None:
    try:
        # Verificar que exista la columna 'status'
        if 'status' not in df.columns:
            logger.warning("Columna 'status' no encontrada en los datos. Generando valores por defecto.")
            df['status'] = 'no monitoreada'
            
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Crear máscaras para cada estado
        estable = df['status'] == 'estable'
        precaucion = df['status'] == 'precaución'
        riesgo = df['status'] == 'potencial riesgo'
        otros = ~(estable | precaucion | riesgo)
        
        # Graficar por estado con colores diferenciados
        if estable.any():
            ax.scatter(df[estable]['lon'], df[estable]['lat'], c='green', alpha=0.7, 
                      label='Estable', edgecolor='white', linewidth=0.5)
        if precaucion.any():
            ax.scatter(df[precaucion]['lon'], df[precaucion]['lat'], c='orange', alpha=0.7, 
                      label='Precaución', edgecolor='white', linewidth=0.5)
        if riesgo.any():
            ax.scatter(df[riesgo]['lon'], df[riesgo]['lat'], c='red', alpha=0.7, 
                      label='Potencial riesgo', edgecolor='white', linewidth=0.5)
        if otros.any():
            ax.scatter(df[otros]['lon'], df[otros]['lat'], c='gray', alpha=0.3, 
                      label='No monitoreada', edgecolor='white', linewidth=0.5)
        
        # Añadir mapa base
        ctx.add_basemap(ax, crs='EPSG:4326')
        
        ax.set_title('Monitorización pasiva de aeronaves: Estado actual')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/monitoring_status_map.png")
        plt.close()
        
        logger.info(f"Mapa de estados de monitorización generado en {RESULTS_DIR}")
    except Exception as e:
        logger.error(f"Error al generar mapa de estados: {str(e)}")

def process_monitoring(input_file: str, output_file: str) -> None:
    """
    Procesa la monitorización pasiva y guarda los resultados en un archivo.
    
    Args:
        input_file (str): Ruta del archivo con datos clasificados por prioridad.
        output_file (str): Ruta para guardar los resultados del monitoreo.
        
    Returns:
        None
    """
    try:
        logger.info(f"Cargando datos desde: {input_file}")
        
        # Cargar datos clasificados por prioridad
        df = pd.read_parquet(input_file)
        
        # Realizar la monitorización pasiva
        monitored_df = monitor_aircraft(df)
        
        # Guardar resultados actualizados
        monitored_df.to_parquet(output_file)
        plot_status_map(monitored_df, MONITORING_VIZ_DIR)
        logger.info(f"Resultados del monitoreo guardados en: {output_file}")
        
    except Exception as e:
        logger.error(f"Error en proceso de monitorización: {str(e)}")
        raise

if __name__ == "__main__":
    from src.config import PRIORITY_OUTPUT_FILE, MONITORING_OUTPUT_FILE
    
    try:
        process_monitoring(PRIORITY_OUTPUT_FILE, MONITORING_OUTPUT_FILE)
    except Exception as e:
        logger.critical(f"Fallo en el proceso de monitorización: {str(e)}")
        exit(1)
