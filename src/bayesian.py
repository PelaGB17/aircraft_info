"""
Módulo para cálculo de probabilidad bayesiana de colisión
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from src.utils.logger import configure_logger
from src.utils.geometry import calculate_relative_velocity
from src.config import BAYESIAN_VIZ_DIR

logger = configure_logger("bayesian")

# Parámetros ajustados para cálculo bayesiano
DISTANCE_PARAMS = {"loc": 20, "scale": 30}  # Ajustado para distancias más realistas
SPEED_PARAMS = {"loc": 300, "scale": 200}     # Ajustado para velocidades de cierre
TIME_PARAMS = {"loc": 300, "scale": 300}     # Ajustado para tiempos de conflicto
MAX_TIME_TO_CONFLICT = 1800  # 30 minutos como valor máximo razonable

def calculate_bayesian_probability(distance: float, speed: float, time_to_conflict: float) -> float:
    """
    Calcula la probabilidad bayesiana de colisión basada en distancia, velocidad y tiempo al conflicto.
    
    Args:
        distance (float): Distancia entre aeronaves en km.
        speed (float): Velocidad relativa en knots.
        time_to_conflict (float): Tiempo estimado al conflicto en segundos.
        
    Returns:
        float: Probabilidad de colisión entre 0 y 1.
    """
    try:
        # Validación de parámetros
        if not isinstance(distance, (int, float)) or distance < 0:
            logger.warning(f"Distancia inválida: {distance}, usando valor por defecto")
            distance = DISTANCE_PARAMS["loc"]
            
        if not isinstance(speed, (int, float)) or speed < 0:
            logger.warning(f"Velocidad inválida: {speed}, usando valor por defecto")
            speed = SPEED_PARAMS["loc"] / 2
            
        if not isinstance(time_to_conflict, (int, float)) or time_to_conflict < 0:
            logger.warning(f"Tiempo inválido: {time_to_conflict}, usando valor por defecto")
            time_to_conflict = MAX_TIME_TO_CONFLICT / 2
            
        # Probabilidad basada en distancia (más probable si es pequeña)
        distance_prob = 1 - norm.cdf(distance, loc=DISTANCE_PARAMS["loc"], scale=DISTANCE_PARAMS["scale"])
        
        # Probabilidad basada en velocidad (más probable si es alta)
        speed_prob = norm.cdf(speed, loc=SPEED_PARAMS["loc"], scale=SPEED_PARAMS["scale"])
        
        # Probabilidad basada en tiempo (más probable si es corto)
        time_prob = 1 - norm.cdf(min(time_to_conflict, MAX_TIME_TO_CONFLICT), 
                                loc=TIME_PARAMS["loc"], scale=TIME_PARAMS["scale"])
        
        # Combinar probabilidades (producto bayesiano)
        bayesian_prob = distance_prob * speed_prob * time_prob
        
        # Registrar valores de diagnóstico para valores atípicos
        if bayesian_prob > 0.8 or bayesian_prob < 0.01:
            logger.debug(f"Valores: dist={distance:.2f}, speed={speed:.2f}, time={time_to_conflict:.2f}")
            logger.debug(f"Probs: dist={distance_prob:.4f}, speed={speed_prob:.4f}, time={time_prob:.4f}, total={bayesian_prob:.6f}")
            
        return min(max(bayesian_prob, 0), 1)  # Asegurar que esté entre 0 y 1
    
    except Exception as e:
        logger.error(f"Error en cálculo bayesiano: {str(e)}")
        return 0.0

def calculate_time_to_conflict(row: pd.Series) -> float:
    try:
        # Asegurar conversión numérica adecuada
        distance = pd.to_numeric(row.get('distance', 0), errors='coerce')
        closing_speed = pd.to_numeric(row.get('closing_speed', 0), errors='coerce')
        
        # Manejar valores NaN
        if pd.isna(distance) or pd.isna(closing_speed):
            return float('inf')
        
        # Si la distancia es muy grande, probablemente no sea un conflicto real
        if row['distance'] > 200:  # 200 km es un umbral razonable
            return MAX_TIME_TO_CONFLICT
            
        # Si la velocidad de cierre es muy pequeña
        closing_speed = row.get('closing_speed', 0)
        if closing_speed < 10:  # Umbral más realista
            closing_speed = max(row['speed'] * 0.5, 10)
            
        # Cálculo normal
        time_seconds = (row['distance'] / (closing_speed * 1.852 / 3600))
        return min(time_seconds, MAX_TIME_TO_CONFLICT)
        
    except Exception as e:
        logger.error(f"Error calculando tiempo al conflicto: {str(e)}")
        return MAX_TIME_TO_CONFLICT

def plot_bayesian_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Genera un histograma de probabilidades bayesianas.
    
    Args:
        df (pd.DataFrame): DataFrame con resultados de probabilidad bayesiana.
        output_dir (str): Directorio donde guardar las visualizaciones.
    """
    try:
        # Asegurar que el directorio existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        # Crear histograma con bins personalizados
        bins = np.linspace(0, 1, 41)  # 40 divisiones entre 0 y 1
        plt.hist(df['bayesian_probability'], bins=bins, alpha=0.7, color='purple', 
                 edgecolor='black', linewidth=0.5)
        
        # Añadir líneas para umbrales importantes
        plt.axvline(x=0.4, color='orange', linestyle='--', linewidth=2, label='Umbral de precaución (0.4)')
        plt.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='Umbral de alerta (0.7)')
        
        # Añadir estadísticas
        mean_prob = df['bayesian_probability'].mean()
        max_prob = df['bayesian_probability'].max()
        plt.axvline(x=mean_prob, color='green', linestyle='-', linewidth=2, 
                   label=f'Valor medio ({mean_prob:.4f})')
        
        # Configurar gráfico
        plt.xlabel('Probabilidad bayesiana de colisión', fontsize=12)
        plt.ylabel('Número de aeronaves', fontsize=12)
        plt.title('Distribución de probabilidades bayesianas', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Añadir anotaciones con estadísticas
        plt.annotate(f'Max: {max_prob:.4f}\nMedia: {mean_prob:.4f}\nTotal: {len(df)}', 
                     xy=(0.75, 0.9), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bayesian_probability_distribution.png", dpi=300)
        plt.close()
        
        logger.info(f"Histograma de probabilidades bayesianas generado en {output_dir}")
        
        # Crear scatter plot de distancia vs. probabilidad
        plt.figure(figsize=(12, 8))
        plt.scatter(df['distance'], df['bayesian_probability'], alpha=0.5, c='blue', edgecolor='black')
        plt.xlabel('Distancia (km)', fontsize=12)
        plt.ylabel('Probabilidad bayesiana', fontsize=12)
        plt.title('Relación entre distancia y probabilidad de colisión', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/distance_vs_probability.png", dpi=300)
        plt.close()
        
        logger.info("Gráfico de distancia vs. probabilidad generado")
        
    except Exception as e:
        logger.error(f"Error generando visualizaciones: {str(e)}")

def process_bayesian_evaluation(input_file: str, output_file: str) -> None:
    """
    Procesa evaluación bayesiana para todas las aeronaves clasificadas.
    
    Args:
        input_file (str): Ruta al archivo con datos clasificados y evaluados previamente.
        output_file (str): Ruta para guardar resultados con probabilidad bayesiana.
        
    Returns:
        None
    """
    try:
        # Verificar que los archivos de entrada/salida sean válidos
        if not input_file or not output_file:
            raise ValueError("Las rutas de archivo de entrada y salida deben ser válidas")

        logger.info(f"Cargando datos desde: {input_file}")
        
        # Cargar datos clasificados y evaluados previamente
        df = pd.read_parquet(input_file)

        # Verificar si hay datos
        if df.empty:
            logger.warning("El archivo de entrada no contiene datos")
            return
        
        # Añadir después de cargar el DataFrame
        logger.info(f"Estadísticas de distancia - Min: {df['distance'].min():.2f}, Max: {df['distance'].max():.2f}, Mean: {df['distance'].mean():.2f}")

        # Convertir de metros a kilómetros
        logger.info("Convirtiendo distancias a kilómetros")
        df['distance'] = df['distance'] / 1000
        
        # Verificar si closing_speed existe, si no, calcularla
        if 'closing_speed' not in df.columns:
            logger.info("Calculando velocidad de cierre (closing_speed)")
            df['closing_speed'] = df.apply(
                lambda row: calculate_relative_velocity(row['speed'], row['track']), 
                axis=1
            )
            logger.info(f"Velocidad de cierre media: {df['closing_speed'].mean():.2f} knots")
        
        # Calcular tiempo al conflicto
        logger.info("Calculando tiempo al conflicto")
        df['time_to_conflict'] = df.apply(calculate_time_to_conflict, axis=1)

        # Agregar después del cálculo de time_to_conflict
        logger.info(f"Tiempo al conflicto - Min: {df['time_to_conflict'].min():.2f}, Mean: {df['time_to_conflict'].mean():.2f}")
        
        # Registrar datos de diagnóstico
        inf_values = (df['time_to_conflict'] >= MAX_TIME_TO_CONFLICT).sum()
        logger.info(f"Valores máximos en time_to_conflict: {inf_values} de {len(df)} ({inf_values/len(df)*100:.1f}%)")
        
        # Calcular probabilidad bayesiana para cada aeronave
        logger.info("Calculando probabilidades bayesianas")
        df['bayesian_probability'] = df.apply(
            lambda row: calculate_bayesian_probability(
                row['distance'],
                row['speed'],
                row['time_to_conflict']
            ), 
            axis=1
        )
        
        # Agregar después del cálculo de bayesian_probability para diagnóstico
        if (df['bayesian_probability'] == 0).all():
            # Tomar muestra para diagnóstico
            sample_rows = df.sample(min(5, len(df)))
            for _, row in sample_rows.iterrows():
                # Calcular componentes individuales
                distance_prob = 1 - norm.cdf(row['distance'], loc=DISTANCE_PARAMS["loc"], scale=DISTANCE_PARAMS["scale"])
                speed_prob = norm.cdf(row['speed'], loc=SPEED_PARAMS["loc"], scale=SPEED_PARAMS["scale"])
                time_prob = 1 - norm.cdf(min(row['time_to_conflict'], MAX_TIME_TO_CONFLICT), 
                                    loc=TIME_PARAMS["loc"], scale=TIME_PARAMS["scale"])
                logger.debug(f"Muestra - dist: {row['distance']:.2f}, speed: {row['speed']:.2f}, time: {row['time_to_conflict']:.2f}")
                logger.debug(f"Probs - dist: {distance_prob:.6f}, speed: {speed_prob:.6f}, time: {time_prob:.6f}")

        # Información estadística
        prob_stats = df['bayesian_probability'].describe()
        logger.info(f"Estadísticas de probabilidad bayesiana:\n{prob_stats}")
        
        # Contar aeronaves en diferentes rangos de probabilidad
        low_risk = (df['bayesian_probability'] < 0.4).sum()
        medium_risk = ((df['bayesian_probability'] >= 0.4) & (df['bayesian_probability'] < 0.7)).sum()
        high_risk = (df['bayesian_probability'] >= 0.7).sum()
        
        logger.info(f"Distribución de riesgo: Bajo ({low_risk}), Medio ({medium_risk}), Alto ({high_risk})")
        
        # Verificar que el directorio para visualizaciones existe
        if not Path(BAYESIAN_VIZ_DIR).exists():
            logger.info(f"Creando directorio para visualizaciones: {BAYESIAN_VIZ_DIR}")
            Path(BAYESIAN_VIZ_DIR).mkdir(parents=True, exist_ok=True)
            
        # Generar visualizaciones
        plot_bayesian_distribution(df, BAYESIAN_VIZ_DIR)
        
        # Guardar resultados actualizados
        df.to_parquet(output_file)
        logger.info(f"Resultados de evaluación bayesiana guardados en: {output_file}")
        
    except Exception as e:
        logger.error(f"Error en proceso de evaluación bayesiana: {str(e)}")
        raise

if __name__ == "__main__":
    from src.config import FUZZY_OUTPUT_FILE, BAYESIAN_OUTPUT_FILE
    
    try:
        process_bayesian_evaluation(FUZZY_OUTPUT_FILE, BAYESIAN_OUTPUT_FILE)
    except Exception as e:
        logger.critical(f"Fallo en evaluación bayesiana: {str(e)}")
        exit(1)
