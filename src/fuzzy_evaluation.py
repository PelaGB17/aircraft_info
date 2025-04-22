"""
Módulo para evaluación de riesgo de colisión usando lógica difusa
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
from src.utils.logger import configure_logger
from src.utils.geometry import calculate_relative_velocity
from src.config import FUZZY_VIZ_DIR
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from typing import Dict, List
from pathlib import Path
from simpful import *

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logger = configure_logger("fuzzy_eval")


class FuzzyCollisionEvaluator:
    def __init__(self):
        """Inicializa el sistema de inferencia difusa usando simpful."""
        self.FS = FuzzySystem()
        # Almacenar reglas como atributo de clase para visualización
        self.rules_text = []
        self._configure_fuzzy_system()

    def _configure_fuzzy_system(self):
        """Configura las variables lingüísticas y las reglas del sistema difuso."""
        # Definir conjuntos borrosos para las entradas
        distance_close = FuzzySet(points=[[0, 1], [10, 1], [30, 0]], term="Cerca")
        distance_medium = FuzzySet(points=[[10, 0], [30, 1], [50, 0]], term="Media")
        distance_far = FuzzySet(points=[[30, 0], [50, 1], [70, 1]], term="Lejos")
        self.FS.add_linguistic_variable("Distance", LinguisticVariable([distance_close, distance_medium, distance_far]))

        # Diferenciar entre valores positivos y negativos para speed_diff
        # Valores positivos: aeronaves acercándose - Valores negativos: aeronaves alejándose
        speed_diff_negative = FuzzySet(points=[[-500, 1], [-250, 1], [-50, 0]], term="Negativa")
        speed_diff_low = FuzzySet(points=[[-100, 0], [0, 1], [100, 0]], term="Baja")
        speed_diff_medium = FuzzySet(points=[[50, 0], [150, 1], [250, 0]], term="Media")
        speed_diff_high = FuzzySet(points=[[200, 0], [350, 1], [500, 1]], term="Alta")
        self.FS.add_linguistic_variable("SpeedDiff", LinguisticVariable([speed_diff_negative, speed_diff_low, speed_diff_medium, speed_diff_high]))

        # Añadir categoría para tiempo negativo (alejándose)
        time_negative = FuzzySet(points=[[-600, 1], [-300, 1], [-100, 0]], term="Alejamiento")
        time_short = FuzzySet(points=[[0, 1], [150, 1], [300, 0]], term="Corto")
        time_moderate = FuzzySet(points=[[150, 0], [300, 1], [450, 0]], term="Moderado")
        time_long = FuzzySet(points=[[300, 0], [450, 1], [600, 1]], term="Largo")
        self.FS.add_linguistic_variable("TimeToConflict", LinguisticVariable([time_negative, time_short, time_moderate, time_long]))

        priority_low = FuzzySet(points=[[0, 1], [50, 1], [100, 0]], term="Baja")
        priority_medium = FuzzySet(points=[[50, 0], [150, 1], [250, 0]], term="Media")
        priority_high = FuzzySet(points=[[150, 0], [250, 1], [300, 1]], term="Alta")
        self.FS.add_linguistic_variable("Priority", LinguisticVariable([priority_low, priority_medium, priority_high]))

        # Definir conjuntos borrosos para la salida
        risk_low = FuzzySet(points=[[0, 1], [0.3, 1], [0.5, 0]], term="Bajo")
        risk_medium = FuzzySet(points=[[0.3, 0], [0.5, 1], [0.7, 0]], term="Medio")
        risk_high = FuzzySet(points=[[0.6, 0], [0.8, 1], [1, 1]], term="Alto")
        self.FS.add_linguistic_variable("CollisionRisk", LinguisticVariable([risk_low, risk_medium, risk_high]))

        # Definir reglas difusas con mejoras para velocidades negativas
        self.rules_text = [
            # Reglas para aeronaves alejándose (tiempo negativo)
            "IF (TimeToConflict IS Alejamiento) THEN (CollisionRisk IS Bajo)",
            "IF (SpeedDiff IS Negativa) THEN (CollisionRisk IS Bajo)",
            
            # Reglas para aeronaves acercándose (tiempo positivo)
            "IF (Distance IS Cerca) AND (SpeedDiff IS Alta) THEN (CollisionRisk IS Alto)",
            "IF (Distance IS Cerca) AND (TimeToConflict IS Corto) THEN (CollisionRisk IS Alto)",
            "IF (Distance IS Media) AND (SpeedDiff IS Alta) THEN (CollisionRisk IS Medio)",
            "IF (Distance IS Lejos) AND (TimeToConflict IS Largo) THEN (CollisionRisk IS Bajo)",
            "IF (Priority IS Alta) AND (Distance IS Cerca) THEN (CollisionRisk IS Alto)",
            "IF (Priority IS Media) AND (Distance IS Media) THEN (CollisionRisk IS Medio)",
            "IF (Priority IS Alta) AND (TimeToConflict IS Corto) THEN (CollisionRisk IS Alto)",
            "IF (Priority IS Media) AND (TimeToConflict IS Moderado) THEN (CollisionRisk IS Medio)"
        ]
        self.FS.add_rules(self.rules_text)

    def evaluate(self, input_values: Dict[str, float]) -> Dict[str, float]:
        """
        Evalúa el riesgo de colisión usando lógica difusa.
        
        Args:
            input_values: Diccionario con valores de entrada.
            
        Returns:
            Dict: Resultados de la inferencia difusa.
        """
        try:
            # Asignar valores de entrada
            self.FS.set_variable("Distance", input_values["distance"])
            self.FS.set_variable("SpeedDiff", input_values["speed_diff"])
            self.FS.set_variable("TimeToConflict", input_values["time_to_conflict"])
            self.FS.set_variable("Priority", input_values["priority_score"])

            # Realizar inferencia
            risk = self.FS.Mamdani_inference(["CollisionRisk"])["CollisionRisk"]
            return {"collision_risk": risk}
        except Exception as e:
            logger.error(f"Error en evaluación difusa: {str(e)}")
            return {"collision_risk": 0.0}
        
    def plot_firing_strengths(self, input_values: Dict[str, float], results_dir: str) -> None:
        """
        Visualiza las fuerzas de activación (firing strengths) de cada regla para unos valores de entrada dados.
        
        Args:
            input_values: Diccionario con valores de entrada.
            results_dir: Directorio donde guardar la visualización.
        """
        try:
            # Establecer variables de entrada
            self.FS.set_variable("Distance", input_values["distance"])
            self.FS.set_variable("SpeedDiff", input_values["speed_diff"])
            self.FS.set_variable("TimeToConflict", input_values["time_to_conflict"])
            self.FS.set_variable("Priority", input_values["priority_score"])
            
            # Obtener fuerzas de activación
            firing_strengths = self.FS.get_firing_strengths()
            
            # Crear etiquetas simplificadas para las reglas
            rule_labels = []
            for i, rule in enumerate(self.rules_text):
                # Extraer la parte IF y THEN de cada regla para simplificar
                if_part = rule.split("THEN")[0].replace("IF ", "")
                then_part = rule.split("THEN")[1]
                simplified = f"R{i+1}: {if_part.strip()} → {then_part.strip()}"
                rule_labels.append(simplified)
            
            # Crear gráfico
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(firing_strengths))
            
            # Crear barras con colores basados en el valor (más intenso = más activado)
            bars = plt.barh(y_pos, firing_strengths, align='center', 
                           color=plt.cm.YlOrRd(firing_strengths))
            
            # Añadir etiquetas y título
            plt.yticks(y_pos, rule_labels, fontsize=9)
            plt.xlabel('Fuerza de Activación')
            plt.title('Fuerzas de Activación de Reglas Difusas')
            
            # Añadir valores exactos al final de cada barra
            for i, v in enumerate(firing_strengths):
                plt.text(v + 0.01, i, f'{v:.2f}', va='center')
            
            # Ajustar límites para asegurar que las etiquetas de texto sean visibles
            plt.xlim(0, max(firing_strengths) * 1.2 if max(firing_strengths) > 0 else 1)
            
            # Añadir información de valores de entrada
            info_text = (f"Distancia: {input_values['distance']:.1f} km\n" +
                        f"Dif. Velocidad: {input_values['speed_diff']:.1f} knots\n" +
                        f"Tiempo al conflicto: {input_values['time_to_conflict']:.1f} seg\n" +
                        f"Prioridad: {input_values['priority_score']:.1f}")
                        
            plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Guardar figura
            plt.tight_layout()
            plt.savefig(f"{results_dir}/fuzzy_firing_strengths.png", dpi=300)
            plt.close()
            
            logger.info(f"Gráfico de fuerzas de activación guardado en {results_dir}/fuzzy_firing_strengths.png")
            
        except Exception as e:
            logger.error(f"Error al generar gráfico de fuerzas de activación: {str(e)}")
    
    def plot_membership_functions(self, results_dir: str) -> None:
        """
        Visualiza las funciones de pertenencia de todas las variables lingüísticas.
        
        Args:
            results_dir: Directorio donde guardar la visualización.
        """
        try:
            # Asegurarse que el directorio existe
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            
            # Generar figura de funciones de membresía con simpful
            fig = self.FS.produce_figure(outputfile=f"{results_dir}/membership_functions.png")
            
            logger.info(f"Funciones de membresía guardadas en {results_dir}/membership_functions.png")
            
        except Exception as e:
            logger.error(f"Error al generar gráfico de funciones de membresía: {str(e)}")


def calculate_conflict_time(row: pd.Series) -> float:
    """
    Calcula el tiempo estimado al conflicto.
    Para velocidades negativas (aeronaves alejándose), retorna un valor finito
    pero con signo negativo para distinguirlo en el sistema difuso.
    """
    closing_speed = row.get('closing_speed', 0)
    
    # Check if closing_speed is a timestamp
    if isinstance(closing_speed, pd.Timestamp):
        logger.warning(f"Unexpected timestamp value for closing_speed: {closing_speed}")
        closing_speed = 0
    
    try:
        # Make sure distance is numeric before division
        distance = row.get('distance', 0)
        if isinstance(distance, pd.Timestamp):
            logger.warning(f"Unexpected timestamp value for distance: {distance}")
            return float('inf')
        
        # Si la velocidad es cero exactamente, no hay movimiento relativo
        if closing_speed == 0:
            return float('inf')
            
        # Para velocidades negativas (alejamiento), calcular tiempo pero con signo negativo
        # Para velocidades positivas (acercamiento), calcular tiempo normalmente
        time_value = distance / (abs(closing_speed) * 1.852 / 3600)
        
        # Limitar el valor máximo para cualquier dirección
        MAX_TIME = 600  # 10 minutos como valor máximo
        time_result = min(abs(time_value), MAX_TIME)
        
        # Preservar el signo para indicar dirección
        return time_result if closing_speed > 0 else -time_result
        
    except ZeroDivisionError:
        return float('inf')
    except TypeError as e:
        logger.error(f"Type error in conflict time calculation: {e}, values: distance={row.get('distance')}, closing_speed={closing_speed}")
        return float('inf')

def process_aircraft_batch(batch_data: List[Dict], evaluator: FuzzyCollisionEvaluator, 
                         priority_mapping: Dict[str, int]) -> List[float]:
    """
    Procesa un lote de aeronaves utilizando el evaluador difuso.
    
    Args:
        batch_data: Lista de diccionarios con datos de aeronaves.
        evaluator: Objeto evaluador difuso.
        priority_mapping: Mapeo de prioridades a valores numéricos.
        
    Returns:
        Lista de valores de riesgo calculados.
    """
    results = []
    
    for aircraft in batch_data:
        try:
            # Preparar datos de entrada para esta aeronave
            input_values = {
                'distance': float(aircraft.get('distance', 0)),
                'speed_diff': float(aircraft.get('speed_diff', 0)),
                'time_to_conflict': float(aircraft.get('time_to_conflict', 300)),
                'priority_score': float(priority_mapping.get(aircraft.get('priority', 'Unknown'), 0))
            }
            # Evaluar riesgo con el sistema fuzzy
            risk = evaluator.evaluate(input_values)['collision_risk']
            results.append(risk)
        except Exception as e:
            logger.warning(f"Error procesando aeronave: {str(e)}")
            results.append(0.0)
    
    return results

def visualize_fuzzy_decision_surface(evaluator, results_dir: str) -> None:
    """Genera visualización 3D de la superficie de decisión"""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Configurar malla de valores
        distance = np.linspace(0, 50, 25)  # Reducido a 25 puntos para menos errores
        speed_diff = np.linspace(-500, 500, 25)  # Reducido a 25 puntos
        distance_grid, speed_diff_grid = np.meshgrid(distance, speed_diff)
        
        # Calcular riesgos
        risks = np.zeros_like(distance_grid)
        for i in range(len(speed_diff)):
            for j in range(len(distance)):
                try:
                    inputs = {
                        'distance': float(distance[j]),
                        'speed_diff': float(speed_diff[i]),
                        'time_to_conflict': 300.0,
                        'priority_score': 150.0
                    }
                    risks[i,j] = evaluator.evaluate(inputs)['collision_risk']
                except Exception as inner_e:
                    logger.warning(f"Error en punto ({i},{j}): {str(inner_e)}")
                    risks[i,j] = 0.0
        
        # Visualización 3D
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(distance_grid, speed_diff_grid, risks, cmap='viridis')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel('Distancia (km)')
        ax.set_ylabel('Diferencia de Velocidad (knots)')
        ax.set_zlabel('Riesgo de Colisión')
        plt.savefig(f"{results_dir}/decision_surface_3d.png", dpi=300)
        plt.close()

        # Visualización de contorno 2D para mejor interpretabilidad
        plt.figure(figsize=(12, 10))
        contour = plt.contourf(distance_grid, speed_diff_grid, risks, 20, cmap='viridis')
        plt.colorbar(contour, label='Riesgo de Colisión')
        plt.xlabel('Distancia (km)')
        plt.ylabel('Diferencia de Velocidad (knots)')
        plt.title('Mapa de Contorno de Riesgo de Colisión')
        plt.grid(alpha=0.3)
        plt.savefig(f"{results_dir}/decision_surface_contour.png", dpi=300)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error en visualización de superficie difusa: {str(e)}")

def process_fuzzy_evaluation(input_file: str, output_file: str) -> None:
    """Procesa evaluación difusa para todas las aeronaves procesando el DataFrame en lotes directamente"""
    try:
        # Asegurar que el directorio de visualización existe
        Path(FUZZY_VIZ_DIR).mkdir(parents=True, exist_ok=True)
        
        # Cargar datos
        df = pd.read_parquet(input_file)
        
        if df.empty:
            logger.warning("No hay aeronaves para evaluar")
            return
            
        # Create working copy of all aircraft
        working_df = df.copy()
        
        # Calcular parámetros para todas las aeronaves
        working_df['speed_diff'] = working_df['speed'] - working_df['speed'].mean()
        working_df['closing_speed'] = working_df.apply(
            lambda x: calculate_relative_velocity(x['speed'], x['track']), axis=1
        )
        working_df['time_to_conflict'] = working_df.apply(
            lambda x: calculate_conflict_time(x), axis=1
        )
        
        # Initialize the fuzzy evaluator
        evaluator = FuzzyCollisionEvaluator()
        
        # Visualizar funciones de membresía
        evaluator.plot_membership_functions(FUZZY_VIZ_DIR)
        
        # Map priority levels to priority scores for fuzzy evaluation
        priority_score_mapping = {
            'priority0': 180, 'priority1': 160, 'priority2': 140,
            'priority3': 120, 'priority4': 100, 'priority5': 90,
            'priority6': 80, 'priority7': 60, 'priority8': 40,
            'priority9': 20, 'Unknown': 0
        }
        
        # Procesar en lotes directamente desde el DataFrame
        total_rows = len(working_df)
        batch_size = 5000
        collision_risks = np.zeros(total_rows)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            # Dividir el trabajo en lotes basados en índices del DataFrame
            for i in range(0, total_rows, batch_size):
                end_idx = min(i + batch_size, total_rows)
                # Extraer lote del DataFrame
                batch_df = working_df.iloc[i:end_idx]
                # Convertir solo este lote a diccionarios
                batch_data = batch_df.to_dict('records')
                
                future = executor.submit(
                    process_aircraft_batch, batch_data, evaluator, priority_score_mapping
                )
                futures.append((future, i, end_idx))
                
            # Recopilar resultados manteniendo el orden
            for future, start_idx, end_idx in futures:
                batch_results = future.result()
                # Asignar resultados al array en la posición correcta
                collision_risks[start_idx:end_idx] = batch_results
                logger.info(f"Procesado lote [{start_idx}:{end_idx}]: {len(batch_results)} aeronaves")
        
        # Asignar resultados al dataframe
        working_df['collision_risk'] = collision_risks
        df['collision_risk'] = working_df['collision_risk']
        
        # Log some statistics
        logger.info(f"Evaluación difusa completada para {len(collision_risks)} aeronaves")
        risk_stats = df['collision_risk'].describe()
        logger.info(f"Estadísticas de riesgo: min={risk_stats['min']:.4f}, "
                  f"max={risk_stats['max']:.4f}, mean={risk_stats['mean']:.4f}")
        
        # Generar visualización de ejemplos representativos
        try:
            # Seleccionar ejemplos de casos con diferentes niveles de riesgo
            high_risk = df.nlargest(1, 'collision_risk').iloc[0]
            medium_risk = df.loc[(df['collision_risk'] - 0.5).abs().idxmin()]
            low_risk = df.nsmallest(1, 'collision_risk').iloc[0]
            
            # Visualizar fuerzas de activación para caso de alto riesgo
            high_risk_inputs = {
                'distance': float(high_risk.get('distance', 0)),
                'speed_diff': float(high_risk.get('speed_diff', 0)),
                'time_to_conflict': float(high_risk.get('time_to_conflict', 300)),
                'priority_score': float(priority_score_mapping.get(high_risk.get('priority', 'Unknown'), 0))
            }
            evaluator.plot_firing_strengths(high_risk_inputs, FUZZY_VIZ_DIR)
            
            logger.info(f"Visualizada aeronave de alto riesgo (risk={high_risk['collision_risk']:.2f})")
            
            # También generar visualización para un caso de riesgo medio
            medium_risk_inputs = {
                'distance': float(medium_risk.get('distance', 0)),
                'speed_diff': float(medium_risk.get('speed_diff', 0)),
                'time_to_conflict': float(medium_risk.get('time_to_conflict', 300)),
                'priority_score': float(priority_score_mapping.get(medium_risk.get('priority', 'Unknown'), 0))
            }
            evaluator.plot_firing_strengths(medium_risk_inputs, FUZZY_VIZ_DIR)
            
        except Exception as viz_error:
            logger.warning(f"Error generando visualizaciones de ejemplo: {str(viz_error)}")
        
        # Save results
        df.to_parquet(output_file)
        
        # Generate visualization of decision surface
        visualize_fuzzy_decision_surface(evaluator, FUZZY_VIZ_DIR)
        
        logger.info(f"Resultados guardados en {output_file}")
        
    except Exception as e:
        logger.error(f"Error en proceso difuso: {str(e)}")
        raise

if __name__ == "__main__":
    from src.config import PRIORITY_OUTPUT_FILE, FUZZY_OUTPUT_FILE
    
    try:
        process_fuzzy_evaluation(PRIORITY_OUTPUT_FILE, FUZZY_OUTPUT_FILE)
    except Exception as e:
        logger.critical(f"Fallo en evaluación difusa: {str(e)}")
        exit(1)