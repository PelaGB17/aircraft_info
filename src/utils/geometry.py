"""
Módulo para cálculos geométricos relacionados con trayectorias y proximidad
"""
import math
from typing import Optional
from src.utils.logger import configure_logger

logger = configure_logger("geometry")

def calculate_relative_velocity(
    speed: float, 
    track: float, 
    reference_track: Optional[float] = None
) -> float:
    """
    Calcula la velocidad de cierre relativa basada en la dirección de la trayectoria.
    
    Args:
        speed: Velocidad de la aeronave en knots (0-2000)
        track: Dirección de la trayectoria en grados (0-360)
        reference_track: Dirección de referencia en grados (None para cálculo relativo general)
        
    Returns:
        Velocidad relativa en knots
        
    Raises:
        ValueError: Si los parámetros están fuera de rango
    """
    try:
        # Validación de parámetros
        if not 0 <= speed <= 2000:
            raise ValueError("La velocidad debe estar entre 0 y 2000 knots")
        if not 0 <= track <= 360:
            raise ValueError("La dirección debe estar entre 0 y 360 grados")
        
        # Si no se especifica referencia, usar diferencia angular mínima
        if reference_track is None:
            return speed  # Máxima velocidad relativa
            
        # Normalizar ángulos
        track = math.radians(track % 360)
        reference_track = math.radians(reference_track % 360)
        
        # Calcular diferencia angular
        angle_diff = min(
            abs(track - reference_track),
            2 * math.pi - abs(track - reference_track)
        )
        
        # Calcular velocidad relativa usando proyección
        relative_velocity = speed * math.cos(angle_diff)
       
        return relative_velocity
    
    except ValueError as ve:
        raise
    except Exception as e:
        return 0.0

def calculate_distance(
    lat1: float, 
    lon1: float, 
    lat2: float, 
    lon2: float
) -> float:
    """
    Calcula la distancia entre dos puntos geográficos usando la fórmula Haversine.
    
    Args:
        lat1: Latitud del primer punto en grados (-90 a 90)
        lon1: Longitud del primer punto en grados (-180 a 180)
        lat2: Latitud del segundo punto en grados (-90 a 90)
        lon2: Longitud del segundo punto en grados (-180 a 180)
        
    Returns:
        Distancia en kilómetros
        
    Raises:
        ValueError: Si las coordenadas son inválidas
    """
    try:
        # Validación de coordenadas
        if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90):
            raise ValueError("Latitudes deben estar entre -90 y 90 grados")
        if not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
            raise ValueError("Longitudes deben estar entre -180 y 180 grados")
        
        # Radio de la Tierra en kilómetros
        R = 6371.0
        
        # Convertir grados a radianes
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        
        # Diferencias
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Fórmula Haversine
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        distance = R * c
        
        logger.debug(f"Distancia calculada: {distance:.2f} km")
        
        return distance
    
    except ValueError as ve:
        raise
    except Exception as e:
        return 0.0

if __name__ == "__main__":
    # Pruebas de validación
    try:
        print(calculate_relative_velocity(450, 90, 0))  # 0 knots
        print(calculate_distance(40.7128, -74.0060, 34.0522, -118.2437))  # ~3935 km
    except Exception as e:
        logger.critical(f"Error en pruebas: {str(e)}")
