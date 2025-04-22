"""
Módulo para generación de recomendaciones basadas en análisis de riesgo
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.logger import configure_logger
from src.config import RECOMMENDATIONS_VIZ_DIR

logger = configure_logger("recommendations")

def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera recomendaciones basadas en la probabilidad de colisión y prioridad.
    
    Args:
        df (pd.DataFrame): DataFrame con datos clasificados y evaluados previamente.
        
    Returns:
        pd.DataFrame: DataFrame con columna adicional 'recommendation'.
    """
    logger.info("Generando recomendaciones para aeronaves")

    # Verificar columnas requeridas - Ahora incluimos collision_risk
    required_columns = ['priority', 'bayesian_probability', 'collision_risk']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        logger.error(f"Columnas requeridas no encontradas: {missing}")
        raise ValueError(f"El DataFrame no contiene las columnas requeridas: {missing}")

    def recommendation_logic(row):
        """
        Lógica mejorada para generar recomendaciones basadas en prioridad, 
        riesgo bayesiano, evaluación difusa y estado de monitorización.
        
        Args:
            row: Fila del DataFrame con datos de aeronave
            
        Returns:
            str: Recomendación específica para la aeronave
        """
        # Calcular un indicador de riesgo combinado para decisiones más robustas
        # Ponderación: 60% bayesiano, 40% difuso
        combined_risk = row['bayesian_probability'] * 0.6 + row['collision_risk'] * 0.4
        
        # Prioridades altas (0-2): Acciones inmediatas basadas en evaluación combinada
        if row['priority'] == 'priority0':
            if combined_risk > 0.7 or row['collision_risk'] > 0.8:
                return "ALERTA CRÍTICA: Maniobra evasiva inmediata requerida"
            elif combined_risk > 0.4 or row['collision_risk'] > 0.6:
                return "ALERTA ALTA: Ejecutar protocolo de separación de emergencia"
            else:
                return "Vigilancia constante y preparación para maniobra"
                
        elif row['priority'] == 'priority1':
            if combined_risk > 0.7:
                return "Maniobra inmediata requerida - Riesgo elevado de conflicto"
            elif combined_risk > 0.4 or (row['bayesian_probability'] > 0.5 and row['collision_risk'] > 0.5):
                return "Maniobra preventiva sugerida - Monitoreo continuo"
            else:
                return "Mantener vigilancia activa - Preparar alternativas"
                
        elif row['priority'] == 'priority2':
            if combined_risk > 0.6 or row['collision_risk'] > 0.7:
                return "Iniciar maniobra de evasión preventiva"
            elif combined_risk > 0.3 or row['collision_risk'] > 0.5:
                return "Monitorización activa con preparación para maniobra"
            else:
                return "Seguimiento activo - Sin acción inmediata requerida"
        
        # Prioridades medias (3-5): Monitorizadas por monitoring.py, con apoyo de evaluación difusa
        elif row['priority'] in ['priority3', 'priority4', 'priority5']:
            # Considerar el estado asignado por monitoring.py y la evaluación difusa
            if row.get('status') == 'potencial riesgo' or row['collision_risk'] > 0.6:
                return "Evaluar situación - Posible conflicto detectado por monitoreo"
            elif row.get('status') == 'precaución' or row['collision_risk'] > 0.4:
                return "Verificar periódicamente - Estado de precaución detectado"
            elif row.get('status') == 'estable' and row['collision_risk'] < 0.3:
                return "Monitorización pasiva suficiente - Condición estable"
            else:
                return "Continuar monitorización pasiva estándar"
        
        # Prioridades bajas (6-9): Seguimiento de rutina, pero considerando valores atípicos
        elif row['priority'] == 'priority6':
            if row['collision_risk'] > 0.5:  # Valor atípico para esta prioridad
                return "Seguimiento con atención especial - Riesgo difuso elevado"
            return "Seguimiento rutinario - Incluir en reporte periódico"
            
        elif row['priority'] == 'priority7':
            if row['collision_risk'] > 0.6:  # Valor muy atípico para esta prioridad
                return "Verificar evaluación - Posible reclasificación necesaria"
            return "Seguimiento espaciado - Sin acción específica"
            
        elif row['priority'] == 'priority8':
            return "Registro automático - Verificación aleatoria"
            
        elif row['priority'] == 'priority9':
            return "Registro básico - Sin seguimiento activo"
            
        else:
            return "Sin acción requerida - Prioridad no determinada"

    # Aplicar lógica de recomendación
    df['recommendation'] = df.apply(recommendation_logic, axis=1)
    
    # Estadísticas de las recomendaciones generadas
    recommendation_stats = df['recommendation'].value_counts()
    logger.info(f"Estadísticas de recomendaciones:\n{recommendation_stats}")
    
    return df

def plot_recommendations_by_priority(df, RESULTS_DIR):
    """
    Genera visualizaciones mejoradas de recomendaciones por nivel de prioridad.
    
    Args:
        df (pd.DataFrame): DataFrame con recomendaciones y prioridades.
        RESULTS_DIR (str): Directorio donde guardar las visualizaciones.
    """
    # Validar columnas requeridas
    required_columns = ['recommendation', 'priority']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        logger.error(f"Columnas requeridas no encontradas en el DataFrame: {missing_cols}")
        raise ValueError(f"Columnas faltantes en DataFrame: {missing_cols}")
        
    # Verificar y crear directorio si no existe
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Crear tabla pivote de recomendaciones por prioridad
    pivot = pd.crosstab(df['recommendation'], df['priority'])
    
    # Crear categorías para agrupar recomendaciones similares
    recommendation_categories = {
        # Alertas críticas y maniobras inmediatas
        'ALERTA CRÍTICA: Maniobra evasiva inmediata requerida': 'Maniobra inmediata',
        'ALERTA ALTA: Ejecutar protocolo de separación de emergencia': 'Maniobra inmediata',
        'Maniobra inmediata requerida - Riesgo elevado de conflicto': 'Maniobra inmediata',
        
        # Maniobras preventivas
        'Maniobra preventiva sugerida - Monitoreo continuo': 'Maniobra preventiva',
        'Iniciar maniobra de evasión preventiva': 'Maniobra preventiva',
        
        # Monitorización activa
        'Vigilancia constante y preparación para maniobra': 'Monitorización activa',
        'Mantener vigilancia activa - Preparar alternativas': 'Monitorización activa',
        'Monitorización activa con preparación para maniobra': 'Monitorización activa',
        'Seguimiento activo - Sin acción inmediata requerida': 'Monitorización activa',
        'Evaluar situación - Posible conflicto detectado por monitoreo pasivo': 'Monitorización activa',
        
        # Monitorización pasiva
        'Verificar periódicamente - Estado de precaución en monitoreo pasivo': 'Monitorización pasiva',
        'Monitorización pasiva suficiente - Condición estable': 'Monitorización pasiva',
        'Continuar monitorización pasiva estándar': 'Monitorización pasiva',
        
        # Seguimiento rutinario
        'Seguimiento rutinario - Incluir en reporte periódico': 'Seguimiento rutinario',
        'Seguimiento espaciado - Sin acción específica': 'Seguimiento rutinario',
        
        # Registro básico
        'Registro automático - Verificación aleatoria': 'Registro básico',
        'Registro básico - Sin seguimiento activo': 'Registro básico',
        'Sin acción requerida - Prioridad no determinada': 'Registro básico'
    }
    
    # Añadir categoría para recomendaciones no mapeadas
    df['recommendation_category'] = df['recommendation'].map(lambda x: recommendation_categories.get(x, 'Otra'))
    
    # Crear tabla pivote utilizando las categorías
    category_pivot = pd.crosstab(df['recommendation_category'], df['priority'])
    
    # Orden de categorías para visualización
    category_order = [
        'Maniobra inmediata',
        'Maniobra preventiva',
        'Monitorización activa',
        'Monitorización pasiva',
        'Seguimiento rutinario',
        'Registro básico',
        'Otra'
    ]
    
    # Reordenar pivot por categorías
    category_pivot = category_pivot.reindex(
        [cat for cat in category_order if cat in category_pivot.index]
    )
    
    # Definir una paleta de colores más distintiva y accesible
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#e377c2']
    
    # Crear figura y ejes para categorías
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Graficar categorías
    category_pivot.plot(kind='bar', ax=ax, color=colors[:len(category_pivot)], 
                        width=0.7, edgecolor='black', linewidth=0.5)
    
    # Configurar apariencia
    ax.set_title('Recomendaciones por nivel de prioridad', fontsize=16)
    ax.set_xlabel('Categoría de recomendación', fontsize=12)
    ax.set_ylabel('Número de aeronaves', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Añadir valores sobre las barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontsize=9, padding=3)
    
    ax.legend(title='Prioridad', title_fontsize=12, fontsize=10)
    plt.xticks(rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/recommendations_by_priority_category.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generar también una versión horizontal para mejor visualización de etiquetas
    fig, ax = plt.subplots(figsize=(14, 10))
    category_pivot.plot(kind='barh', ax=ax, color=colors[:len(category_pivot)], 
                        width=0.7, edgecolor='black', linewidth=0.5)
    
    # Configurar apariencia de la versión horizontal
    ax.set_title('Recomendaciones por nivel de prioridad (vista horizontal)', fontsize=16)
    ax.set_xlabel('Número de aeronaves', fontsize=12)
    ax.set_ylabel('Categoría de recomendación', fontsize=12)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Añadir valores sobre las barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontsize=9, padding=3)
    
    ax.legend(title='Prioridad', title_fontsize=12, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/recommendations_by_priority_horizontal.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generar gráfico detallado con todas las recomendaciones específicas
    plt.figure(figsize=(16, 14))
    
    # Usar seaborn para un heatmap si está disponible
    try:
        import seaborn as sns
        sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt="d", linewidths=.5)
        plt.title('Detalle de recomendaciones específicas por prioridad', fontsize=16)
        plt.xlabel('Prioridad', fontsize=12)
        plt.ylabel('Recomendación específica', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/recommendations_detailed_heatmap.png", dpi=300, bbox_inches='tight')
    except ImportError:
        # Alternativa si seaborn no está disponible
        pivot.plot(kind='barh', figsize=(16, 14))
        plt.title('Detalle de todas las recomendaciones por prioridad', fontsize=16)
        plt.xlabel('Número de aeronaves', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/recommendations_detailed.png", dpi=300, bbox_inches='tight')
    
    plt.close()
    
    logger.info("Visualizaciones de recomendaciones por prioridad generadas")

def process_recommendations(input_file: str, output_file: str) -> None:
    """
    Procesa generación de recomendaciones y guarda resultados en un archivo.
    
    Args:
        input_file (str): Ruta al archivo con datos evaluados previamente.
        output_file (str): Ruta para guardar resultados con recomendaciones.
        
    Returns:
        None
    """
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"El archivo de entrada no existe: {input_file}")
            
        logger.info(f"Cargando datos desde: {input_file}")
        
        # Crear directorio de salida si no existe
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        logger.info(f"Cargando datos desde: {input_file}")
        
        # Cargar datos evaluados previamente
        df = pd.read_parquet(input_file)
        
        # Generar recomendaciones
        df_with_recommendations = generate_recommendations(df)
        
        # Guardar resultados actualizados
        df_with_recommendations.to_parquet(output_file)
        plot_recommendations_by_priority(df_with_recommendations, RECOMMENDATIONS_VIZ_DIR)
        logger.info(f"Resultados con recomendaciones guardados en: {output_file}")
        
    except Exception as e:
        logger.error(f"Error en generación de recomendaciones: {str(e)}")
        raise

if __name__ == "__main__":
    from src.config import BAYESIAN_OUTPUT_FILE, RECOMMENDATIONS_OUTPUT_FILE
    
    try:
        process_recommendations(BAYESIAN_OUTPUT_FILE, RECOMMENDATIONS_OUTPUT_FILE)
    except Exception as e:
        logger.critical(f"Fallo en generación de recomendaciones: {str(e)}")
        exit(1)
