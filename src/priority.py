"""
Módulo para clasificación de prioridad de aeronaves usando sistema de puntuación
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils.logger import configure_logger
from src.config import PRIORITY_VIZ_DIR
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logger = configure_logger("priority")

# Mapa de colores para visualización de prioridades
PRIORITY_COLORS = {
    "priority0": "darkred",
    "priority1": "red",
    "priority2": "orange",
    "priority3": "blue",
    "priority4": "green",
    "priority5": "purple",
    "priority6": "cyan",
    "priority7": "magenta",
    "priority8": "yellow",
    "priority9": "lightgreen",
    "Unknown": "gray"
}

# Definir características numéricas y categóricas
NUMERIC_FEATURES = [
    "speed", "distance", "rssi", "wind_speed", "wind_direction", 
    "alt_baro", "altgeom", "ias", "tas", "mach", "track", 
    "track_rate", "roll", "mag_heading", "true_heading", "baro_rate", "geom_rate", 
    "nic", "rc", "nic_baro", "nac_p", "nac_v", "sil", 
    "gva", "sda", "nav_qnh", "nav_altitude_mcp", "nav_altitude_fms", "nav_heading"
]

CATEGORICAL_FEATURES = [
    "category", "air_ground", "sil_type", "addr_type", 
    "emergency", "autopilot", "vnav", "althold", "approach", 
    "lnav", "tcas"
]

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia entre dos puntos geográficos usando la fórmula Haversine"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371.0  # Radio de la Tierra en kilómetros
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance

def add_distance_column(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula distancias entre aeronaves y punto de control"""
    logger.info("Calculando distancias entre aeronaves")
    
    # Coordenadas del receptor en Asturias
    control_lat, control_lon = 43.358611, -5.506111
    
    df['distance'] = df.apply(lambda row: calculate_distance(row['lat'], row['lon'], control_lat, control_lon), axis=1)
    
    logger.info("Distancias calculadas exitosamente")
    return df

def calculate_priority_score(row):
    """
    Sistema de puntuación que calcula los puntos totales para una aeronave.
    
    Args:
        row: Fila del DataFrame con datos de aeronave
        
    Returns:
        int: Puntuación total
    """
    score = 0
    
    # Distancia (hasta 100 puntos)
    if pd.notna(row.get('distance')):
        if row['distance'] <= 2: score += 100
        elif row['distance'] <= 5: score += 90
        elif row['distance'] <= 10: score += 80
        elif row['distance'] <= 20: score += 70
        elif row['distance'] <= 30: score += 60
        elif row['distance'] <= 40: score += 10
    
    # Velocidad (hasta 80 puntos)
    if pd.notna(row.get('speed')):
        if row['speed'] >= 300: score += 80
        elif row['speed'] >= 250: score += 70
        elif row['speed'] >= 200: score += 60
        elif row['speed'] >= 150: score += 50
        elif row['speed'] >= 120: score += 40
        elif row['speed'] >= 100: score += 10
    
    # RSSI (hasta 15 puntos)
    if pd.notna(row.get('rssi')):
        if row['rssi'] >= -21: score += 15
        elif row['rssi'] >= -26: score += 10
        elif row['rssi'] >= -32: score += 8
        elif row['rssi'] >= -37: score += 5
        elif row['rssi'] >= -39: score += 3
        elif row['rssi'] >= -42: score += 1
    
    # Viento (hasta 6 puntos)
    if pd.notna(row.get('wind_speed')):
        if row['wind_speed'] >= 20: score += 6
        elif row['wind_speed'] >= 15: score += 5
        elif row['wind_speed'] >= 10: score += 4
        elif row['wind_speed'] >= 5: score += 3
        elif row['wind_speed'] >= 3: score += 2
        elif row['wind_speed'] >= 2: score += 1
    
    # Altitud (hasta 15 puntos)
    if pd.notna(row.get('alt_baro')):
        if row['alt_baro'] >= 37000: score += 15
        elif row['alt_baro'] >= 35000: score += 12
        elif row['alt_baro'] >= 33000: score += 9
        elif row['alt_baro'] >= 30000: score += 6
        elif row['alt_baro'] >= 25000: score += 3
    
    # Cambio de altitud (hasta 10 puntos)
    if pd.notna(row.get('baro_rate')):
        baro_rate_abs = abs(row['baro_rate'])
        if baro_rate_abs >= 500: score += 10
        elif baro_rate_abs >= 400: score += 8
        elif baro_rate_abs >= 300: score += 6
        elif baro_rate_abs >= 200: score += 4
        elif baro_rate_abs >= 100: score += 2
    
    # Rate de giro (hasta 5 puntos)
    if pd.notna(row.get('track_rate')):
        track_rate_abs = abs(row['track_rate'])
        if track_rate_abs >= 0.05: score += 5
        elif track_rate_abs >= 0.04: score += 4
        elif track_rate_abs >= 0.03: score += 3
        elif track_rate_abs >= 0.02: score += 2
        elif track_rate_abs >= 0.01: score += 1
    
    return score  # Devuelve el valor numérico


def score_to_priority(score):
    """
    Convierte una puntuación numérica a un nivel de prioridad.
    
    Args:
        score: Puntuación numérica
        
    Returns:
        str: Nivel de prioridad
    """
    if score >= 170: return "priority0"
    elif score >= 150: return "priority1"
    elif score >= 130: return "priority2"
    elif score >= 110: return "priority3"
    elif score >= 100: return "priority4"
    elif score >= 90: return "priority5"
    elif score >= 70: return "priority6"
    elif score >= 50: return "priority7"
    elif score >= 30: return "priority8"
    elif score >= 10: return "priority9"
    else: return "Unknown"

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa los datos para la clasificación"""
    logger.info("Preprocesando datos para clasificación")
    
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calcular distancia si no existe
    if 'distance' not in df.columns and 'lat' in df.columns and 'lon' in df.columns:
        df = add_distance_column(df)
    
    # Aplicar sistema de puntuación
    df['priority_score'] = df.apply(calculate_priority_score, axis=1)
    
    # Convertir puntuación a nivel de prioridad
    df['priority'] = df['priority_score'].apply(score_to_priority)
    
    logger.info("Preprocesamiento completado")
    return df

def evaluate_model(clf, X_test, y_test, feature_names, output_dir, model_type="modelo"):
    """Evalúa cualquier modelo de clasificación y genera visualizaciones"""
    logger.info(f"Evaluando {model_type}")
    
    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Predecir usando el modelo entrenado
    y_pred = clf.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title(f'Matriz de Confusión - {model_type}')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type.lower().replace(' ', '_')}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # Calcular importancia de características si está disponible
    try:
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'named_steps') and hasattr(clf.named_steps.get('classifier', None), 'feature_importances_'):
            importances = clf.named_steps['classifier'].feature_importances_
        else:
            perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=42)
            importances = perm_importance.importances_mean
        
        # Visualizar importancia de características 
        plt.figure(figsize=(12, 8))
        indices = np.argsort(importances)[::-1]
        plt.barh(range(min(20, len(indices))), importances[indices][:20], align='center')
        plt.yticks(range(min(20, len(indices))), [feature_names[i] for i in indices][:20])
        plt.title(f'Importancia de Características - {model_type}')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_type.lower().replace(' ', '_')}_feature_importance.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"No se pudo calcular la importancia de características: {str(e)}")
    
    # Calcular distribución de prioridades
    priority_counts = pd.Series(y_pred).value_counts()
    plt.figure(figsize=(10, 8))
    plt.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', 
            colors=[PRIORITY_COLORS.get(p, 'gray') for p in priority_counts.index])
    plt.title(f'Distribución de Prioridades - {model_type}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type.lower().replace(' ', '_')}_priority_distribution.png", dpi=300)
    plt.close()
    
    # Guardar reporte detallado
    with open(f"{output_dir}/{model_type.lower().replace(' ', '_')}_evaluation.txt", "w") as f:
        f.write(f"EVALUACIÓN DEL {model_type.upper()}\n")
        f.write(f"{'='*len(f'EVALUACIÓN DEL {model_type.upper()}')}\n\n")
        f.write(f"Precisión global: {accuracy:.4f}\n\n")
        f.write(f"Reporte de clasificación:\n")
        for cls in report:
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                f.write(f"\nClase: {cls}\n")
                f.write(f"  Precisión:    {report[cls]['precision']:.4f}\n")
                f.write(f"  Recall:       {report[cls]['recall']:.4f}\n")
                f.write(f"  F1-score:     {report[cls]['f1-score']:.4f}\n")
                f.write(f"  Muestras:     {report[cls]['support']}\n")
        
        f.write("\n\nDistribución de Prioridades:\n")
        f.write("===========================\n\n")
        for priority, count in priority_counts.items():
            f.write(f"{priority}: {count} ({count/len(y_pred)*100:.1f}%)\n")
    
    logger.info(f"Evaluación de {model_type} generada en {output_dir}")
    
    return {
        'accuracy': accuracy,
        'report': report,
        'priority_distribution': priority_counts.to_dict()
    }

def plot_priority_scatter(df, output_dir):
    """Genera gráficos de dispersión para visualizar la relación entre variables y niveles de prioridad"""
    logger.info("Generando visualizaciones de prioridades")
    
    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Hacer una copia del dataframe para evitar advertencias de vista
    plot_df = df.copy()
    
    # Asegurar que las columnas numéricas sean realmente numéricas
    for col in ['distance', 'speed', 'rssi', 'priority_score']:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    
    # Crear máscara para cada nivel de prioridad
    priority_masks = {}
    for i in range(10):
        priority_masks[f"priority{i}"] = plot_df['priority'] == f"priority{i}"
    
    unknown = plot_df['priority'] == 'Unknown'
    
    # Gráfico de distancia vs velocidad
    try:
        plt.figure(figsize=(12, 10))
        
        # Graficar cada grupo con color distintivo, asegurando valores numéricos
        if np.any(unknown):
            valid_unknown = unknown & plot_df['distance'].notna() & plot_df['speed'].notna()
            if np.any(valid_unknown):
                plt.scatter(
                    plot_df[valid_unknown]['distance'], 
                    plot_df[valid_unknown]['speed'], 
                    color=PRIORITY_COLORS['Unknown'], 
                    alpha=0.5, 
                    label='Unknown'
                )
        
        for i in range(9, -1, -1):
            priority = f"priority{i}"
            if np.any(priority_masks[priority]):
                valid_points = priority_masks[priority] & plot_df['distance'].notna() & plot_df['speed'].notna()
                if np.any(valid_points):
                    plt.scatter(
                        plot_df[valid_points]['distance'], 
                        plot_df[valid_points]['speed'], 
                        color=PRIORITY_COLORS[priority], 
                        alpha=0.7, 
                        label=f'Priority {i}'
                    )
        
        plt.xlabel("Distancia (km)")
        plt.ylabel("Velocidad (knots)")
        plt.title("Clasificación por prioridad: Relación Distancia-Velocidad")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/priority_distance_speed.png")
        plt.close()
        logger.info("Gráfico de distancia vs velocidad guardado correctamente")
    except Exception as e:
        logger.error(f"Error al crear gráfico de distancia vs velocidad: {str(e)}")
    
    # Gráfico de RSSI vs distancia si está disponible
    if 'rssi' in plot_df.columns:
        try:
            plt.figure(figsize=(12, 10))
            
            if np.any(unknown):
                valid_unknown = unknown & plot_df['distance'].notna() & plot_df['rssi'].notna()
                if np.any(valid_unknown):
                    plt.scatter(
                        plot_df[valid_unknown]['distance'], 
                        plot_df[valid_unknown]['rssi'], 
                        color=PRIORITY_COLORS['Unknown'], 
                        alpha=0.5, 
                        label='Unknown'
                    )
            
            for i in range(5, -1, -1):
                priority = f"priority{i}"
                if np.any(priority_masks[priority]):
                    valid_points = priority_masks[priority] & plot_df['distance'].notna() & plot_df['rssi'].notna()
                    if np.any(valid_points):
                        plt.scatter(
                            plot_df[valid_points]['distance'], 
                            plot_df[valid_points]['rssi'], 
                            color=PRIORITY_COLORS[priority], 
                            alpha=0.7, 
                            label=f'Priority {i}'
                        )
            
            plt.xlabel("Distancia (km)")
            plt.ylabel("RSSI (dBm)")
            plt.title("Relación entre Distancia y RSSI por Prioridad")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/priority_distance_rssi.png")
            plt.close()
            logger.info("Gráfico de RSSI vs distancia guardado correctamente")
        except Exception as e:
            logger.error(f"Error al crear gráfico de RSSI vs distancia: {str(e)}")
    
    # Gráfico de puntuación vs prioridad
    if 'priority_score' in plot_df.columns:
        try:
            plt.figure(figsize=(12, 8))
            
            # Crear boxplot de puntuación por prioridad
            priority_order = [f"priority{i}" for i in range(10)] + ["Unknown"]
            data_to_plot = []
            labels = []
            
            for priority in priority_order:
                if priority in plot_df['priority'].values:
                    # Seleccionar solo valores numéricos para el boxplot
                    priority_data = plot_df[plot_df['priority'] == priority]['priority_score'].dropna()
                    if len(priority_data) > 0:
                        data_to_plot.append(priority_data)
                        labels.append(priority)
            
            if data_to_plot:  # Solo graficar si hay datos
                plt.boxplot(data_to_plot, labels=labels)
                plt.title("Distribución de Puntuación por Nivel de Prioridad")
                plt.ylabel("Puntuación")
                plt.xlabel("Nivel de Prioridad")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/priority_score_distribution.png", dpi=300)
                plt.close()
                logger.info("Gráfico de distribución de puntuación guardado correctamente")
        except Exception as e:
            logger.error(f"Error al crear gráfico de distribución de puntuación: {str(e)}")
    
    # Crear gráfico de torta con distribución de prioridades
    try:
        priority_counts = plot_df['priority'].value_counts()
        plt.figure(figsize=(10, 8))
        plt.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', 
               colors=[PRIORITY_COLORS.get(p, 'gray') for p in priority_counts.index])
        plt.title('Distribución de Prioridades de Aeronaves')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/priority_distribution.png", dpi=300)
        plt.close()
        logger.info("Gráfico de distribución de prioridades guardado correctamente")
    except Exception as e:
        logger.error(f"Error al crear gráfico de distribución de prioridades: {str(e)}")
    
    logger.info(f"Visualizaciones guardadas en {output_dir}")


def train_decision_tree(df: pd.DataFrame) -> tuple[pd.DataFrame, DecisionTreeClassifier]:
    """
    Entrena un modelo de árbol de decisión como complemento al sistema de puntuación.
    Este modelo ayuda a visualizar las reglas de decisión aprendidas.
    """
    logger.info("Entrenando árbol de decisión complementario")
    
    # Verificar si hay datos suficientes
    if len(df) < 10:
        logger.warning(f"Datos insuficientes para entrenamiento: solo {len(df)} registros")
        return df, None
    
    # Seleccionar características disponibles
    numeric_features = [col for col in NUMERIC_FEATURES if col in df.columns]
    
    # Preparar datos para entrenamiento
    X = df[numeric_features].copy()
    y = df['priority'].copy()
    
    # Manejar valores faltantes
    X = X.fillna(X.median())
    
    # Filtrar etiquetas desconocidas
    mask = y != 'Unknown'
    if mask.sum() < 10:
        logger.warning("No hay suficientes muestras etiquetadas para entrenar el modelo")
        return df, None
    
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    try:
        # División simple sin estratificación para evitar errores
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.25, random_state=42
        )
        
        # Entrenar árbol de decisión
        clf = DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_split=10, min_samples_leaf=5, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        logger.info("Árbol de decisión entrenado exitosamente")
        
        # Evaluar modelo
        evaluate_model(clf, X_test, y_test, numeric_features, PRIORITY_VIZ_DIR, "Árbol de Decisión")
        
        # Visualizar árbol
        plt.figure(figsize=(20, 15))
        plot_tree(clf, feature_names=numeric_features, class_names=clf.classes_,
                filled=True, rounded=True, max_depth=4)
        plt.title("Árbol de Decisión para Clasificación de Prioridad")
        plt.tight_layout()
        plt.savefig(f"{PRIORITY_VIZ_DIR}/decision_tree.png", dpi=300)
        plt.close()
        
        # Exportar árbol como texto
        tree_text = export_text(clf, feature_names=numeric_features, show_weights=True)
        with open(f"{PRIORITY_VIZ_DIR}/decision_tree.txt", "w") as f:
            f.write(tree_text)
        
        return df, clf
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento del árbol: {str(e)}")
        return df, None

def train_gradient_boosting(df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """
    Entrena un modelo de Gradient Boosting como complemento al sistema de puntuación.
    Este modelo puede capturar relaciones más complejas entre variables.
    """
    logger.info("Entrenando modelo Gradient Boosting complementario")
    
    # Verificar si hay datos suficientes
    if len(df) < 10:
        logger.warning(f"Datos insuficientes para entrenamiento: solo {len(df)} registros")
        return df, None
    
    # Seleccionar características disponibles en el dataset
    available_numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    available_categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    
    logger.info(f"Características numéricas disponibles: {available_numeric}")
    logger.info(f"Características categóricas disponibles: {available_categorical}")
    
    # Filtrar filas con prioridad 'Unknown'
    mask = df['priority'] != 'Unknown'
    
    if mask.sum() < 10:
        logger.warning(f"Datos clasificados insuficientes: solo {mask.sum()} registros después de filtrar")
        return df, None
    
    # Preparar datos
    X = df[available_numeric + available_categorical].copy()
    y = df['priority'].copy()
    
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    try:
        # División en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.25, random_state=42
        )
        
        # Preprocesamiento para columnas numéricas y categóricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Preprocesador columnar
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, available_numeric),
                ('cat', categorical_transformer, available_categorical)
            ],
            remainder='drop'
        )
        
        # Pipeline completo
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(n_estimators=300, 
                                                     learning_rate=0.05, 
                                                     max_depth=4,
                                                     subsample=0.8,
                                                     min_samples_leaf=10, 
                                                     random_state=42))
        ])
        
        # Entrenar modelo simplificado (sin Grid Search para evitar errores)
        pipeline.fit(X_train, y_train)
        
        logger.info("Gradient Boosting entrenado exitosamente")
        
        # Evaluar modelo
        feature_names = available_numeric.copy()
        if available_categorical:
            try:
                # Obtener nombres de características reales después de transformación
                preprocessor = pipeline.named_steps['preprocessor']
                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, cols in preprocessor.transformers_:
                        if name == 'cat' and hasattr(transformer, 'named_steps'):
                            if 'onehot' in transformer.named_steps:
                                encoder = transformer.named_steps['onehot']
                                if hasattr(encoder, 'get_feature_names_out'):
                                    cat_features = encoder.get_feature_names_out(input_features=available_categorical)
                                    feature_names.extend(cat_features)
            except Exception as e:
                logger.warning(f"No se pudieron obtener nombres de características: {str(e)}")
        
        evaluate_model(pipeline, X_test, y_test, feature_names, PRIORITY_VIZ_DIR, "Gradient Boosting")
        
        return df, pipeline
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento de Gradient Boosting: {str(e)}")
        return df, None

def process_priority(input_file: str, output_file: str) -> None:
    """
    Procesa clasificación de prioridad utilizando principalmente el sistema de puntuación,
    con modelos complementarios de árbol de decisión y gradient boosting.
    
    Args:
        input_file (str): Ruta al archivo de entrada con datos preprocesados.
        output_file (str): Ruta al archivo de salida para guardar resultados.
    """
    try:
        logger.info(f"Cargando datos desde {input_file}")
        
        # Crear directorio si no existe
        Path(PRIORITY_VIZ_DIR).mkdir(parents=True, exist_ok=True)
        
        # Cargar datos procesados
        df = pd.read_parquet(input_file)
        
        # Preprocesar datos y aplicar sistema de puntuación
        df = preprocess_data(df)
             
        logger.info("Prioridades asignadas mediante sistema de puntuación")
        
        # Entrenar modelos complementarios para análisis
        # Estos no afectarán la asignación de prioridades del sistema de puntuación
        _, tree_model = train_decision_tree(df)
        _, gb_model = train_gradient_boosting(df)
        
        # Generar visualizaciones del sistema de puntuación
        plot_priority_scatter(df, PRIORITY_VIZ_DIR)
        logger.info("Visualizaciones generadas")
        
        # Filtrar aeronaves sin clasificar (Unknown)
        df_final = df[df['priority'] != 'Unknown']
        
        removed_count = len(df) - len(df_final)
        removed_percentage = (removed_count / len(df)) * 100 if len(df) > 0 else 0
        
        logger.info(f"Datos filtrados: {removed_count} registros eliminados ({removed_percentage:.2f}% Unknown)")
        
        # Guardar resultados clasificados en archivo Parquet
        df_final.to_parquet(output_file)
        
        logger.info(f"Resultados guardados en {output_file}")
        
    except Exception as e:
        logger.error(f"Error en clasificación de prioridad: {str(e)}")
        raise

if __name__ == "__main__":
    from src.config import PROCESSED_DATA_FILE, PRIORITY_OUTPUT_FILE
    
    try:
        process_priority(PROCESSED_DATA_FILE, PRIORITY_OUTPUT_FILE)
    except Exception as e:
        logger.critical(f"Fallo en la clasificación de prioridad: {str(e)}")
        exit(1)
