"""
Módulo para configurar el sistema de logging
"""

import logging
from pathlib import Path

def configure_logger(module_name: str, log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """
    Configura un logger para el módulo especificado.

    Args:
        module_name (str): Nombre del módulo que utilizará el logger.
        log_dir (str): Directorio donde se almacenarán los archivos de log. Por defecto es 'logs'.
        log_level (int): Nivel de logging. Por defecto es logging.INFO.

    Returns:
        logging.Logger: Logger configurado.
    """
    # Crear el directorio de logs si no existe
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Configurar el formato del log
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file = Path(log_dir) / f"{module_name}.log"

    # Configurar el logger
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    # Configurar el handler para archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Configurar el handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Evitar agregar múltiples handlers si ya existen
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    # Ejemplo de uso
    logger = configure_logger("example")
    logger.info("Este es un mensaje informativo")
    logger.error("Este es un mensaje de error")
