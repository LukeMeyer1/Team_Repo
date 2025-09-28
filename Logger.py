import os
import logging

# Read environment variables
log_file = os.getenv("LOG_FILE", "app.log")
log_level_env = os.getenv("LOG_LEVEL", "0")

# Map verbosity
verbosity_map = {
    "0": logging.CRITICAL + 1,  # Silent
    "1": logging.INFO,
    "2": logging.DEBUG,
}
log_level = verbosity_map.get(log_level_env, logging.CRITICAL + 1)

# Configure global logging
logging.basicConfig(
    filename=log_file,
    filemode="a",
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

def get_logger(name: str):
    """
    Return a logger tagged with the given module name.\n
    Best Practice: Use Function to get a logger per file.\n\t\t\t
                   "name" as file name.
    """
    return logging.getLogger(name)
    
    # Example usage:
    # from Logger import get_logger
    # logger = get_logger(__file_name__)
    # ...  