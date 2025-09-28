import os
import logging

# Read environment variables
log_file = os.getenv("LOG_FILE")  # no default; don't silently create a file
log_level_env = os.getenv("LOG_LEVEL", "0")

_level_map = {"0": logging.CRITICAL + 1, "1": logging.INFO, "2": logging.DEBUG}
log_level = _level_map.get(log_level_env, logging.CRITICAL + 1)


kwargs = dict(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

# only attach a file handler if LOG_FILE was provided
if log_file:
    kwargs.update(filename=log_file, filemode="r+")   # don't create new files

# logging.basicConfig(**kwargs)

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
