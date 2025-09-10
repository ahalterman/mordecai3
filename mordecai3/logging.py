import logging

def setup_logging(level: int=logging.INFO, format_string: str | None=None, 
                  quiet_third_party: bool=True) -> None:
    """Setup logging for mordecai3."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if quiet_third_party:
        # Reduce noise from verbose 3rd party packages
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("elasticsearch").setLevel(logging.ERROR)
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler()]
    )