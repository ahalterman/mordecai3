

class SpacyModelError(Exception):
    """Raised when the required spaCy model is not installed."""

    def __init__(self, message=None):
        if message is None:
            message = (
                "The required spaCy model 'en_core_web_trf' is not installed.\n"
                "Install it with:\n\n"
                "    python -m spacy download en_core_web_trf\n"
            )
        super().__init__(message)


class ElasticsearchConnectionError(Exception):
    """Raised when Elasticsearch is not reachable."""

    def __init__(self, message=None):
        if message is None:
            message = (
                "Could not connect to Elasticsearch.\n"
                "Make sure Elasticsearch is running. You can start it with Docker:\n\n"
                "    docker run -d -p 9200:9200 -e \"discovery.type=single-node\" elasticsearch:7.10.1\n"
                "\nSee the mordecai3 README for full setup instructions."
            )
        super().__init__(message)


class GeonamesIndexError(Exception):
    """Raised when Elasticsearch is running but the geonames index is missing."""

    def __init__(self, message=None):
        if message is None:
            message = (
                "Connected to Elasticsearch, but the 'geonames' index was not found.\n"
                "You need to set up the geonames index using the es-geonames tool.\n"
                "See the mordecai3 README for setup instructions:\n\n"
                "    https://github.com/ahalterman/mordecai3#setup\n"
            )
        super().__init__(message)
