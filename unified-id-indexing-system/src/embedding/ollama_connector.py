class OllamaConnector:
    """Ollama embedding service connector."""

    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url

    def check_connection(self) -> bool:
        """Check if the connection to the Ollama service is successful."""
        # Implement connection check logic here
        pass

    def get_available_models(self) -> list:
        """Retrieve the list of available embedding models from the Ollama service."""
        # Implement logic to fetch available models
        pass

    def get_text_embedding(self, text: str) -> list:
        """Get the embedding for a given text."""
        # Implement logic to get text embedding
        pass

    def get_query_embedding(self, query: str) -> list:
        """Get the embedding for a given query."""
        # Implement logic to get query embedding
        pass