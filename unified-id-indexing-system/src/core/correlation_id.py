class CorrelationID:
    """Generates and manages correlation IDs for request tracing."""
    
    def __init__(self):
        self.current_id = None

    def generate(self):
        """Generates a new correlation ID."""
        import uuid
        self.current_id = str(uuid.uuid4())
        return self.current_id

    def get_current_id(self):
        """Returns the current correlation ID."""
        return self.current_id

    def reset(self):
        """Resets the correlation ID."""
        self.current_id = None