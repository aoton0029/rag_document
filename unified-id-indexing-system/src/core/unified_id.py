class UnifiedID:
    import uuid
    from datetime import datetime

    def __init__(self):
        self.unified_id = self.generate_unified_id()

    def generate_unified_id(self) -> str:
        """Generates a new unified ID in UUID4 format with a timestamp."""
        return f"{uuid.uuid4()}_{int(datetime.now().timestamp())}"

    def get_unified_id(self) -> str:
        """Returns the generated unified ID."""
        return self.unified_id

    def __str__(self) -> str:
        """Returns a string representation of the unified ID."""
        return self.unified_id

    @staticmethod
    def validate_unified_id(unified_id: str) -> bool:
        """Validates the format of a given unified ID."""
        try:
            uuid_part, timestamp_part = unified_id.split('_')
            uuid.UUID(uuid_part)  # Validate UUID part
            int(timestamp_part)   # Validate timestamp part
            return True
        except (ValueError, IndexError):
            return False