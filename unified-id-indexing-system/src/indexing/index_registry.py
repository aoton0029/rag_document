from typing import Dict, Any

class IndexRegistry:
    """Class to manage the status of indexes across different databases."""
    
    def __init__(self):
        self.index_status: Dict[str, Dict[str, Any]] = {}

    def register_index(self, unified_id: str, db_name: str, status: str) -> None:
        """Register a new index status."""
        if unified_id not in self.index_status:
            self.index_status[unified_id] = {}
        self.index_status[unified_id][db_name] = {
            "status": status,
            "created_at": self.get_current_time()
        }

    def update_index_status(self, unified_id: str, db_name: str, status: str) -> None:
        """Update the status of an existing index."""
        if unified_id in self.index_status and db_name in self.index_status[unified_id]:
            self.index_status[unified_id][db_name]["status"] = status
            self.index_status[unified_id][db_name]["updated_at"] = self.get_current_time()

    def get_index_status(self, unified_id: str) -> Dict[str, Any]:
        """Retrieve the status of indexes for a given unified ID."""
        return self.index_status.get(unified_id, {})

    def get_current_time(self) -> str:
        """Get the current time as a string."""
        from datetime import datetime
        return datetime.now().isoformat()