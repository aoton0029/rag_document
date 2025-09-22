from typing import List, Dict, Any
from src.db.database_manager import db_manager

class GraphSearch:
    """Handles graph-based search operations using unified IDs."""
    
    def __init__(self):
        self.db_manager = db_manager

    def search_by_unified_id(self, unified_id: str) -> Dict[str, Any]:
        """Search for a document or chunk by its unified ID in the graph database."""
        try:
            # Example query to Neo4j to find related nodes by unified ID
            query = """
            MATCH (n)
            WHERE n.unified_id = $unified_id
            RETURN n
            """
            result = self.db_manager.neo4j.run(query, unified_id=unified_id)
            return result.data()
        except Exception as e:
            raise RuntimeError(f"Error searching by unified ID: {e}")

    def find_related_documents(self, unified_id: str) -> List[Dict[str, Any]]:
        """Find documents related to the given unified ID."""
        try:
            # Example query to find related documents
            query = """
            MATCH (doc)-[:RELATED_TO]->(related_doc)
            WHERE doc.unified_id = $unified_id
            RETURN related_doc
            """
            result = self.db_manager.neo4j.run(query, unified_id=unified_id)
            return result.data()
        except Exception as e:
            raise RuntimeError(f"Error finding related documents: {e}")

    def get_graph_structure(self) -> List[Dict[str, Any]]:
        """Retrieve the overall structure of the graph."""
        try:
            query = "MATCH (n) RETURN n"
            result = self.db_manager.neo4j.run(query)
            return result.data()
        except Exception as e:
            raise RuntimeError(f"Error retrieving graph structure: {e}")