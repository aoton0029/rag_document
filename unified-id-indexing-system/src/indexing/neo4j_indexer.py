from db.neo4j_client import Neo4jClient
from core.unified_id import UnifiedID
from core.correlation_id import CorrelationID
from core.global_sequence import GlobalSequence
import logging

logger = logging.getLogger(__name__)

class Neo4jIndexer:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j_client = neo4j_client

    def create_index(self, unified_id: str, data: dict):
        try:
            query = f"""
            CREATE (n:Document {{unified_id: '{unified_id}', data: $data}})
            """
            self.neo4j_client.run_query(query, parameters={"data": data})
            logger.info(f"Index created for unified ID: {unified_id}")
        except Exception as e:
            logger.error(f"Failed to create index for unified ID {unified_id}: {e}")

    def update_index(self, unified_id: str, data: dict):
        try:
            query = f"""
            MATCH (n:Document {{unified_id: '{unified_id}'}})
            SET n.data = $data
            """
            self.neo4j_client.run_query(query, parameters={"data": data})
            logger.info(f"Index updated for unified ID: {unified_id}")
        except Exception as e:
            logger.error(f"Failed to update index for unified ID {unified_id}: {e}")

    def delete_index(self, unified_id: str):
        try:
            query = f"""
            MATCH (n:Document {{unified_id: '{unified_id}'}})
            DELETE n
            """
            self.neo4j_client.run_query(query)
            logger.info(f"Index deleted for unified ID: {unified_id}")
        except Exception as e:
            logger.error(f"Failed to delete index for unified ID {unified_id}: {e}")