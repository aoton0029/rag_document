from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
from datetime import datetime

class Neo4jClient:
    """Neo4j client for graph database operations with model support."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.url = 'bolt://neo4j:7687'
        self.driver = GraphDatabase.driver(self.url, auth=(user, password))
        
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships in the database."""
        query = "MATCH (n) DETACH DELETE n"
        try:
            with self.driver.session() as session:
                session.run(query)
        except Exception as e:
            print(f"Error clearing database: {e}")

    def get_total_node_count(self) -> int:
        """Get total count of all nodes in the graph."""
        query = "MATCH (n) RETURN count(n) AS c"
        try:
            with self.driver.session() as session:
                result = session.run(query)
                record = result.single()
                return record["c"] if record else 0
        except Exception as e:
            print(f"Error getting total node count: {e}")
            return 0
    
    def get_all_nodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all nodes in the graph with a limit."""
        query = "MATCH (n) RETURN n LIMIT $limit"
        try:
            with self.driver.session() as session:
                result = session.run(query, limit=limit)
                nodes = []
                for record in result:
                    node = record["n"]
                    node_dict = {
                        "id": node.element_id,
                        "labels": list(node.labels),
                        "properties": dict(node)
                    }
                    nodes.append(node_dict)
                return nodes
        except Exception as e:
            print(f"Error getting all nodes: {e}")
            return []
    
    def count_nodes(self):
        """Get the total count of nodes in the database."""
        query = "MATCH (n) RETURN count(n) AS node_count"
        try:
            with self.driver.session() as session:
                result = session.run(query)
                record = result.single()
                return record["node_count"] if record else 0
        except Exception as e:
            print(f"Error counting nodes: {e}")
            return 0
    
    def count_relationships(self):
        """Get the total count of relationships in the database."""
        query = "MATCH ()-[r]->() RETURN count(r) AS relationship_count"
        try:
            with self.driver.session() as session:
                result = session.run(query)
                record = result.single()
                return record["relationship_count"] if record else 0
        except Exception as e:
            print(f"Error counting relationships: {e}")
            return 0
    
    def get_node_labels(self):
        """Get all unique node labels in the database."""
        query = "CALL db.labels()"
        try:
            with self.driver.session() as session:
                result = session.run(query)
                labels = [record["label"] for record in result]
                return labels
        except Exception as e:
            print(f"Error getting node labels: {e}")
            return []
