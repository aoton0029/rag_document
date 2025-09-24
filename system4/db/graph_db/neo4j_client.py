"""
Neo4j client for RAGShelf graph database operations.
"""
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import models
from .models import Document, Concept, User, Query
from .models import Contains, RelatedTo, Searched, Retrieved


class Neo4jClient:
    """Neo4j client for graph database operations with model support."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
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

    # Node operations
    def create_document_node(self, document: Document) -> bool:
        """Create a document node using the Document model."""
        query = """
        CREATE (d:Document {
            document_id: $document_id,
            title: $title,
            content_type: $content_type,
            language: $language,
            author: $author,
            source: $source,
            created_at: $created_at,
            embedding_count: $embedding_count
        })
        """
        try:
            with self.driver.session() as session:
                session.run(query, 
                           document_id=document.document_id,
                           title=document.title,
                           content_type=document.content_type,
                           language=document.language,
                           author=document.author,
                           source=document.source,
                           created_at=document.created_at.isoformat(),
                           embedding_count=document.embedding_count)
            return True
        except Exception as e:
            print(f"Error creating document node: {e}")
            return False
    
    def create_concept_node(self, concept: Concept) -> bool:
        """Create a concept node using the Concept model."""
        query = """
        CREATE (c:Concept {
            concept_id: $concept_id,
            name: $name,
            type: $type,
            definition: $definition,
            confidence_score: $confidence_score,
            frequency: $frequency,
            created_at: $created_at
        })
        """
        try:
            with self.driver.session() as session:
                session.run(query,
                           concept_id=concept.concept_id,
                           name=concept.name,
                           type=concept.type,
                           definition=concept.definition,
                           confidence_score=concept.confidence_score,
                           frequency=concept.frequency,
                           created_at=concept.created_at.isoformat())
            return True
        except Exception as e:
            print(f"Error creating concept node: {e}")
            return False
    
    def create_user_node(self, user: User) -> bool:
        """Create a user node using the User model."""
        query = """
        CREATE (u:User {
            user_id: $user_id,
            username: $username,
            email: $email,
            preferences: $preferences,
            created_at: $created_at,
            last_active: $last_active
        })
        """
        try:
            with self.driver.session() as session:
                session.run(query,
                           user_id=user.user_id,
                           username=user.username,
                           email=str(user.email) if user.email else None,
                           preferences=user.preferences.dict() if user.preferences else None,
                           created_at=user.created_at.isoformat(),
                           last_active=user.last_active.isoformat() if user.last_active else None)
            return True
        except Exception as e:
            print(f"Error creating user node: {e}")
            return False
    
    def create_query_node(self, query_model: Query) -> bool:
        """Create a query node using the Query model."""
        query = """
        CREATE (q:Query {
            query_id: $query_id,
            text: $text,
            normalized_text: $normalized_text,
            intent: $intent,
            timestamp: $timestamp,
            response_time: $response_time
        })
        """
        try:
            with self.driver.session() as session:
                session.run(query,
                           query_id=query_model.query_id,
                           text=query_model.text,
                           normalized_text=query_model.normalized_text,
                           intent=query_model.intent,
                           timestamp=query_model.timestamp.isoformat(),
                           response_time=query_model.response_time)
            return True
        except Exception as e:
            print(f"Error creating query node: {e}")
            return False
    
    # Relationship operations
    def create_contains_relationship(self, document_id: str, concept_id: str, 
                                   relationship: Contains) -> bool:
        """Create a CONTAINS relationship between document and concept."""
        query = """
        MATCH (d:Document {document_id: $document_id}), (c:Concept {concept_id: $concept_id})
        CREATE (d)-[r:CONTAINS {
            frequency: $frequency,
            positions: $positions,
            relevance_score: $relevance_score,
            extraction_method: $extraction_method
        }]->(c)
        """
        try:
            with self.driver.session() as session:
                session.run(query,
                           document_id=document_id,
                           concept_id=concept_id,
                           frequency=relationship.frequency,
                           positions=relationship.positions,
                           relevance_score=relationship.relevance_score,
                           extraction_method=relationship.extraction_method)
            return True
        except Exception as e:
            print(f"Error creating CONTAINS relationship: {e}")
            return False
    
    def create_related_to_relationship(self, from_concept_id: str, to_concept_id: str,
                                     relationship: RelatedTo) -> bool:
        """Create a RELATED_TO relationship between concepts."""
        query = """
        MATCH (c1:Concept {concept_id: $from_concept_id}), (c2:Concept {concept_id: $to_concept_id})
        CREATE (c1)-[r:RELATED_TO {
            relationship_type: $relationship_type,
            strength: $strength,
            co_occurrence_count: $co_occurrence_count,
            confidence: $confidence
        }]->(c2)
        """
        try:
            with self.driver.session() as session:
                session.run(query,
                           from_concept_id=from_concept_id,
                           to_concept_id=to_concept_id,
                           relationship_type=relationship.relationship_type,
                           strength=relationship.strength,
                           co_occurrence_count=relationship.co_occurrence_count,
                           confidence=relationship.confidence)
            return True
        except Exception as e:
            print(f"Error creating RELATED_TO relationship: {e}")
            return False
    
    def create_searched_relationship(self, user_id: str, query_id: str, 
                                   relationship: Searched) -> bool:
        """Create a SEARCHED relationship between user and query."""
        query = """
        MATCH (u:User {user_id: $user_id}), (q:Query {query_id: $query_id})
        CREATE (u)-[r:SEARCHED {
            timestamp: $timestamp,
            session_id: $session_id,
            context: $context
        }]->(q)
        """
        try:
            with self.driver.session() as session:
                session.run(query,
                           user_id=user_id,
                           query_id=query_id,
                           timestamp=relationship.timestamp.isoformat(),
                           session_id=relationship.session_id,
                           context=relationship.context)
            return True
        except Exception as e:
            print(f"Error creating SEARCHED relationship: {e}")
            return False
    
    def create_retrieved_relationship(self, query_id: str, document_id: str,
                                    relationship: Retrieved) -> bool:
        """Create a RETRIEVED relationship between query and document."""
        query = """
        MATCH (q:Query {query_id: $query_id}), (d:Document {document_id: $document_id})
        CREATE (q)-[r:RETRIEVED {
            relevance_score: $relevance_score,
            rank: $rank,
            click_through: $click_through,
            dwell_time: $dwell_time
        }]->(d)
        """
        try:
            with self.driver.session() as session:
                session.run(query,
                           query_id=query_id,
                           document_id=document_id,
                           relevance_score=relationship.relevance_score,
                           rank=relationship.rank,
                           click_through=relationship.click_through,
                           dwell_time=relationship.dwell_time)
            return True
        except Exception as e:
            print(f"Error creating RETRIEVED relationship: {e}")
            return False 
    
    # Query operations
    def find_related_documents(self, concept_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find documents related to a concept."""
        query = """
        MATCH path = (c:Concept {concept_id: $concept_id})<-[*1..""" + str(max_depth) + """]-(d:Document)
        RETURN d.document_id as document_id, d.title as title, d.content_type as content_type,
               d.language as language, d.author as author, length(path) as distance
        ORDER BY distance
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, concept_id=concept_id)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error finding related documents: {e}")
            return []
    
    def find_concept_relationships(self, concept_id: str) -> List[Dict[str, Any]]:
        """Find relationships for a concept."""
        query = """
        MATCH (c:Concept {concept_id: $concept_id})-[r]-(other)
        RETURN type(r) as relationship_type, 
               other.concept_id as related_concept_id,
               other.name as related_concept_name,
               other.type as related_concept_type,
               properties(r) as relationship_properties
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, concept_id=concept_id)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error finding concept relationships: {e}")
            return []
    
    def get_document_concepts(self, document_id: str) -> List[Dict[str, Any]]:
        """Get concepts contained in a document."""
        query = """
        MATCH (d:Document {document_id: $document_id})-[r:CONTAINS]->(c:Concept)
        RETURN c.concept_id as concept_id, c.name as name, c.type as type,
               c.confidence_score as confidence_score, c.frequency as frequency,
               r.relevance_score as relevance_score, r.frequency as contains_frequency
        ORDER BY r.relevance_score DESC
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, document_id=document_id)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error getting document concepts: {e}")
            return []
    
    def get_user_search_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's search history."""
        query = """
        MATCH (u:User {user_id: $user_id})-[s:SEARCHED]->(q:Query)
        RETURN q.query_id as query_id, q.text as query_text, q.intent as intent,
               s.timestamp as search_timestamp, s.session_id as session_id
        ORDER BY s.timestamp DESC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, user_id=user_id, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error getting user search history: {e}")
            return []
    
    def find_similar_queries(self, query_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find queries that retrieved similar documents."""
        query = """
        MATCH (q1:Query {query_id: $query_id})-[:RETRIEVED]->(d:Document)<-[:RETRIEVED]-(q2:Query)
        WHERE q1 <> q2
        RETURN q2.query_id as similar_query_id, q2.text as similar_query_text,
               q2.intent as intent, count(d) as common_documents
        ORDER BY common_documents DESC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, query_id=query_id, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error finding similar queries: {e}")
            return []
    
    def get_concept_co_occurrences(self, concept_id: str, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """Get concepts that co-occur with the given concept."""
        query = """
        MATCH (c1:Concept {concept_id: $concept_id})<-[:CONTAINS]-(d:Document)-[:CONTAINS]->(c2:Concept)
        WHERE c1 <> c2
        WITH c2, count(d) as co_occurrence_count
        WHERE co_occurrence_count >= $min_frequency
        RETURN c2.concept_id as concept_id, c2.name as name, c2.type as type,
               co_occurrence_count
        ORDER BY co_occurrence_count DESC
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, concept_id=concept_id, min_frequency=min_frequency)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error getting concept co-occurrences: {e}")
            return []
    
    # Node retrieval operations
    def get_document_node(self, document_id: str) -> Optional[Document]:
        """Get a document node as a Document model."""
        query = """
        MATCH (d:Document {document_id: $document_id})
        RETURN d.document_id as document_id, d.title as title, d.content_type as content_type,
               d.language as language, d.author as author, d.source as source,
               d.created_at as created_at, d.embedding_count as embedding_count
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, document_id=document_id)
                record = result.single()
                if record:
                    return Document(
                        document_id=record["document_id"],
                        title=record["title"],
                        content_type=record["content_type"],
                        language=record["language"],
                        author=record["author"],
                        source=record["source"],
                        created_at=datetime.fromisoformat(record["created_at"]),
                        embedding_count=record["embedding_count"]
                    )
        except Exception as e:
            print(f"Error getting document node: {e}")
        return None
    
    def get_concept_node(self, concept_id: str) -> Optional[Concept]:
        """Get a concept node as a Concept model."""
        query = """
        MATCH (c:Concept {concept_id: $concept_id})
        RETURN c.concept_id as concept_id, c.name as name, c.type as type,
               c.definition as definition, c.confidence_score as confidence_score,
               c.frequency as frequency, c.created_at as created_at
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, concept_id=concept_id)
                record = result.single()
                if record:
                    return Concept(
                        concept_id=record["concept_id"],
                        name=record["name"],
                        type=record["type"],
                        definition=record["definition"],
                        confidence_score=record["confidence_score"],
                        frequency=record["frequency"],
                        created_at=datetime.fromisoformat(record["created_at"])
                    )
        except Exception as e:
            print(f"Error getting concept node: {e}")
        return None
    
    # Delete operations
    def delete_document_graph(self, document_id: str) -> bool:
        """Delete a document and all its relationships."""
        query = """
        MATCH (d:Document {document_id: $document_id})
        DETACH DELETE d
        """
        try:
            with self.driver.session() as session:
                session.run(query, document_id=document_id)
            return True
        except Exception as e:
            print(f"Error deleting document graph: {e}")
            return False
    
    def delete_concept_node(self, concept_id: str) -> bool:
        """Delete a concept and all its relationships."""
        query = """
        MATCH (c:Concept {concept_id: $concept_id})
        DETACH DELETE c
        """
        try:
            with self.driver.session() as session:
                session.run(query, concept_id=concept_id)
            return True
        except Exception as e:
            print(f"Error deleting concept node: {e}")
            return False
    
    # Graph analytics
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get overall graph statistics."""
        stats_query = """
        MATCH (n)
        RETURN labels(n) as node_type, count(n) as count
        UNION ALL
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        """
        try:
            with self.driver.session() as session:
                result = session.run(stats_query)
                stats = {"nodes": {}, "relationships": {}}
                
                for record in result:
                    if "node_type" in record:
                        node_type = record["node_type"][0] if record["node_type"] else "Unknown"
                        stats["nodes"][node_type] = record["count"]
                    elif "relationship_type" in record:
                        stats["relationships"][record["relationship_type"]] = record["count"]
                
                return stats
        except Exception as e:
            print(f"Error getting graph statistics: {e}")
            return {"nodes": {}, "relationships": {}}
    
    def create_indexes(self):
        """Create database indexes for better performance."""
        indexes = [
            "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.document_id)",
            "CREATE INDEX concept_id_index IF NOT EXISTS FOR (c:Concept) ON (c.concept_id)",
            "CREATE INDEX user_id_index IF NOT EXISTS FOR (u:User) ON (u.user_id)",
            "CREATE INDEX query_id_index IF NOT EXISTS FOR (q:Query) ON (q.query_id)",
            "CREATE INDEX concept_name_index IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX concept_type_index IF NOT EXISTS FOR (c:Concept) ON (c.type)"
        ]
        
        try:
            with self.driver.session() as session:
                for index_query in indexes:
                    session.run(index_query)
            return True
        except Exception as e:
            print(f"Error creating indexes: {e}")
            return False