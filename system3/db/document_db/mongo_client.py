"""
MongoDB client for RAGShelf document management.
"""
from pymongo import MongoClient as PyMongoClient
from typing import Dict, List, Any, Optional
import datetime
from bson import ObjectId

# Import models
from .models import Document, Chunk, Conversation, Message


class MongoClient:
    """MongoDB client for document storage and management with model support."""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017", database_name: str = "ragshelf"):
        self.client = PyMongoClient(connection_string)
        self.db = self.client[database_name]
    
    def list_collections(self) -> List[str]:
        """Get list of all collections in the database."""
        try:
            return self.db.list_collection_names()
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def count_documents(self, collection_name: str, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in a specific collection."""
        try:
            collection = self.db[collection_name]
            if filter_dict is None:
                filter_dict = {}
            return collection.count_documents(filter_dict)
        except Exception as e:
            print(f"Error counting documents in {collection_name}: {e}")
            return 0
    
    def find_documents(self, collection_name: str, filter_dict: Optional[Dict[str, Any]] = None, 
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find documents in a specific collection."""
        try:
            collection = self.db[collection_name]
            if filter_dict is None:
                filter_dict = {}
            
            cursor = collection.find(filter_dict)
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert ObjectId to string for JSON serialization
            documents = []
            for doc in cursor:
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error finding documents in {collection_name}: {e}")
            return []
    
    def close(self):
        """Close the database connection."""
        self.client.close()