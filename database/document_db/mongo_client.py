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
        self.documents = self.db.documents
        self.chunks = self.db.chunks
        self.conversations = self.db.conversations
        
    # Document operations
    def save_document(self, document: Document) -> str:
        """Save a document using the Document model."""
        doc_dict = document.dict(by_alias=True, exclude={"id"})
        result = self.documents.insert_one(doc_dict)
        return str(result.inserted_id)
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID and return as Document model."""
        doc_dict = self.documents.find_one({"document_id": document_id})
        if doc_dict:
            return Document(**doc_dict)
        return None
    
    def update_document(self, document_id: str, update_data: Dict[str, Any]) -> bool:
        """Update document fields."""
        update_data["updated_at"] = datetime.datetime.utcnow()
        result = self.documents.update_one(
            {"document_id": document_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        # Delete related chunks first
        self.chunks.delete_many({"document_id": document_id})
        
        # Delete document
        result = self.documents.delete_one({"document_id": document_id})
        return result.deleted_count > 0
    
    def search_documents(self, query: Dict[str, Any], limit: int = 10) -> List[Document]:
        """Search documents and return as Document models."""
        cursor = self.documents.find(query).limit(limit)
        return [Document(**doc) for doc in cursor]
    
    def get_documents_by_status(self, status: str) -> List[Document]:
        """Get documents by processing status."""
        cursor = self.documents.find({"processing_status": status})
        return [Document(**doc) for doc in cursor]
    
    # Chunk operations
    def save_chunk(self, chunk: Chunk) -> str:
        """Save a chunk using the Chunk model."""
        chunk_dict = chunk.dict(by_alias=True, exclude={"id"})
        result = self.chunks.insert_one(chunk_dict)
        return str(result.inserted_id)
    
    def save_chunks(self, chunks: List[Chunk]) -> List[str]:
        """Save multiple chunks."""
        chunk_dicts = [chunk.dict(by_alias=True, exclude={"id"}) for chunk in chunks]
        result = self.chunks.insert_many(chunk_dicts)
        return [str(oid) for oid in result.inserted_ids]
    
    def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a document."""
        cursor = self.chunks.find({"document_id": document_id}).sort("chunk_index", 1)
        return [Chunk(**chunk) for chunk in cursor]
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a specific chunk."""
        chunk_dict = self.chunks.find_one({"chunk_id": chunk_id})
        if chunk_dict:
            return Chunk(**chunk_dict)
        return None
    
    def update_chunk_embedding_id(self, chunk_id: str, embedding_vector_id: str) -> bool:
        """Update chunk with embedding vector ID."""
        result = self.chunks.update_one(
            {"chunk_id": chunk_id},
            {"$set": {"embedding_vector_id": embedding_vector_id}}
        )
        return result.modified_count > 0
    
    def delete_chunks_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        result = self.chunks.delete_many({"document_id": document_id})
        return result.deleted_count
    
    # Conversation operations
    def save_conversation(self, conversation: Conversation) -> str:
        """Save a conversation using the Conversation model."""
        conv_dict = conversation.dict(by_alias=True, exclude={"id"})
        result = self.conversations.insert_one(conv_dict)
        return str(result.inserted_id)
    
    def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Get a conversation by session ID."""
        conv_dict = self.conversations.find_one({"session_id": session_id})
        if conv_dict:
            return Conversation(**conv_dict)
        return None
    
    def add_message_to_conversation(self, session_id: str, message: Message) -> bool:
        """Add a message to an existing conversation."""
        result = self.conversations.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message.dict()},
                "$set": {"updated_at": datetime.datetime.utcnow()}
            }
        )
        return result.modified_count > 0
    
    def update_conversation_status(self, session_id: str, status: str) -> bool:
        """Update conversation status."""
        result = self.conversations.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": status,
                    "updated_at": datetime.datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0
    
    def get_user_conversations(self, user_id: str, limit: int = 20) -> List[Conversation]:
        """Get conversations for a user."""
        cursor = self.conversations.find({"user_id": user_id}).limit(limit).sort("created_at", -1)
        return [Conversation(**conv) for conv in cursor]
    
    def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation."""
        result = self.conversations.delete_one({"session_id": session_id})
        return result.deleted_count > 0
        
        return list(self.documents.find(mongo_query))
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents as Document models."""
        cursor = self.documents.find()
        return [Document(**doc) for doc in cursor]
    
    # Collection management
    def create_indexes(self):
        """Create database indexes for better performance."""
        # Document indexes
        self.documents.create_index("document_id", unique=True)
        self.documents.create_index("processing_status")
        self.documents.create_index("created_at")
        self.documents.create_index("content_type")
        self.documents.create_index("meta_keywords")
        
        # Chunk indexes
        self.chunks.create_index("chunk_id", unique=True)
        self.chunks.create_index("document_id")
        self.chunks.create_index([("document_id", 1), ("chunk_index", 1)])
        self.chunks.create_index("embedding_vector_id")
        
        # Conversation indexes
        self.conversations.create_index("session_id", unique=True)
        self.conversations.create_index("user_id")
        self.conversations.create_index("status")
        self.conversations.create_index("created_at")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        return {
            "documents": {
                "count": self.documents.count_documents({}),
                "by_status": list(self.documents.aggregate([
                    {"$group": {"_id": "$processing_status", "count": {"$sum": 1}}}
                ]))
            },
            "chunks": {
                "count": self.chunks.count_documents({})
            },
            "conversations": {
                "count": self.conversations.count_documents({}),
                "by_status": list(self.conversations.aggregate([
                    {"$group": {"_id": "$status", "count": {"$sum": 1}}}
                ]))
            }
        }
    
    def close(self):
        """Close the database connection."""
        self.client.close()