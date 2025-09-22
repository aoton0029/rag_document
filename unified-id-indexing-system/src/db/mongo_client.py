from pymongo import MongoClient as BaseMongoClient
from config.settings import settings

class MongoClient:
    """Handles interactions with MongoDB."""
    
    def __init__(self):
        self.client = BaseMongoClient(settings.mongodb_url)
        self.db = self.client[settings.mongodb_database]

    def insert_document(self, collection_name: str, document: dict) -> str:
        """Inserts a document into the specified collection."""
        collection = self.db[collection_name]
        result = collection.insert_one(document)
        return str(result.inserted_id)

    def find_document(self, collection_name: str, query: dict) -> dict:
        """Finds a document in the specified collection."""
        collection = self.db[collection_name]
        return collection.find_one(query)

    def update_document(self, collection_name: str, query: dict, update: dict) -> int:
        """Updates a document in the specified collection."""
        collection = self.db[collection_name]
        result = collection.update_one(query, {'$set': update})
        return result.modified_count

    def delete_document(self, collection_name: str, query: dict) -> int:
        """Deletes a document from the specified collection."""
        collection = self.db[collection_name]
        result = collection.delete_one(query)
        return result.deleted_count

    def close(self):
        """Closes the MongoDB connection."""
        self.client.close()