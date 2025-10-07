from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility
from typing import List, Dict, Any, Optional, Tuple
import json

class MilvusClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connect()

    def connect(self):
        connections.connect("default", host=self.host, port=self.port)

    def close(self):
        connections.disconnect("default")

    def has_collection(self, name: str) -> bool:
        return utility.has_collection(name)

    def reset_collection(self, collection_name, index_field_name, fields: Optional[List[FieldSchema]] = None) -> None:
        self.drop_collection(collection_name)
        self.create_collection(collection_name, fields)
        self.create_index(collection_name, index_field_name)

    def create_collection(self,
                          name: str,
                          fields: List[FieldSchema],
                          shard_num: int = 2) -> Collection:
        """"""
        if self.has_collection(name):
            return Collection(name)

        schema = CollectionSchema(fields=fields, description=f"collection_{name}", shards_num=shard_num)
        collection = Collection(name=name, schema=schema)
        return collection

    def drop_collection(self, name: str) -> None:
        try:
            if not self.has_collection(name):
                return False
            Collection(name).drop()
            return True
        except Exception as e:
            return False

    def _get_collection(self, name: str, raise_if_missing: bool = True) -> Optional[Collection]:
        if not self.has_collection(name):
            if raise_if_missing:
                raise ValueError(f"Collection '{name}' does not exist")
            return None
        return Collection(name)

    def create_index(self,
                     collection_name: str,
                     field_name: str,
                     index_type: str = "IVF_FLAT",
                     metric_type: str = "L2",
                     params: Optional[Dict[str, Any]] = None) -> None:
        """
        params example: {"nlist": 128}
        """
        collection = self._get_collection(collection_name)
        if params is None:
            params = {"nlist": 128}
        collection.create_index(field_name, {"index_type": index_type, "metric_type": metric_type, "params": params})

    def load_collection(self, collection_name: str) -> None:
        collection = self._get_collection(collection_name)
        collection.load()

    def release_collection(self, collection_name: str) -> None:
        collection = self._get_collection(collection_name)
        collection.release()

    def count(self, collection_name: str) -> int:
        """Return number of entities in collection."""
        collection = self._get_collection(collection_name)
        return collection.num_entities

    def query(self, collection_name: str, expr: str, output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Query by expression (Milvus expr)."""
        collection = self._get_collection(collection_name)
        res = collection.query(expr=expr, output_fields=output_fields or [])
        return res
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information including schema and statistics."""
        try:
            # Get all collections
            collections = utility.list_collections()
            if not collections:
                return {"collections": [], "total_collections": 0}
            
            collection_info = []
            for collection_name in collections:
                try:
                    collection = Collection(collection_name)
                    
                    # Get schema information
                    schema_info = {
                        "name": collection_name,
                        "description": collection.description,
                        "fields": []
                    }
                    
                    # Add field information
                    for field in collection.schema.fields:
                        field_info = {
                            "name": field.name,
                            "type": str(field.dtype),
                            "is_primary": field.is_primary_key,
                            "auto_id": field.auto_id
                        }
                        if hasattr(field, 'params'):
                            field_info["params"] = field.params
                        schema_info["fields"].append(field_info)
                    
                    # Get collection stats
                    stats = {
                        "num_entities": collection.num_entities,
                        "is_loaded": utility.loading_progress(collection_name)["loading_progress"] == "100%"
                    }
                    
                    collection_info.append({
                        "schema": schema_info,
                        "statistics": stats
                    })
                    
                except Exception as e:
                    collection_info.append({
                        "name": collection_name,
                        "error": str(e)
                    })
            
            return {
                "collections": collection_info,
                "total_collections": len(collections)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def count_vectors(self) -> int:
        """Count the total number of vectors across all collections."""
        try:
            total_count = 0
            collections = utility.list_collections()
            
            for collection_name in collections:
                try:
                    collection = Collection(collection_name)
                    total_count += collection.num_entities
                except Exception as e:
                    print(f"Error counting vectors in collection {collection_name}: {e}")
                    continue
            
            return total_count
            
        except Exception as e:
            print(f"Error counting vectors: {e}")
            return 0
   