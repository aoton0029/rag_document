from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility
from typing import List, Dict, Any, Optional, Tuple
from models import BaseCollection

class MilvusClient:
    """Milvus client for vector storage and similarity search."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connect()

    def connect(self):
        """Connect to Milvus server."""
        connections.connect("default", host=self.host, port=self.port)

    def close(self):
        """Close connection."""
        connections.disconnect("default")

    def has_collection(self, name: str) -> bool:
        """Return True if collection exists."""
        return utility.has_collection(name)

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
        """Drop collection if exists."""
        if self.has_collection(name):
            Collection(name).drop()

    def _get_collection(self, name: str, raise_if_missing: bool = True) -> Optional[Collection]:
        if not self.has_collection(name):
            if raise_if_missing:
                raise ValueError(f"Collection '{name}' does not exist")
            return None
        return Collection(name)

    def create_index(self,
                     collection_name: str,
                     field_name: str = "embedding",
                     index_type: str = "IVF_FLAT",
                     metric_type: str = "L2",
                     params: Optional[Dict[str, Any]] = None) -> None:
        """
        Create an index on the given vector field.
        params example: {"nlist": 128}
        """
        collection = self._get_collection(collection_name)
        if params is None:
            params = {"nlist": 128}
        collection.create_index(field_name, {"index_type": index_type, "metric_type": metric_type, "params": params})

    def load_collection(self, collection_name: str) -> None:
        """Load collection into memory for search."""
        collection = self._get_collection(collection_name)
        collection.load()

    def release_collection(self, collection_name: str) -> None:
        """Release collection from memory."""
        collection = self._get_collection(collection_name)
        collection.release()

    def insert(self, collection_name: str, data: List[BaseCollection]) -> List[int]:
        """Insert data into collection."""
        if not data:
            return []
        
        collection = self._get_collection(collection_name)
        
        # Convert Pydantic models to Milvus format
        insert_data = []
        for item in data:
            # Convert to dict using aliases
            item_dict = item.model_dump(by_alias=True)
            
            # Handle metadata field - convert to JSON string if it exists
            if 'metadata' in item_dict and item_dict['metadata'] is not None:
                import json
                if isinstance(item_dict['metadata'], dict):
                    item_dict['metadata'] = json.dumps(item_dict['metadata'])
                elif hasattr(item_dict['metadata'], 'dict'):
                    item_dict['metadata'] = json.dumps(item_dict['metadata'].dict(by_alias=True))
            
            insert_data.append(item_dict)
        
        result = collection.insert(insert_data)
        collection.flush()
        
        return result.primary_keys
        
        
    def search(self,
               collection_name: str,
               query_vectors: List[List[float]],
               top_k: int = 10,
               vector_field: str = "embedding_vector",
               params: Optional[Dict[str, Any]] = None,
               output_fields: Optional[List[str]] = None,
               expr: Optional[str] = None) -> List[List[Dict[str, Any]]]:
        """
        Perform a vector search against a collection and return structured results.
        """
        collection = self._get_collection(collection_name)
        # ensure collection is loaded for search
        try:
            collection.load()
        except Exception:
            # ignore if already loaded or load not required
            pass

        # Default search params
        if params is None:
            params = {"nprobe": 10}

        res = collection.search(query_vectors,
                                anns_field=vector_field,
                                param=params,
                                limit=top_k,
                                expr=expr,
                                output_fields=output_fields)

        # Parse results into simple dicts
        formatted_results: List[List[Dict[str, Any]]] = []
        for hits in res:
            row: List[Dict[str, Any]] = []
            for hit in hits:
                item: Dict[str, Any] = {}
                # id / primary key
                item["id"] = getattr(hit, "id", None) or getattr(hit, "pk", None) or getattr(hit, "primary_key", None)
                # score / distance
                score = getattr(hit, "distance", None)
                if score is None:
                    score = getattr(hit, "score", None)
                item["score"] = score

                # attach requested output fields when available
                if output_fields:
                    for field in output_fields:
                        value = None
                        # many SDKs expose hit.entity as dict-like
                        try:
                            if hasattr(hit, "entity") and isinstance(hit.entity, dict):
                                value = hit.entity.get(field)
                            else:
                                # try attribute access
                                value = getattr(hit, field, None)
                        except Exception:
                            value = None
                        item[field] = value

                row.append(item)
            formatted_results.append(row)

        return formatted_results

    def count(self, collection_name: str) -> int:
        """Return number of entities in collection."""
        collection = self._get_collection(collection_name)
        return collection.num_entities

    def query(self, collection_name: str, expr: str, output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Query by expression (Milvus expr)."""
        collection = self._get_collection(collection_name)
        res = collection.query(expr=expr, output_fields=output_fields or [])
        return res