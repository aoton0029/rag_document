"""
インデキシングファクトリークラス
"""
from typing import Dict, Any, List
from . import IndexConfig, LlamaIndexManager
from .vector_store_indexer import VectorStoreIndexer, MultiVectorStoreIndexer

class IndexerFactory:
    """インデクサー作成のファクトリークラス"""
    
    @staticmethod
    def create_indexer(index_type: str, config: IndexConfig):
        """指定されたタイプでインデクサーを作成"""
        
        if index_type == "vector":
            return VectorStoreIndexer(config)
        
        elif index_type == "llamaindex_vector":
            manager = LlamaIndexManager(config)
            return LlamaIndexVectorIndexer(manager)
        
        elif index_type == "llamaindex_graph":
            manager = LlamaIndexManager(config)
            return LlamaIndexGraphIndexer(manager)
        
        elif index_type == "llamaindex_tree":
            manager = LlamaIndexManager(config)
            return LlamaIndexTreeIndexer(manager)
        
        elif index_type == "multi_vector":
            # 複数の設定からマルチベクターインデクサーを作成
            return MultiVectorStoreIndexer([config])
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    @staticmethod
    def create_from_config(indexing_config: Dict[str, Any]) -> Dict[str, Any]:
        """設定ファイルからインデクサーを作成"""
        indexers = {}
        
        for name, config_dict in indexing_config.items():
            index_config = IndexConfig(
                index_type=config_dict.get('type', 'vector'),
                storage_path=config_dict.get('storage_path', f'./indices/{name}'),
                vector_store_config=config_dict.get('vector_store', {}),
                embedding_config=config_dict.get('embedding', {}),
                chunk_config=config_dict.get('chunking', {}),
                metadata=config_dict.get('metadata', {})
            )
            
            indexer = IndexerFactory.create_indexer(index_config.index_type, index_config)
            indexers[name] = indexer
        
        return indexers

class LlamaIndexVectorIndexer:
    """LlamaIndex ベクターインデクサーのラッパー"""
    
    def __init__(self, manager: LlamaIndexManager):
        self.manager = manager
        self.index = None
    
    def create_index(self, documents: List[Any]) -> Any:
        """ベクターインデックスを作成"""
        self.index = self.manager.create_vector_index(documents)
        return self.index
    
    def load_index(self, storage_path: str) -> Any:
        """インデックスをロード"""
        self.index = self.manager.load_index(storage_path, "vector")
        return self.index
    
    def save_index(self, storage_path: str) -> bool:
        """インデックスを保存"""
        return self.manager.save_index(storage_path)

class LlamaIndexGraphIndexer:
    """LlamaIndex グラフインデクサーのラッパー"""
    
    def __init__(self, manager: LlamaIndexManager):
        self.manager = manager
        self.index = None
    
    def create_index(self, documents: List[Any]) -> Any:
        """グラフインデックスを作成"""
        self.index = self.manager.create_graph_index(documents)
        return self.index
    
    def load_index(self, storage_path: str) -> Any:
        """インデックスをロード"""
        self.index = self.manager.load_index(storage_path, "graph")
        return self.index
    
    def save_index(self, storage_path: str) -> bool:
        """インデックスを保存"""
        return self.manager.save_index(storage_path)

class LlamaIndexTreeIndexer:
    """LlamaIndex ツリーインデクサーのラッパー"""
    
    def __init__(self, manager: LlamaIndexManager):
        self.manager = manager
        self.index = None
    
    def create_index(self, documents: List[Any]) -> Any:
        """ツリーインデックスを作成"""
        self.index = self.manager.create_tree_index(documents)
        return self.index
    
    def load_index(self, storage_path: str) -> Any:
        """インデックスをロード"""
        self.index = self.manager.load_index(storage_path, "tree")
        return self.index
    
    def save_index(self, storage_path: str) -> bool:
        """インデックスを保存"""
        return self.manager.save_index(storage_path)

def create_indexing_pipeline(documents: List[Any], 
                           chunking_config: Dict[str, Any],
                           embedding_config: Dict[str, Any],
                           indexing_config: Dict[str, Any]) -> Dict[str, Any]:
    """完全なインデキシングパイプラインを作成"""
    
    # 1. チャンキング
    from ..chunking.chunker_factory import ChunkerFactory
    chunker = ChunkerFactory.create_chunker(
        chunking_config.get('strategy', 'fixed_size'),
        chunking_config,
        chunking_config.get('domain_config', {})
    )
    
    # 文書をチャンクに分割
    all_chunks = []
    for doc in documents:
        if isinstance(doc, dict):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
        else:
            content = str(doc)
            metadata = {}
            
        chunks = chunker.chunk(content, metadata)
        all_chunks.extend(chunks)
    
    # 2. インデキシング
    index_config = IndexConfig(
        index_type=indexing_config.get('type', 'vector'),
        storage_path=indexing_config.get('storage_path', './index'),
        vector_store_config=indexing_config.get('vector_store', {}),
        embedding_config=embedding_config,
        chunk_config=chunking_config,
        metadata={'created_from': 'pipeline'}
    )
    
    indexer = IndexerFactory.create_indexer(index_config.index_type, index_config)
    
    # チャンクからインデックスを作成
    chunk_docs = []
    for chunk in all_chunks:
        chunk_docs.append({
            'content': chunk.content,
            'metadata': {
                **chunk.metadata,
                'chunk_id': chunk.id,
                'chunk_type': chunk.chunk_type.value,
                'importance_score': chunk.importance_score
            }
        })
    
    index = indexer.create_index(chunk_docs)
    
    return {
        'indexer': indexer,
        'index': index,
        'chunks': all_chunks,
        'chunk_count': len(all_chunks),
        'config': index_config
    }