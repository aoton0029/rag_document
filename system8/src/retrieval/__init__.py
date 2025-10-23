"""
検索モジュールの基本実装
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """検索結果のデータクラス"""
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str = ""
    
class BaseRetriever(ABC):
    """検索システムの基底クラス"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """クエリに基づいて検索"""
        pass

class LlamaIndexRetriever(BaseRetriever):
    """LlamaIndexを使用した検索"""
    
    def __init__(self, index, retrieval_config: Dict[str, Any]):
        self.index = index
        self.config = retrieval_config
        self.query_engine = None
        self._setup_query_engine()
    
    def _setup_query_engine(self):
        """クエリエンジンをセットアップ"""
        if self.index is None:
            return
            
        try:
            similarity_top_k = self.config.get('similarity_top_k', 10)
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k
            )
        except Exception as e:
            print(f"Failed to setup query engine: {e}")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """クエリに基づいて検索"""
        if self.query_engine is None:
            return []
            
        try:
            response = self.query_engine.query(query)
            
            results = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes[:top_k]:
                    result = RetrievalResult(
                        content=node.node.text,
                        score=node.score if hasattr(node, 'score') else 1.0,
                        metadata=node.node.metadata or {},
                        chunk_id=node.node.node_id
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Retrieval failed: {e}")
            return []