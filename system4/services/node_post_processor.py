from db.database_manager import db_manager
from llm.ollama_connector import OllamaConnector
from llama_index.core.postprocessor import (
    SimilarityPostprocessor, KeywordNodePostprocessor, PrevNextNodePostprocessor, 
    MetadataReplacementPostProcessor, PIINodePostprocessor, LLMRerank, SentenceEmbeddingOptimizer,
    EmbeddingRecencyPostprocessor, FixedRecencyPostprocessor
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from typing import List, Optional, Dict, Any
import logging

class NodePostProcessor:
    def __init__(self, 
                 ollama: OllamaConnector):
        self.ollama = ollama
        self.logger = logging.getLogger(__name__)
        self._postprocessors_cache: Dict[str, BaseNodePostprocessor] = {}
    
    def create_similarity_postprocessor(self, 
                                      similarity_cutoff: float = 0.7) -> SimilarityPostprocessor:
        """類似度による後処理器を作成"""
        try:
            self.logger.info(f"Creating Similarity Postprocessor with cutoff: {similarity_cutoff}")
            
            postprocessor = SimilarityPostprocessor(
                similarity_cutoff=similarity_cutoff
            )
            
            return postprocessor
            
        except Exception as e:
            self.logger.error(f"Failed to create Similarity Postprocessor: {e}")
            raise
    
    def create_keyword_postprocessor(self, 
                                   required_keywords: Optional[List[str]] = None,
                                   exclude_keywords: Optional[List[str]] = None) -> KeywordNodePostprocessor:
        """キーワードによる後処理器を作成"""
        try:
            self.logger.info("Creating Keyword Postprocessor")
            
            postprocessor = KeywordNodePostprocessor(
                required_keywords=required_keywords or [],
                exclude_keywords=exclude_keywords or []
            )
            
            return postprocessor
            
        except Exception as e:
            self.logger.error(f"Failed to create Keyword Postprocessor: {e}")
            raise
    
    def create_prev_next_postprocessor(self, 
                                     docstore=None,
                                     num_nodes: int = 1,
                                     mode: str = "both") -> PrevNextNodePostprocessor:
        """前後のノードを含める後処理器を作成"""
        try:
            self.logger.info("Creating PrevNext Postprocessor")
            
            if docstore is None:
                docstore = db_manager.docstore
            
            postprocessor = PrevNextNodePostprocessor(
                docstore=docstore,
                num_nodes=num_nodes,
                mode=mode
            )
            
            return postprocessor
            
        except Exception as e:
            self.logger.error(f"Failed to create PrevNext Postprocessor: {e}")
            raise
    
    def create_metadata_replacement_postprocessor(self, 
                                                target_key: str = "window") -> MetadataReplacementPostProcessor:
        """メタデータ置換後処理器を作成"""
        try:
            self.logger.info("Creating Metadata Replacement Postprocessor")
            
            postprocessor = MetadataReplacementPostProcessor(
                target_metadata_key=target_key
            )
            
            return postprocessor
            
        except Exception as e:
            self.logger.error(f"Failed to create Metadata Replacement Postprocessor: {e}")
            raise
    
    def create_pii_postprocessor(self, 
                               pii_node_info_type: str = "PII.PERSON") -> PIINodePostprocessor:
        """PII（個人識別情報）後処理器を作成"""
        try:
            self.logger.info("Creating PII Postprocessor")
            
            postprocessor = PIINodePostprocessor(
                pii_node_info_type=pii_node_info_type
            )
            
            return postprocessor
            
        except Exception as e:
            self.logger.error(f"Failed to create PII Postprocessor: {e}")
            raise
    
    def create_llm_rerank_postprocessor(self, 
                                      top_n: int = 5,
                                      choice_batch_size: int = 10) -> LLMRerank:
        """LLMによる再ランキング後処理器を作成"""
        try:
            self.logger.info("Creating LLM Rerank Postprocessor")
            
            postprocessor = LLMRerank(
                llm=self.ollama.llm,
                top_n=top_n,
                choice_batch_size=choice_batch_size
            )
            
            return postprocessor
            
        except Exception as e:
            self.logger.error(f"Failed to create LLM Rerank Postprocessor: {e}")
            raise
    
    def create_sentence_embedding_optimizer(self, 
                                          embed_model=None,
                                          percentile_cutoff: float = 0.5,
                                          threshold_cutoff: float = 0.7) -> SentenceEmbeddingOptimizer:
        """文埋め込み最適化後処理器を作成"""
        try:
            self.logger.info("Creating Sentence Embedding Optimizer")
            
            if embed_model is None:
                embed_model = self.ollama.embedding_model
            
            postprocessor = SentenceEmbeddingOptimizer(
                embed_model=embed_model,
                percentile_cutoff=percentile_cutoff,
                threshold_cutoff=threshold_cutoff
            )
            
            return postprocessor
            
        except Exception as e:
            self.logger.error(f"Failed to create Sentence Embedding Optimizer: {e}")
            raise
    
    def create_combined_postprocessor(self, 
                                    similarity_cutoff: float = 0.7,
                                    required_keywords: Optional[List[str]] = None,
                                    exclude_keywords: Optional[List[str]] = None,
                                    top_n: int = 5,
                                    include_prev_next: bool = False) -> List[BaseNodePostprocessor]:
        """複数の後処理器を組み合わせた後処理器リストを作成"""
        try:
            self.logger.info("Creating Combined Postprocessor")
            
            postprocessors = []
            
            # 1. 類似度フィルタリング
            postprocessors.append(
                self.create_similarity_postprocessor(similarity_cutoff)
            )
            
            # 2. キーワードフィルタリング
            if required_keywords or exclude_keywords:
                postprocessors.append(
                    self.create_keyword_postprocessor(required_keywords, exclude_keywords)
                )
            
            # 3. 前後ノード追加
            if include_prev_next:
                postprocessors.append(
                    self.create_prev_next_postprocessor()
                )
            
            # 4. LLM再ランキング
            postprocessors.append(
                self.create_llm_rerank_postprocessor(top_n=top_n)
            )
            
            # 5. 文埋め込み最適化
            postprocessors.append(
                self.create_sentence_embedding_optimizer()
            )
            
            self.logger.info(f"Combined Postprocessor created with {len(postprocessors)} processors")
            return postprocessors
            
        except Exception as e:
            self.logger.error(f"Failed to create Combined Postprocessor: {e}")
            raise
    
    def apply_postprocessors(self, 
                           nodes: List[NodeWithScore],
                           query_bundle: QueryBundle,
                           postprocessors: List[BaseNodePostprocessor]) -> List[NodeWithScore]:
        """複数の後処理器を順次適用"""
        try:
            self.logger.info(f"Applying {len(postprocessors)} postprocessors to {len(nodes)} nodes")
            
            processed_nodes = nodes
            
            for i, postprocessor in enumerate(postprocessors):
                try:
                    processed_nodes = postprocessor.postprocess_nodes(
                        processed_nodes, query_bundle
                    )
                    self.logger.debug(f"Postprocessor {i+1}/{len(postprocessors)}: {len(processed_nodes)} nodes remaining")
                except Exception as e:
                    self.logger.warning(f"Postprocessor {i+1} failed: {e}")
                    continue
            
            self.logger.info(f"Postprocessing complete: {len(processed_nodes)} nodes remaining")
            return processed_nodes
            
        except Exception as e:
            self.logger.error(f"Failed to apply postprocessors: {e}")
            raise
    
    def get_postprocessor_by_type(self, 
                                processor_type: str,
                                **kwargs) -> BaseNodePostprocessor:
        """タイプに応じた後処理器を取得"""
        try:
            # キャッシュをチェック
            cache_key = f"{processor_type}_{hash(str(sorted(kwargs.items())))}"
            if cache_key in self._postprocessors_cache:
                return self._postprocessors_cache[cache_key]
            
            processors = {
                'similarity': self.create_similarity_postprocessor,
                'keyword': self.create_keyword_postprocessor,
                'prev_next': self.create_prev_next_postprocessor,
                'metadata_replacement': self.create_metadata_replacement_postprocessor,
                'pii': self.create_pii_postprocessor,
                'llm_rerank': self.create_llm_rerank_postprocessor,
                'sentence_optimizer': self.create_sentence_embedding_optimizer
            }
            
            if processor_type not in processors:
                raise ValueError(f"Unknown processor type: {processor_type}")
            
            postprocessor = processors[processor_type](**kwargs)
            
            # キャッシュに保存
            self._postprocessors_cache[cache_key] = postprocessor
            
            return postprocessor
            
        except Exception as e:
            self.logger.error(f"Failed to get postprocessor: {e}")
            raise
    
    def create_quality_filter_chain(self, 
                                  similarity_cutoff: float = 0.7,
                                  min_text_length: int = 50,
                                  max_text_length: int = 5000) -> List[BaseNodePostprocessor]:
        """品質フィルタリング用の後処理器チェーンを作成"""
        try:
            self.logger.info("Creating Quality Filter Chain")
            
            # カスタム品質フィルター
            class QualityFilter(BaseNodePostprocessor):
                def __init__(self, min_length: int, max_length: int):
                    self.min_length = min_length
                    self.max_length = max_length
                
                def _postprocess_nodes(self, 
                                     nodes: List[NodeWithScore],
                                     query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
                    filtered_nodes = []
                    for node in nodes:
                        text_length = len(node.node.text)
                        if self.min_length <= text_length <= self.max_length:
                            filtered_nodes.append(node)
                    return filtered_nodes
            
            postprocessors = [
                self.create_similarity_postprocessor(similarity_cutoff),
                QualityFilter(min_text_length, max_text_length),
                self.create_sentence_embedding_optimizer()
            ]
            
            return postprocessors
            
        except Exception as e:
            self.logger.error(f"Failed to create Quality Filter Chain: {e}")
            raise
    
    def clear_cache(self):
        """後処理器のキャッシュをクリア"""
        self._postprocessors_cache.clear()
        self.logger.info("Postprocessor cache cleared")