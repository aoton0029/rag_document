"""
チャンカーファクトリー
設定に基づいて適切なチャンカーを作成・管理
"""

from typing import Dict, Any, List, Optional, Type, Union
from pathlib import Path

from .base_chunker import BaseChunker, ChunkingStrategy, ChunkEvaluator, ChunkingResult
from .traditional_chunkers import FixedSizeChunker, SentenceBasedChunker, RecursiveChunker, TokenBasedChunker, ParagraphChunker
from .semantic_chunker import SemanticChunker, HierarchicalSemanticChunker, TopicBasedChunker, SemanticSectionChunker
from .adaptive_chunker import AdaptiveChunker
from ..utils import get_logger, ConfigManager, performance_monitor


class ChunkerFactory:
    """チャンカーファクトリークラス"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager
        self.logger = get_logger("chunker_factory")
        
        # 利用可能なチャンカーを登録
        self._chunkers: Dict[str, Type[BaseChunker]] = {
            "fixed_size": FixedSizeChunker,
            "sentence_based": SentenceBasedChunker,
            "recursive": RecursiveChunker,
            "token_based": TokenBasedChunker,
            "paragraph": ParagraphChunker,
            "semantic": SemanticChunker,
            "hierarchical_semantic": HierarchicalSemanticChunker,
            "topic_based": TopicBasedChunker,
            "semantic_section": SemanticSectionChunker,
            "adaptive": AdaptiveChunker
        }
        
        # エイリアス
        self._aliases = {
            "fixed": "fixed_size",
            "sentence": "sentence_based",
            "sentences": "sentence_based",
            "tokens": "token_based",
            "paragraphs": "paragraph",
            "para": "paragraph",
            "semantic_hierarchy": "hierarchical_semantic",
            "topics": "topic_based",
            "sections": "semantic_section",
            "smart": "adaptive",
            "auto": "adaptive"
        }
        
        self.logger.info("ChunkerFactory initialized",
                        available_chunkers=list(self._chunkers.keys()),
                        aliases=self._aliases)
    
    def create_chunker(self, strategy: str, 
                      config: Optional[Dict[str, Any]] = None) -> BaseChunker:
        """指定された戦略でチャンカーを作成"""
        
        # エイリアス解決
        normalized_strategy = self._aliases.get(strategy, strategy)
        
        if normalized_strategy not in self._chunkers:
            raise ValueError(f"Unknown chunking strategy: {strategy}. "
                           f"Available strategies: {list(self._chunkers.keys())}")
        
        # 設定を取得
        chunker_config = self._get_chunker_config(normalized_strategy, config)
        
        # チャンカーを作成
        chunker_class = self._chunkers[normalized_strategy]
        
        try:
            chunker = chunker_class(chunker_config)
            
            self.logger.info("Chunker created", 
                           strategy=normalized_strategy,
                           chunk_size=chunker_config.get("chunk_size"),
                           chunk_overlap=chunker_config.get("chunk_overlap"))
            
            return chunker
            
        except Exception as e:
            self.logger.error(f"Failed to create chunker for {normalized_strategy}: {e}")
            raise
    
    def _get_chunker_config(self, strategy: str, 
                          override_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """チャンカーの設定を取得"""
        
        # デフォルト設定から開始
        base_config = self._get_default_config(strategy)
        
        # 設定マネージャーから設定を取得
        if self.config_manager:
            try:
                file_config = self.config_manager.get_chunking_config(strategy)
                base_config.update(file_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config for {strategy}: {e}")
        
        # オーバーライド設定を適用
        if override_config:
            base_config.update(override_config)
        
        return base_config
    
    def _get_default_config(self, strategy: str) -> Dict[str, Any]:
        """戦略別のデフォルト設定を取得"""
        
        defaults = {
            "fixed_size": {
                "chunk_size": 1024,
                "chunk_overlap": 100,
                "min_chunk_size": 50,
                "max_chunk_size": 4096
            },
            
            "sentence_based": {
                "max_chunk_size": 800,
                "min_chunk_size": 100,
                "overlap_sentences": 1,
                "chunk_overlap": 50
            },
            
            "recursive": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", "。", " ", ""],
                "keep_separator": True
            },
            
            "token_based": {
                "token_chunk_size": 256,
                "token_overlap": 25,
                "chunk_size": 1024,  # 文字数でのフォールバック
                "tokenizer": {"type": "tiktoken", "encoding_name": "cl100k_base"}
            },
            
            "paragraph": {
                "paragraph_separators": ["\n\n", "　　"],
                "merge_short_paragraphs": True,
                "min_paragraph_length": 50,
                "chunk_size": 1500
            },
            
            "semantic": {
                "chunk_size": 512,
                "similarity_threshold": 0.7,
                "sentence_window": 3,
                "clustering_method": "similarity_threshold",
                "max_clusters": 10,
                "embedding": {
                    "provider": "sentence_transformers",
                    "model_name": "all-MiniLM-L6-v2"
                }
            },
            
            "hierarchical_semantic": {
                "chunk_size": 512,
                "similarity_threshold": 0.7,
                "hierarchy_levels": [
                    {"chunk_size": 2048, "similarity_threshold": 0.6},
                    {"chunk_size": 1024, "similarity_threshold": 0.7},
                    {"chunk_size": 512, "similarity_threshold": 0.8}
                ]
            },
            
            "topic_based": {
                "chunk_size": 512,
                "similarity_threshold": 0.7,
                "topic_change_threshold": 0.3,
                "topic_window_size": 5
            },
            
            "semantic_section": {
                "chunk_size": 512,
                "similarity_threshold": 0.7,
                "section_patterns": [
                    r'^#+\s',
                    r'^\d+\.\s',
                    r'^第\d+[章節]\s',
                    r'^[■□▼▽]'
                ],
                "respect_sections": True,
                "max_section_size": 4096
            },
            
            "adaptive": {
                "min_chunk_size": 200,
                "max_chunk_size": 1500,
                "target_chunk_size": 800,
                "content_analysis_enabled": True,
                "structure_detection_enabled": True,
                "semantic_analysis_enabled": True,
                "strategy_weights": {
                    "semantic": 0.4,
                    "sentence_based": 0.3,
                    "fixed_size": 0.2,
                    "structure_based": 0.1
                }
            }
        }
        
        return defaults.get(strategy, {})
    
    def get_available_strategies(self) -> List[str]:
        """利用可能な戦略一覧を取得"""
        return list(self._chunkers.keys())
    
    def get_strategy_aliases(self) -> Dict[str, str]:
        """戦略エイリアス一覧を取得"""
        return self._aliases.copy()
    
    def register_chunker(self, strategy: str, chunker_class: Type[BaseChunker]) -> None:
        """新しいチャンカーを登録"""
        if not issubclass(chunker_class, BaseChunker):
            raise ValueError("Chunker class must inherit from BaseChunker")
        
        self._chunkers[strategy] = chunker_class
        self.logger.info("New chunker registered", 
                        strategy=strategy,
                        class_name=chunker_class.__name__)
    
    def create_chunkers_from_config(self, 
                                   chunkers_config: Dict[str, Dict[str, Any]]) -> Dict[str, BaseChunker]:
        """設定から複数のチャンカーを作成"""
        chunkers = {}
        
        for name, config in chunkers_config.items():
            strategy = config.get("strategy")
            if strategy:
                chunkers[name] = self.create_chunker(strategy, config)
            else:
                self.logger.warning(f"No strategy specified for chunker {name}")
        
        return chunkers
    
    def benchmark_chunkers(self, 
                          text: str,
                          strategies: List[str],
                          configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """複数のチャンキング戦略でベンチマークを実行"""
        
        results = {}
        evaluator = ChunkEvaluator()
        
        for strategy in strategies:
            config = configs.get(strategy, {}) if configs else {}
            
            try:
                chunker = self.create_chunker(strategy, config)
                
                # パフォーマンス測定付きでチャンキング
                @performance_monitor(f"chunking_{strategy}")
                def chunk_with_monitoring():
                    return chunker.chunk_text(text)
                
                chunking_result = chunk_with_monitoring()
                
                # 評価実行
                evaluation = evaluator.evaluate_chunks(chunking_result)
                
                # 結果をまとめる
                results[strategy] = {
                    "success": True,
                    "chunk_count": len(chunking_result.chunks),
                    "total_chars": sum(len(chunk.text) for chunk in chunking_result.chunks),
                    "avg_chunk_length": chunking_result.metadata.get("avg_chunk_length", 0),
                    "evaluation": evaluation,
                    "overall_score": evaluation.get("overall_score", 0)
                }
                
            except Exception as e:
                self.logger.error(f"Benchmarking failed for {strategy}: {e}")
                results[strategy] = {
                    "success": False,
                    "error": str(e)
                }
        
        # 最良の戦略を特定
        successful_results = {k: v for k, v in results.items() if v.get("success", False)}
        if successful_results:
            best_strategy = max(successful_results.keys(), 
                              key=lambda k: successful_results[k].get("overall_score", 0))
            results["best_strategy"] = best_strategy
        
        return results
    
    def validate_chunker_availability(self, strategy: str) -> Dict[str, Any]:
        """チャンカーの利用可能性を検証"""
        validation_result = {
            "strategy": strategy,
            "available": False,
            "issues": []
        }
        
        # エイリアス解決
        normalized_strategy = self._aliases.get(strategy, strategy)
        
        if normalized_strategy not in self._chunkers:
            validation_result["issues"].append(f"Unknown strategy: {strategy}")
            return validation_result
        
        try:
            # 簡単なテストチャンキングを試行
            test_config = self._get_default_config(normalized_strategy)
            chunker = self._chunkers[normalized_strategy](test_config)
            test_result = chunker.chunk_text("これはテスト文です。チャンキングが正常に動作するかテストしています。")
            
            if test_result.chunks and len(test_result.chunks) > 0:
                validation_result["available"] = True
                validation_result["test_chunk_count"] = len(test_result.chunks)
            else:
                validation_result["issues"].append("No chunks produced in test")
                
        except Exception as e:
            validation_result["issues"].append(f"Initialization error: {str(e)}")
        
        return validation_result
    
    def get_strategy_recommendations(self, text_analysis: Dict[str, Any]) -> List[str]:
        """テキスト分析に基づいて推奨戦略を提供"""
        recommendations = []
        
        text_length = text_analysis.get("text_length", 0)
        content_type = text_analysis.get("content_type", "unknown")
        structure_complexity = text_analysis.get("structure_complexity", "medium")
        
        # 長さベースの推奨
        if text_length < 500:
            recommendations.append("sentence_based")
        elif text_length > 10000:
            recommendations.append("adaptive")
            recommendations.append("hierarchical_semantic")
        else:
            recommendations.append("semantic")
            recommendations.append("recursive")
        
        # コンテンツタイプベースの推奨
        if content_type == "academic_paper":
            recommendations.extend(["semantic", "semantic_section"])
        elif content_type == "manual":
            recommendations.extend(["recursive", "paragraph"])
        elif content_type == "code":
            recommendations.extend(["fixed_size", "token_based"])
        
        # 構造複雑さベースの推奨
        if structure_complexity == "high":
            recommendations.extend(["adaptive", "semantic_section"])
        elif structure_complexity == "low":
            recommendations.extend(["sentence_based", "fixed_size"])
        
        # 重複除去と順序保持
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:5]  # 上位5つまで


# モジュールレベルの便利関数
def create_chunker(strategy: str, 
                  config: Optional[Dict[str, Any]] = None,
                  config_manager: Optional[ConfigManager] = None) -> BaseChunker:
    """便利関数：チャンカーを作成"""
    factory = ChunkerFactory(config_manager)
    return factory.create_chunker(strategy, config)


def get_chunking_strategies() -> List[str]:
    """便利関数：利用可能な戦略一覧を取得"""
    factory = ChunkerFactory()
    return factory.get_available_strategies()


def benchmark_chunking_strategies(text: str, 
                                strategies: Optional[List[str]] = None,
                                configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """便利関数：チャンキング戦略のベンチマーク"""
    factory = ChunkerFactory()
    
    if strategies is None:
        # デフォルトの代表的な戦略
        strategies = ["fixed_size", "sentence_based", "recursive", "semantic", "adaptive"]
    
    return factory.benchmark_chunkers(text, strategies, configs)


def recommend_chunking_strategy(text: str) -> str:
    """便利関数：テキストに最適なチャンキング戦略を推奨"""
    # 簡単なテキスト分析
    analysis = {
        "text_length": len(text),
        "word_count": len(text.split()),
        "sentence_count": text.count('。') + text.count('？') + text.count('！'),
        "paragraph_count": text.count('\n\n') + 1
    }
    
    # コンテンツタイプ推定（簡易版）
    if any(keyword in text.lower() for keyword in ["abstract", "introduction", "methodology", "参考文献"]):
        analysis["content_type"] = "academic_paper"
    elif any(keyword in text.lower() for keyword in ["手順", "操作", "設定", "step"]):
        analysis["content_type"] = "manual"
    else:
        analysis["content_type"] = "general"
    
    # 構造複雑さ推定
    if text.count('#') > 5 or text.count('\n\n') > text.count('\n') / 4:
        analysis["structure_complexity"] = "high"
    elif text.count('\n') < len(text) / 100:
        analysis["structure_complexity"] = "low"
    else:
        analysis["structure_complexity"] = "medium"
    
    factory = ChunkerFactory()
    recommendations = factory.get_strategy_recommendations(analysis)
    
    return recommendations[0] if recommendations else "sentence_based"


def validate_all_chunkers() -> Dict[str, Dict[str, Any]]:
    """便利関数：全チャンカーの利用可能性を検証"""
    factory = ChunkerFactory()
    strategies = factory.get_available_strategies()
    
    results = {}
    for strategy in strategies:
        results[strategy] = factory.validate_chunker_availability(strategy)
    
    return results