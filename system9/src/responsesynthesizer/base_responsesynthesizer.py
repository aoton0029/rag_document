"""
Response Synthesizer Module
llama_indexのresponse_synthesizersを活用した高度なレスポンス生成機能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

from llama_index.core import Settings
from llama_index.core.response_synthesizers import (
    get_response_synthesizer, BaseSynthesizer, ResponseMode
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import LLM
from llama_index.core.prompts import BasePromptTemplate

from ..utils import get_logger


class SynthesizerType(Enum):
    """シンセサイザータイプ"""
    TREE_SUMMARIZE = "tree_summarize"
    SIMPLE_SUMMARIZE = "simple_summarize"
    REFINE = "refine"
    COMPACT = "compact"
    ACCUMULATE = "accumulate"
    COMPACT_ACCUMULATE = "compact_accumulate"
    GENERATION = "generation"
    NO_TEXT = "no_text"


class ResponseQuality(Enum):
    """レスポンス品質レベル"""
    FAST = "fast"
    BALANCED = "balanced" 
    HIGH_QUALITY = "high_quality"
    CREATIVE = "creative"


@dataclass
class SynthesisConfig:
    """合成設定"""
    synthesizer_type: SynthesizerType
    response_quality: ResponseQuality = ResponseQuality.BALANCED
    streaming: bool = False
    temperature: float = 0.7
    max_tokens: int = 1000
    custom_prompt: Optional[BasePromptTemplate] = None
    use_async: bool = True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SynthesisResult:
    """合成結果"""
    response: Response
    synthesis_config: SynthesisConfig
    source_nodes: List[NodeWithScore]
    processing_time: float
    metadata: Dict[str, Any]


class BaseCustomSynthesizer(ABC):
    """カスタムシンセサイザー基底クラス"""
    
    def __init__(self, config: SynthesisConfig, llm: Optional[LLM] = None):
        self.config = config
        self.llm = llm or Settings.llm
        self.logger = get_logger(f"synthesizer_{self.__class__.__name__}")
    
    @abstractmethod
    async def asynthesize(
        self,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore]
    ) -> SynthesisResult:
        """非同期合成実行"""
        pass
    
    def synthesize(
        self,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore]
    ) -> SynthesisResult:
        """同期合成実行"""
        return asyncio.run(self.asynthesize(query, nodes))


class StandardSynthesizer(BaseCustomSynthesizer):
    """標準シンセサイザー"""
    
    def __init__(self, config: SynthesisConfig, llm: Optional[LLM] = None):
        super().__init__(config, llm)
        
        # ResponseModeマッピング
        mode_mapping = {
            SynthesizerType.TREE_SUMMARIZE: ResponseMode.TREE_SUMMARIZE,
            SynthesizerType.SIMPLE_SUMMARIZE: ResponseMode.SIMPLE_SUMMARIZE,
            SynthesizerType.REFINE: ResponseMode.REFINE,
            SynthesizerType.COMPACT: ResponseMode.COMPACT,
            SynthesizerType.ACCUMULATE: ResponseMode.ACCUMULATE,
            SynthesizerType.COMPACT_ACCUMULATE: ResponseMode.COMPACT_ACCUMULATE,
            SynthesizerType.GENERATION: ResponseMode.GENERATION,
            SynthesizerType.NO_TEXT: ResponseMode.NO_TEXT,
        }
        
        response_mode = mode_mapping.get(config.synthesizer_type, ResponseMode.COMPACT)
        
        # llama_indexのシンセサイザー作成
        self.synthesizer = get_response_synthesizer(
            response_mode=response_mode,
            llm=self.llm,
            streaming=config.streaming,
            use_async=config.use_async
        )
    
    async def asynthesize(
        self,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore]
    ) -> SynthesisResult:
        """非同期合成実行"""
        
        import time
        start_time = time.time()
        
        # QueryBundleに変換
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query
        
        try:
            # 合成実行
            if self.config.use_async:
                response = await self.synthesizer.asynthesize(query_bundle, nodes)
            else:
                response = self.synthesizer.synthesize(query_bundle, nodes)
            
            processing_time = time.time() - start_time
            
            return SynthesisResult(
                response=response,
                synthesis_config=self.config,
                source_nodes=nodes,
                processing_time=processing_time,
                metadata={
                    "synthesizer_type": self.config.synthesizer_type.value,
                    "num_nodes": len(nodes),
                    "query_length": len(query_bundle.query_str)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            
            # エラー時のレスポンス
            error_response = Response(
                response=f"合成エラー: {str(e)}",
                source_nodes=nodes,
                metadata={"error": str(e)}
            )
            
            return SynthesisResult(
                response=error_response,
                synthesis_config=self.config,
                source_nodes=nodes,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class AdaptiveSynthesizer(BaseCustomSynthesizer):
    """適応的シンセサイザー"""
    
    def __init__(self, config: SynthesisConfig, llm: Optional[LLM] = None):
        super().__init__(config, llm)
        
        # 複数のシンセサイザーを準備
        self.synthesizers = {}
        
        for synth_type in SynthesizerType:
            try:
                mode_mapping = {
                    SynthesizerType.TREE_SUMMARIZE: ResponseMode.TREE_SUMMARIZE,
                    SynthesizerType.SIMPLE_SUMMARIZE: ResponseMode.SIMPLE_SUMMARIZE,
                    SynthesizerType.REFINE: ResponseMode.REFINE,
                    SynthesizerType.COMPACT: ResponseMode.COMPACT,
                    SynthesizerType.ACCUMULATE: ResponseMode.ACCUMULATE,
                    SynthesizerType.COMPACT_ACCUMULATE: ResponseMode.COMPACT_ACCUMULATE,
                    SynthesizerType.GENERATION: ResponseMode.GENERATION,
                    SynthesizerType.NO_TEXT: ResponseMode.NO_TEXT,
                }
                
                response_mode = mode_mapping[synth_type]
                synthesizer = get_response_synthesizer(
                    response_mode=response_mode,
                    llm=self.llm,
                    streaming=config.streaming,
                    use_async=config.use_async
                )
                self.synthesizers[synth_type] = synthesizer
                
            except Exception as e:
                self.logger.warning(f"Failed to create synthesizer {synth_type}: {e}")
    
    async def asynthesize(
        self,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore]
    ) -> SynthesisResult:
        """適応的合成実行"""
        
        import time
        start_time = time.time()
        
        # QueryBundleに変換
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query
        
        # 最適なシンセサイザーを選択
        optimal_type = self._select_optimal_synthesizer(query_bundle, nodes)
        
        # 選択されたシンセサイザーで実行
        synthesizer = self.synthesizers.get(optimal_type)
        if not synthesizer:
            # フォールバック
            synthesizer = self.synthesizers.get(SynthesizerType.COMPACT)
        
        try:
            if self.config.use_async and hasattr(synthesizer, 'asynthesize'):
                response = await synthesizer.asynthesize(query_bundle, nodes)
            else:
                response = synthesizer.synthesize(query_bundle, nodes)
            
            processing_time = time.time() - start_time
            
            return SynthesisResult(
                response=response,
                synthesis_config=self.config,
                source_nodes=nodes,
                processing_time=processing_time,
                metadata={
                    "selected_synthesizer": optimal_type.value,
                    "num_nodes": len(nodes),
                    "query_length": len(query_bundle.query_str),
                    "adaptive_selection": True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Adaptive synthesis failed: {e}")
            
            error_response = Response(
                response=f"適応的合成エラー: {str(e)}",
                source_nodes=nodes,
                metadata={"error": str(e)}
            )
            
            return SynthesisResult(
                response=error_response,
                synthesis_config=self.config,
                source_nodes=nodes,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _select_optimal_synthesizer(
        self,
        query: QueryBundle,
        nodes: List[NodeWithScore]
    ) -> SynthesizerType:
        """最適なシンセサイザーを選択"""
        
        num_nodes = len(nodes)
        query_length = len(query.query_str)
        
        # ノード数に基づく選択
        if num_nodes <= 2:
            return SynthesizerType.SIMPLE_SUMMARIZE
        elif num_nodes <= 5:
            return SynthesizerType.COMPACT
        elif num_nodes <= 10:
            return SynthesizerType.TREE_SUMMARIZE
        else:
            return SynthesizerType.REFINE
        
        # より高度な選択ロジックも可能
        # - クエリタイプの分析
        # - ノード内容の類似性
        # - 品質要件


class MultiModeSynthesizer(BaseCustomSynthesizer):
    """マルチモードシンセサイザー"""
    
    def __init__(self, config: SynthesisConfig, llm: Optional[LLM] = None):
        super().__init__(config, llm)
        
        # 複数モードで実行
        self.modes = [
            SynthesizerType.TREE_SUMMARIZE,
            SynthesizerType.REFINE,
            SynthesizerType.COMPACT
        ]
        
        self.synthesizers = {}
        for mode in self.modes:
            try:
                mode_mapping = {
                    SynthesizerType.TREE_SUMMARIZE: ResponseMode.TREE_SUMMARIZE,
                    SynthesizerType.REFINE: ResponseMode.REFINE,
                    SynthesizerType.COMPACT: ResponseMode.COMPACT,
                }
                
                response_mode = mode_mapping[mode]
                synthesizer = get_response_synthesizer(
                    response_mode=response_mode,
                    llm=self.llm,
                    use_async=config.use_async
                )
                self.synthesizers[mode] = synthesizer
                
            except Exception as e:
                self.logger.warning(f"Failed to create synthesizer {mode}: {e}")
    
    async def asynthesize(
        self,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore]
    ) -> SynthesisResult:
        """マルチモード合成実行"""
        
        import time
        start_time = time.time()
        
        # QueryBundleに変換
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query
        
        # 複数のシンセサイザーで並列実行
        tasks = []
        for mode, synthesizer in self.synthesizers.items():
            task = self._safe_synthesize(mode, synthesizer, query_bundle, nodes)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果の結合・選択
        combined_response = self._combine_responses(results, query_bundle, nodes)
        
        processing_time = time.time() - start_time
        
        return SynthesisResult(
            response=combined_response,
            synthesis_config=self.config,
            source_nodes=nodes,
            processing_time=processing_time,
            metadata={
                "num_synthesizers": len(self.synthesizers),
                "synthesis_modes": [mode.value for mode in self.modes],
                "combined": True
            }
        )
    
    async def _safe_synthesize(
        self,
        mode: SynthesizerType,
        synthesizer: Any,
        query: QueryBundle,
        nodes: List[NodeWithScore]
    ) -> tuple:
        """安全な合成実行"""
        
        try:
            if self.config.use_async and hasattr(synthesizer, 'asynthesize'):
                response = await synthesizer.asynthesize(query, nodes)
            else:
                response = synthesizer.synthesize(query, nodes)
            return mode, response, None
        except Exception as e:
            self.logger.error(f"Synthesizer {mode} failed: {e}")
            return mode, None, str(e)
    
    def _combine_responses(
        self,
        results: List[tuple],
        query: QueryBundle,
        nodes: List[NodeWithScore]
    ) -> Response:
        """複数のレスポンスを結合"""
        
        valid_responses = []
        for mode, response, error in results:
            if response and not error:
                valid_responses.append((mode, response))
        
        if not valid_responses:
            # 全て失敗
            return Response(
                response="エラー: 全ての合成モードが失敗しました",
                source_nodes=nodes,
                metadata={"error": "All synthesis modes failed"}
            )
        
        if len(valid_responses) == 1:
            # 1つだけ成功
            return valid_responses[0][1]
        
        # 複数成功した場合は統合
        combined_text = ""
        for mode, response in valid_responses:
            combined_text += f"[{mode.value}モード]\n{response.response}\n\n"
        
        return Response(
            response=combined_text.strip(),
            source_nodes=nodes,
            metadata={
                "combination_method": "multi_mode",
                "successful_modes": [mode.value for mode, _ in valid_responses]
            }
        )


class StreamingSynthesizer(BaseCustomSynthesizer):
    """ストリーミングシンセサイザー"""
    
    def __init__(self, config: SynthesisConfig, llm: Optional[LLM] = None):
        super().__init__(config, llm)
        
        # ストリーミング有効化
        config.streaming = True
        
        self.synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            llm=self.llm,
            streaming=True,
            use_async=config.use_async
        )
    
    async def asynthesize(
        self,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore]
    ) -> SynthesisResult:
        """ストリーミング合成実行"""
        
        import time
        start_time = time.time()
        
        # QueryBundleに変換
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query
        
        try:
            # ストリーミング実行
            streaming_response = await self.synthesizer.asynthesize(query_bundle, nodes)
            
            # ストリームを収集
            full_response = ""
            async for token in streaming_response.async_response_gen():
                full_response += token
                # リアルタイム処理が必要な場合はここでコールバック呼び出し
            
            # 最終レスポンス作成
            final_response = Response(
                response=full_response,
                source_nodes=nodes,
                metadata={"streaming": True}
            )
            
            processing_time = time.time() - start_time
            
            return SynthesisResult(
                response=final_response,
                synthesis_config=self.config,
                source_nodes=nodes,
                processing_time=processing_time,
                metadata={
                    "streaming": True,
                    "response_length": len(full_response)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Streaming synthesis failed: {e}")
            
            error_response = Response(
                response=f"ストリーミング合成エラー: {str(e)}",
                source_nodes=nodes,
                metadata={"error": str(e)}
            )
            
            return SynthesisResult(
                response=error_response,
                synthesis_config=self.config,
                source_nodes=nodes,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class SynthesizerManager:
    """シンセサイザー管理"""
    
    def __init__(self):
        self.logger = get_logger("synthesizer_manager")
        self._synthesizers: Dict[str, BaseCustomSynthesizer] = {}
    
    def register_synthesizer(self, name: str, synthesizer: BaseCustomSynthesizer):
        """シンセサイザーを登録"""
        self._synthesizers[name] = synthesizer
        self.logger.info(f"Registered synthesizer: {name}")
    
    def get_synthesizer(self, name: str) -> Optional[BaseCustomSynthesizer]:
        """シンセサイザーを取得"""
        return self._synthesizers.get(name)
    
    def list_synthesizers(self) -> List[str]:
        """登録済みシンセサイザー一覧"""
        return list(self._synthesizers.keys())
    
    async def synthesize_all(
        self,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore]
    ) -> Dict[str, SynthesisResult]:
        """全シンセサイザーで合成実行"""
        
        tasks = []
        for name, synthesizer in self._synthesizers.items():
            tasks.append(self._safe_synthesize(name, synthesizer, query, nodes))
        
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    async def _safe_synthesize(
        self,
        name: str,
        synthesizer: BaseCustomSynthesizer,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore]
    ) -> tuple:
        """安全な合成実行"""
        
        try:
            result = await synthesizer.asynthesize(query, nodes)
            return name, result
        except Exception as e:
            self.logger.error(f"Synthesizer {name} failed: {e}")
            
            # エラー用の結果作成
            error_result = SynthesisResult(
                response=Response(
                    response=f"エラー: {str(e)}",
                    source_nodes=nodes,
                    metadata={"error": str(e)}
                ),
                synthesis_config=synthesizer.config,
                source_nodes=nodes,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
            
            return name, error_result


# Utility functions
def create_standard_synthesizer(
    synthesizer_type: SynthesizerType = SynthesizerType.COMPACT,
    response_quality: ResponseQuality = ResponseQuality.BALANCED,
    llm: Optional[LLM] = None,
    **kwargs
) -> StandardSynthesizer:
    """標準シンセサイザー作成"""
    
    config = SynthesisConfig(
        synthesizer_type=synthesizer_type,
        response_quality=response_quality,
        **kwargs
    )
    
    return StandardSynthesizer(config, llm)


def create_adaptive_synthesizer(
    response_quality: ResponseQuality = ResponseQuality.HIGH_QUALITY,
    llm: Optional[LLM] = None,
    **kwargs
) -> AdaptiveSynthesizer:
    """適応的シンセサイザー作成"""
    
    config = SynthesisConfig(
        synthesizer_type=SynthesizerType.COMPACT,  # デフォルト（動的選択される）
        response_quality=response_quality,
        **kwargs
    )
    
    return AdaptiveSynthesizer(config, llm)


def create_streaming_synthesizer(
    llm: Optional[LLM] = None,
    **kwargs
) -> StreamingSynthesizer:
    """ストリーミングシンセサイザー作成"""
    
    config = SynthesisConfig(
        synthesizer_type=SynthesizerType.COMPACT,
        streaming=True,
        **kwargs
    )
    
    return StreamingSynthesizer(config, llm)
