"""
Response Synthesizer Module
レスポンス合成の実装
"""

import logging
from typing import List, Optional, Any, Sequence

from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    BaseSynthesizer,
    ResponseMode
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.base.response.schema import Response

logger = logging.getLogger(__name__)


class ResponseSynthesizerFactory:
    """
    ResponseSynthesizerファクトリークラス
    """
    
    @staticmethod
    def create(
        response_mode: str = "compact",
        use_async: bool = False,
        streaming: bool = False,
        structured_answer_filtering: bool = False,
        **kwargs
    ) -> BaseSynthesizer:
        """
        ResponseSynthesizerを作成
        
        Args:
            response_mode: レスポンスモード
                - "refine": 逐次改善
                - "compact": コンパクト化
                - "tree_summarize": ツリー要約
                - "simple_summarize": シンプル要約
                - "generation": 生成のみ
                - "no_text": テキストなし
                - "accumulate": 蓄積
                - "compact_accumulate": コンパクト蓄積
            use_async: 非同期実行するか
            streaming: ストリーミングするか
            structured_answer_filtering: 構造化回答フィルタリング
            **kwargs: 追加パラメータ
            
        Returns:
            BaseSynthesizer
        """
        try:
            synthesizer = get_response_synthesizer(
                response_mode=ResponseMode(response_mode),
                use_async=use_async,
                streaming=streaming,
                structured_answer_filtering=structured_answer_filtering,
                **kwargs
            )
            
            logger.info(f"ResponseSynthesizerを作成: mode={response_mode}")
            return synthesizer
        except Exception as e:
            logger.error(f"ResponseSynthesizer作成エラー: {e}")
            raise
    
    @staticmethod
    def create_refine_synthesizer(
        llm: Optional[Any] = None,
        streaming: bool = False,
        **kwargs
    ) -> BaseSynthesizer:
        """
        Refine ResponseSynthesizerを作成
        
        Args:
            llm: LLMインスタンス
            streaming: ストリーミングするか
            **kwargs: 追加パラメータ
            
        Returns:
            BaseSynthesizer
        """
        kwargs_dict = {"llm": llm} if llm else {}
        kwargs_dict.update(kwargs)
        
        return ResponseSynthesizerFactory.create(
            response_mode="refine",
            streaming=streaming,
            **kwargs_dict
        )
    
    @staticmethod
    def create_compact_synthesizer(
        llm: Optional[Any] = None,
        streaming: bool = False,
        **kwargs
    ) -> BaseSynthesizer:
        """
        Compact ResponseSynthesizerを作成
        
        Args:
            llm: LLMインスタンス
            streaming: ストリーミングするか
            **kwargs: 追加パラメータ
            
        Returns:
            BaseSynthesizer
        """
        kwargs_dict = {"llm": llm} if llm else {}
        kwargs_dict.update(kwargs)
        
        return ResponseSynthesizerFactory.create(
            response_mode="compact",
            streaming=streaming,
            **kwargs_dict
        )
    
    @staticmethod
    def create_tree_summarize_synthesizer(
        llm: Optional[Any] = None,
        use_async: bool = True,
        streaming: bool = False,
        **kwargs
    ) -> BaseSynthesizer:
        """
        Tree Summarize ResponseSynthesizerを作成
        
        Args:
            llm: LLMインスタンス
            use_async: 非同期実行するか
            streaming: ストリーミングするか
            **kwargs: 追加パラメータ
            
        Returns:
            BaseSynthesizer
        """
        kwargs_dict = {"llm": llm} if llm else {}
        kwargs_dict.update(kwargs)
        
        return ResponseSynthesizerFactory.create(
            response_mode="tree_summarize",
            use_async=use_async,
            streaming=streaming,
            **kwargs_dict
        )
    
    @staticmethod
    def create_simple_summarize_synthesizer(
        llm: Optional[Any] = None,
        streaming: bool = False,
        **kwargs
    ) -> BaseSynthesizer:
        """
        Simple Summarize ResponseSynthesizerを作成
        
        Args:
            llm: LLMインスタンス
            streaming: ストリーミングするか
            **kwargs: 追加パラメータ
            
        Returns:
            BaseSynthesizer
        """
        kwargs_dict = {"llm": llm} if llm else {}
        kwargs_dict.update(kwargs)
        
        return ResponseSynthesizerFactory.create(
            response_mode="simple_summarize",
            streaming=streaming,
            **kwargs_dict
        )


class CustomResponseSynthesizer:
    """
    カスタムResponseSynthesizer
    独自のレスポンス合成ロジックを実装
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        text_qa_template: Optional[str] = None,
        refine_template: Optional[str] = None
    ):
        """
        CustomResponseSynthesizerの初期化
        
        Args:
            llm: LLMインスタンス
            text_qa_template: QAテンプレート
            refine_template: Refineテンプレート
        """
        self.llm = llm
        self.text_qa_template = text_qa_template
        self.refine_template = refine_template
    
    def synthesize(
        self,
        query: str,
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """
        レスポンスを合成
        
        Args:
            query: クエリ
            nodes: ノードとスコアのリスト
            **kwargs: 追加パラメータ
            
        Returns:
            Response
        """
        try:
            # ノードからテキストを抽出
            context_texts = [node.node.get_content() for node in nodes]
            
            # コンテキストを結合
            context = "\n\n".join(context_texts)
            
            # プロンプトを構築
            prompt = self._build_prompt(query, context)
            
            # LLMで生成
            if self.llm:
                response_text = self.llm.complete(prompt).text
            else:
                # LLMがない場合はコンテキストをそのまま返す
                response_text = context
            
            # Responseオブジェクトを作成
            response = Response(
                response=response_text,
                source_nodes=nodes,
                metadata={"query": query}
            )
            
            logger.info(f"レスポンスを合成: query={query}")
            return response
        except Exception as e:
            logger.error(f"レスポンス合成エラー: {e}")
            raise
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        プロンプトを構築
        
        Args:
            query: クエリ
            context: コンテキスト
            
        Returns:
            プロンプト
        """
        if self.text_qa_template:
            return self.text_qa_template.format(
                query=query,
                context=context
            )
        else:
            # デフォルトテンプレート
            return f"""以下のコンテキストを使用して質問に回答してください。

コンテキスト:
{context}

質問: {query}

回答:"""
