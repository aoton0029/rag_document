"""
Token management utilities
PromptHelperによるトークンウィンドウ管理
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class TokenConfig:
    """トークン設定"""
    max_input_tokens: int = 4096
    max_output_tokens: int = 512
    chunk_overlap_ratio: float = 0.1
    chunk_size_limit: Optional[int] = None
    separator: str = "\n\n"


class TokenManager:
    """
    トークン管理クラス
    llama_indexのPromptHelperをラップ
    """
    
    def __init__(self, config: Optional[TokenConfig] = None):
        """
        TokenManagerの初期化
        
        Args:
            config: トークン設定
        """
        self.config = config or TokenConfig()
        self._prompt_helper = None
        
    def get_prompt_helper(self):
        """
        PromptHelperを取得
        
        Returns:
            PromptHelper インスタンス
        """
        if self._prompt_helper is None:
            try:
                from llama_index.core import PromptHelper
                
                self._prompt_helper = PromptHelper(
                    context_window=self.config.max_input_tokens,
                    num_output=self.config.max_output_tokens,
                    chunk_overlap_ratio=self.config.chunk_overlap_ratio,
                    chunk_size_limit=self.config.chunk_size_limit,
                    separator=self.config.separator
                )
            except ImportError:
                logger.warning("PromptHelper利用不可、基本的なトークン管理にフォールバック")
                
        return self._prompt_helper
    
    def truncate_text(
        self, 
        text: str, 
        max_tokens: Optional[int] = None,
        from_end: bool = True
    ) -> str:
        """
        テキストをトークン数に基づいて切り詰める
        
        Args:
            text: 入力テキスト
            max_tokens: 最大トークン数
            from_end: 末尾から切り詰めるか
            
        Returns:
            切り詰められたテキスト
        """
        if max_tokens is None:
            max_tokens = self.config.max_input_tokens
        
        # 簡易的なトークン推定（実際のトークナイザーを使用すべき）
        words = text.split()
        estimated_tokens = len(words)
        
        if estimated_tokens <= max_tokens:
            return text
        
        if from_end:
            # 末尾から切り詰め
            words = words[:max_tokens]
        else:
            # 先頭から切り詰め
            words = words[-max_tokens:]
        
        return ' '.join(words)
    
    def count_tokens(self, text: str) -> int:
        """
        テキストのトークン数を推定
        
        Args:
            text: 入力テキスト
            
        Returns:
            推定トークン数
        """
        # 簡易的な推定（実際のトークナイザーを使用すべき）
        return len(text.split())
    
    def split_text_by_tokens(
        self, 
        text: str, 
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[str]:
        """
        テキストをトークン数に基づいて分割
        
        Args:
            text: 入力テキスト
            chunk_size: チャンクサイズ（トークン数）
            overlap: オーバーラップ（トークン数）
            
        Returns:
            分割されたテキストのリスト
        """
        if chunk_size is None:
            chunk_size = 512
        if overlap is None:
            overlap = int(chunk_size * self.config.chunk_overlap_ratio)
        
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
            
            start = end - overlap
        
        return chunks
    
    def validate_token_limit(
        self, 
        text: str, 
        max_tokens: Optional[int] = None
    ) -> bool:
        """
        テキストがトークン制限内かチェック
        
        Args:
            text: 入力テキスト
            max_tokens: 最大トークン数
            
        Returns:
            制限内ならTrue
        """
        if max_tokens is None:
            max_tokens = self.config.max_input_tokens
        
        token_count = self.count_tokens(text)
        return token_count <= max_tokens
    
    def get_available_context_size(self) -> int:
        """
        利用可能なコンテキストサイズを取得
        
        Returns:
            コンテキストサイズ（トークン数）
        """
        return self.config.max_input_tokens - self.config.max_output_tokens


class TokenCounter:
    """
    トークンカウンター
    実際のトークナイザーを使用してトークン数を計算
    """
    
    def __init__(self, tokenizer_name: str = "gpt-3.5-turbo"):
        """
        TokenCounterの初期化
        
        Args:
            tokenizer_name: トークナイザー名
        """
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        
    def _get_tokenizer(self):
        """トークナイザーを取得"""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.encoding_for_model(self.tokenizer_name)
            except Exception as e:
                logger.warning(f"tiktoken利用不可: {e}、単語分割にフォールバック")
                self._tokenizer = "simple"
        
        return self._tokenizer
    
    def count(self, text: str) -> int:
        """
        テキストのトークン数をカウント
        
        Args:
            text: 入力テキスト
            
        Returns:
            トークン数
        """
        tokenizer = self._get_tokenizer()
        
        if tokenizer == "simple":
            # フォールバック: 単語分割
            return len(text.split())
        else:
            # tiktoken使用
            return len(tokenizer.encode(text))
    
    def encode(self, text: str) -> List[int]:
        """
        テキストをトークンIDにエンコード
        
        Args:
            text: 入力テキスト
            
        Returns:
            トークンIDのリスト
        """
        tokenizer = self._get_tokenizer()
        
        if tokenizer == "simple":
            # フォールバック: 文字列のハッシュ
            return [hash(word) % 50000 for word in text.split()]
        else:
            return tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        トークンIDをテキストにデコード
        
        Args:
            token_ids: トークンIDのリスト
            
        Returns:
            デコードされたテキスト
        """
        tokenizer = self._get_tokenizer()
        
        if tokenizer == "simple":
            raise NotImplementedError("簡易モードではデコードは未サポート")
        else:
            return tokenizer.decode(token_ids)


def create_token_manager(config_dict: Dict[str, Any]) -> TokenManager:
    """
    設定辞書からTokenManagerを作成
    
    Args:
        config_dict: 設定辞書
        
    Returns:
        TokenManager インスタンス
    """
    token_config = TokenConfig(
        max_input_tokens=config_dict.get("max_input_tokens", 4096),
        max_output_tokens=config_dict.get("max_output_tokens", 512),
        chunk_overlap_ratio=config_dict.get("chunk_overlap_ratio", 0.1),
        chunk_size_limit=config_dict.get("chunk_size_limit"),
        separator=config_dict.get("separator", "\n\n")
    )
    
    return TokenManager(token_config)
