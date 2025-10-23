"""
トークン管理モジュール
各種トークナイザーのラッパーと統一インターフェース
"""

import tiktoken
from typing import List, Optional, Dict, Any, Union
from transformers import AutoTokenizer
import re


class TokenizerManager:
    """トークナイザー管理クラス"""
    
    def __init__(self):
        self._tokenizers = {}
        self._encoding_cache = {}
    
    def get_tokenizer(self, tokenizer_config: Dict[str, Any]):
        """設定に基づいてトークナイザーを取得"""
        tokenizer_type = tokenizer_config.get("type", "tiktoken")
        
        if tokenizer_type == "tiktoken":
            return self._get_tiktoken_tokenizer(tokenizer_config)
        elif tokenizer_type == "huggingface":
            return self._get_huggingface_tokenizer(tokenizer_config)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    
    def _get_tiktoken_tokenizer(self, config: Dict[str, Any]):
        """tiktoken トークナイザーを取得"""
        encoding_name = config.get("encoding_name", "cl100k_base")
        
        if encoding_name not in self._encoding_cache:
            self._encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        
        return TiktokenWrapper(self._encoding_cache[encoding_name])
    
    def _get_huggingface_tokenizer(self, config: Dict[str, Any]):
        """HuggingFace トークナイザーを取得"""
        tokenizer_name = config.get("tokenizer_name")
        
        if tokenizer_name not in self._tokenizers:
            self._tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
        
        return HuggingFaceWrapper(self._tokenizers[tokenizer_name])


class TiktokenWrapper:
    """tiktoken ラッパークラス"""
    
    def __init__(self, encoding):
        self.encoding = encoding
    
    def encode(self, text: str) -> List[int]:
        """テキストをトークンIDに変換"""
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """トークンIDをテキストに変換"""
        return self.encoding.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        """トークン数をカウント"""
        return len(self.encoding.encode(text))
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """指定されたトークン数でテキストを切り詰め"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])


class HuggingFaceWrapper:
    """HuggingFace トークナイザーラッパークラス"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def encode(self, text: str) -> List[int]:
        """テキストをトークンIDに変換"""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, tokens: List[int]) -> str:
        """トークンIDをテキストに変換"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def count_tokens(self, text: str) -> int:
        """トークン数をカウント"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """指定されたトークン数でテキストを切り詰め"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


class JapaneseTextProcessor:
    """日本語テキスト処理クラス"""
    
    def __init__(self):
        # 日本語の文区切りパターン
        self.sentence_pattern = re.compile(r'[。！？\n]')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
    
    def split_sentences(self, text: str) -> List[str]:
        """文単位で分割"""
        sentences = self.sentence_pattern.split(text)
        # 空文字列を除去し、句読点を復元
        result = []
        for i, sentence in enumerate(sentences[:-1]):  # 最後の空要素を除く
            if sentence.strip():
                # 元の区切り文字を復元
                delimiter = text[len(''.join(sentences[:i+1])) + i]
                result.append(sentence.strip() + delimiter)
        
        # 最後の文が句読点で終わらない場合
        if sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result
    
    def split_paragraphs(self, text: str) -> List[str]:
        """段落単位で分割"""
        paragraphs = self.paragraph_pattern.split(text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def clean_text(self, text: str) -> str:
        """テキストクリーニング"""
        # 連続する空白文字を単一スペースに変換
        text = re.sub(r'\s+', ' ', text)
        # 行頭・行末の空白を除去
        text = text.strip()
        return text
    
    def is_japanese_dominant(self, text: str) -> bool:
        """日本語が主体のテキストかどうか判定"""
        japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
        total_chars = len(re.findall(r'\S', text))  # 空白以外の文字
        
        if total_chars == 0:
            return False
        
        return japanese_chars / total_chars > 0.3


def count_tokens_for_text(text: str, tokenizer_config: Dict[str, Any]) -> int:
    """テキストのトークン数をカウント（ユーティリティ関数）"""
    manager = TokenizerManager()
    tokenizer = manager.get_tokenizer(tokenizer_config)
    return tokenizer.count_tokens(text)


def truncate_text_by_tokens(text: str, max_tokens: int, tokenizer_config: Dict[str, Any]) -> str:
    """トークン数でテキストを切り詰め（ユーティリティ関数）"""
    manager = TokenizerManager()
    tokenizer = manager.get_tokenizer(tokenizer_config)
    return tokenizer.truncate_text(text, max_tokens)