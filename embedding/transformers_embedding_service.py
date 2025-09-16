import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging
from pathlib import Path
import json
from ..database.keyvalue_db.redis_client import RedisClient
from ..llms.sentence_transformer_connector import SentenceTransformerConnector

class TransformersEmbeddingService:
    """Transformersライブラリを使用した埋め込み生成サービス"""
    
    def __init__(self, 
                 sentence_transformer_connector: SentenceTransformerConnector = None,
                 redis_client: RedisClient = None):
        self.sentence_transformer_connector = sentence_transformer_connector or SentenceTransformerConnector()
        self.model = None
        self.tokenizer = None
        self.redis_client = redis_client or RedisClient()
        self._initialize()
        # # 次元数を取得
        # with torch.no_grad():
        #     dummy_input = self.tokenizer("test", return_tensors="pt", truncation=True, padding=True)
        #     dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
        #     outputs = self.model(**dummy_input)
        #     self.embedding_dimension = outputs.last_hidden_state.shape[-1]
        
    def _initialize(self):
        """初期化"""
        self.sentence_transformer = self.sentence_transformer_connector.initialize_embedding()

        # transformersモデルにフォールバック
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.self.sentence_transformer.model_name, 
        )
        self.model = AutoModel.from_pretrained(
            self.sentence_transformer.model_name,
        ).to(self.device)
        



