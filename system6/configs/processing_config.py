import os
from typing import List, Dict, Any, Optional, Union, BinaryIO
from datetime import datetime
from pydantic import BaseModel

class ProcessingConfig(BaseModel):
    """ドキュメント処理設定"""
    # OCR設定
    ocr_enabled: bool = True
    ocr_confidence_threshold: float = 0.5
    ocr_languages: List[str] = None
    
    # チャンク化設定
    chunk_size: int = 800
    chunk_overlap: int = 50
    chunk_method: str = "sentence"  # "sentence", "token", "simple"
    

