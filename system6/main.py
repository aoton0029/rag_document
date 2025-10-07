import os
import sys
import asyncio
import logging
from pprint import pprint
from db.database_manager import db_manager
from llama_index.core import Settings
from llm.ollama_connector import OllamaConnector, LlmModelType
import services
from configs.project_config import ProjectConfig
from configs.db_settings import settings

pprint(sys.path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama接続初期化
ollama = OllamaConnector.generate(
    settings.ollama_base_url,
    LlmModelType.elyza8b, 
    LlmModelType.qwen3embedding)

project_config = ProjectConfig()

def initialize_system():
    """システムの初期化"""
    logger.info("システム初期化開始")
    
    # データベース接続初期化
    db_manager.initialize_connections(project_config, ollama, True)
    
    # グローバル設定
    Settings.llm = ollama.llm
    Settings.embed_model = ollama.embedding
    
    logger.info("システム初期化完了")

