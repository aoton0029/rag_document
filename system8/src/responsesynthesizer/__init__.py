"""
レスポンス生成モジュールの基本実装
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLLM(ABC):
    """LLMの基底クラス"""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: List[str]) -> str:
        """レスポンスを生成"""
        pass

class OllamaLLM(BaseLLM):
    """Ollama LLM実装"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'llama2')
        self.base_url = config.get('base_url', 'http://localhost:11434')
    
    def generate_response(self, prompt: str, context: List[str]) -> str:
        """レスポンスを生成"""
        try:
            from llama_index.llms.ollama import Ollama
            
            llm = Ollama(model=self.model_name, base_url=self.base_url)
            
            # コンテキストを含むプロンプトを作成
            context_text = "\n".join(context)
            full_prompt = f"Context:\n{context_text}\n\nQuestion: {prompt}\n\nAnswer:"
            
            response = llm.complete(full_prompt)
            return str(response)
            
        except Exception as e:
            print(f"Response generation failed: {e}")
            return f"Error generating response: {e}"