"""
データ生成モジュールの基本実装
"""
from typing import List, Dict, Any, Optional
import random

class DataGenerator:
    """評価データセット生成クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def generate_questions(self, documents: List[str], num_questions: int = 10) -> List[Dict[str, Any]]:
        """文書から質問を生成"""
        questions = []
        
        question_templates = [
            "{}について説明してください。",
            "{}とは何ですか？",
            "{}の特徴を教えてください。",
            "{}はどのように機能しますか？",
            "{}の利点は何ですか？"
        ]
        
        for i in range(num_questions):
            doc = random.choice(documents)
            template = random.choice(question_templates)
            
            # 簡易的なキーワード抽出
            words = doc.split()
            if len(words) > 5:
                keyword = random.choice(words[:min(10, len(words))])
                question = template.format(keyword)
                
                questions.append({
                    'question': question,
                    'context': [doc],
                    'answer': '',  # 実際のシステムでは生成
                    'ground_truth': ''  # 人手でアノテーション
                })
        
        return questions