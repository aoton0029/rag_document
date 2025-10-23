"""
ユーティリティモジュール
"""
import yaml
import json
import os
from typing import Dict, Any, List

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """YAML設定ファイルを読み込み"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Failed to load config from {config_path}: {e}")
        return {}

def save_results(results: Dict[str, Any], output_path: str):
    """結果をファイルに保存"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Failed to save results: {e}")

def load_pdf_documents(pdf_path: str) -> List[Dict[str, Any]]:
    """PDFファイルを読み込み"""
    try:
        from llama_index.readers.file import PDFReader
        
        reader = PDFReader()
        documents = reader.load_data(file=pdf_path)
        
        doc_list = []
        for doc in documents:
            doc_list.append({
                'content': doc.text,
                'metadata': doc.metadata or {}
            })
        
        return doc_list
        
    except Exception as e:
        print(f"Failed to load PDF: {e}")
        return []