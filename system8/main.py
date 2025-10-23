#!/usr/bin/env python3
"""
RAG評価フレームワーク - メインアプリケーション

LlamaIndexを使用したRAGシステムの包括的な評価フレームワーク
論文PDF対応のチャンキング、複数の埋め込みモデル、評価指標での比較分析
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Any
import uuid

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_yaml_config, save_results, load_pdf_documents
from src.chunking.chunker_factory import ChunkerFactory
from src.embedding.embedding_factory import EmbeddingFactory
from src.indexing.indexer_factory import IndexerFactory, create_indexing_pipeline
from src.retrieval import LlamaIndexRetriever
from src.responsesynthesizer import OllamaLLM
from src.evaluation import RAGEvaluator
from src.data_generation import DataGenerator
from src.monitoring import ExperimentLogger

class RAGEvaluationFramework:
    """RAG評価フレームワークのメインクラス"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.configs = self._load_configs()
        self.logger = ExperimentLogger()
        self.results = {}
        
    def _load_configs(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        configs = {}
        config_files = [
            'chunking_configs.yaml',
            'embedding_configs.yaml', 
            'evaluation_configs.yaml',
            'test_patterns.yaml',
            'domain_configs.yaml'
        ]
        
        for config_file in config_files:
            config_path = os.path.join(self.config_dir, config_file)
            config_name = config_file.replace('_configs.yaml', '').replace('.yaml', '')
            configs[config_name] = load_yaml_config(config_path)
            
        return configs
    
    def load_documents(self, document_path: str) -> List[Dict[str, Any]]:
        """文書を読み込み"""
        if document_path.endswith('.pdf'):
            return load_pdf_documents(document_path)
        else:
            # テキストファイルの場合
            try:
                with open(document_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    return [{'content': content, 'metadata': {'source': document_path}}]
            except Exception as e:
                print(f"Failed to load document: {e}")
                return []
    
    def run_single_pattern(self, pattern_name: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """単一のテストパターンを実行"""
        
        # パターン設定を取得
        pattern_config = self._get_pattern_config(pattern_name)
        if not pattern_config:
            print(f"Pattern {pattern_name} not found")
            return {}
        
        experiment_id = str(uuid.uuid4())
        log_entry = self.logger.start_experiment(experiment_id, pattern_config)
        
        try:
            print(f"\n=== Running Pattern: {pattern_name} ===")
            print(f"Chunking: {pattern_config['chunking']}")
            print(f"Embedding: {pattern_config['embedding']}")
            print(f"LLM: {pattern_config['llm']}")
            print(f"Retrieval: {pattern_config['retrieval']}")
            
            # 1. チャンキング設定
            chunking_strategy = pattern_config['chunking']
            chunking_config = self.configs['chunking']['chunking_strategies'][chunking_strategy]
            
            # 2. 埋め込み設定
            embedding_key = pattern_config['embedding']
            embedding_config = self._get_embedding_config(embedding_key)
            
            # 3. インデキシングパイプラインを実行
            pipeline_result = create_indexing_pipeline(
                documents=documents,
                chunking_config={
                    'strategy': chunking_strategy,
                    **chunking_config,
                    'domain_config': self.configs['domain']
                },
                embedding_config=embedding_config,
                indexing_config={
                    'type': 'llamaindex_vector',
                    'storage_path': f'./indices/{pattern_name}_{experiment_id}'
                }
            )
            
            # 4. 検索システムを作成
            retriever = LlamaIndexRetriever(
                pipeline_result['index'],
                {'similarity_top_k': 10}
            )
            
            # 5. レスポンス生成システムを作成
            llm_config = self._get_llm_config(pattern_config['llm'])
            response_generator = OllamaLLM(llm_config)
            
            # 6. 評価データを生成
            data_generator = DataGenerator({})
            test_questions = data_generator.generate_questions(
                [doc['content'] for doc in documents[:5]],  # 最初の5文書から
                num_questions=10
            )
            
            # 7. RAGシステムを評価
            evaluation_results = self._evaluate_rag_system(
                retriever, response_generator, test_questions
            )
            
            # 8. 結果を記録
            results = {
                'pattern_name': pattern_name,
                'pattern_config': pattern_config,
                'experiment_id': experiment_id,
                'chunk_count': pipeline_result['chunk_count'],
                'evaluation_results': evaluation_results,
                'timestamp': time.time()
            }
            
            self.logger.end_experiment(experiment_id, evaluation_results)
            print(f"✓ Pattern {pattern_name} completed successfully")
            
            return results
            
        except Exception as e:
            error_msg = f"Pattern {pattern_name} failed: {e}"
            print(f"✗ {error_msg}")
            self.logger.end_experiment(experiment_id, {}, "failed")
            
            return {
                'pattern_name': pattern_name,
                'experiment_id': experiment_id,
                'error': error_msg,
                'timestamp': time.time()
            }
    
    def run_comparison_experiment(self, patterns: List[str], documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """複数パターンの比較実験を実行"""
        print(f"\\n🚀 Starting comparison experiment with {len(patterns)} patterns")
        
        comparison_results = {}
        
        for pattern in patterns:
            pattern_result = self.run_single_pattern(pattern, documents)
            comparison_results[pattern] = pattern_result
            
            # 進捗表示
            print(f"Progress: {len(comparison_results)}/{len(patterns)} patterns completed")
        
        # 比較分析
        analysis_results = self._analyze_comparison_results(comparison_results)
        
        final_results = {
            'experiment_type': 'comparison',
            'patterns': patterns,
            'individual_results': comparison_results,
            'comparison_analysis': analysis_results,
            'timestamp': time.time()
        }
        
        return final_results
    
    def _get_pattern_config(self, pattern_name: str) -> Dict[str, Any]:
        """パターン設定を取得"""
        test_patterns = self.configs['test_patterns']
        
        # 基本パターンから検索
        for category in ['basic_patterns', 'academic_patterns', 'advanced_patterns']:
            if category in test_patterns and pattern_name in test_patterns[category]:
                return test_patterns[category][pattern_name]
        
        return {}
    
    def _get_embedding_config(self, embedding_key: str) -> Dict[str, Any]:
        """埋め込み設定を取得"""
        provider, model_path = embedding_key.split('/', 1)
        embedding_configs = self.configs['embedding']['embedding_models']
        
        if provider in embedding_configs and model_path in embedding_configs[provider]:
            config = embedding_configs[provider][model_path].copy()
            config['provider'] = provider
            return config
        
        return {}
    
    def _get_llm_config(self, llm_key: str) -> Dict[str, Any]:
        """LLM設定を取得"""
        if llm_key.startswith('ollama/'):
            model_name = llm_key.split('/', 1)[1]
            return {
                'model_name': model_name,
                'base_url': 'http://localhost:11434'
            }
        
        return {'model_name': 'llama2'}
    
    def _evaluate_rag_system(self, retriever, response_generator, test_questions: List[Dict[str, Any]]) -> Dict[str, float]:
        """RAGシステムを評価"""
        evaluator = RAGEvaluator(self.configs.get('evaluation', {}))
        
        # 各質問に対してRAGパイプラインを実行
        evaluation_data = []
        
        for question_data in test_questions:
            question = question_data['question']
            
            # 検索実行
            retrieved_docs = retriever.retrieve(question, top_k=5)
            contexts = [doc.content for doc in retrieved_docs]
            
            # 回答生成
            if contexts:
                answer = response_generator.generate_response(question, contexts)
            else:
                answer = "関連する情報が見つかりませんでした。"
            
            evaluation_data.append({
                'question': question,
                'contexts': contexts,
                'answer': answer,
                'ground_truth': question_data.get('ground_truth', '')
            })
        
        # 評価実行
        results = evaluator.evaluate_rag_system(evaluation_data)
        
        # 基本的な統計も追加
        results.update({
            'total_questions': len(test_questions),
            'avg_context_count': sum(len(item['contexts']) for item in evaluation_data) / len(evaluation_data),
            'avg_answer_length': sum(len(item['answer']) for item in evaluation_data) / len(evaluation_data)
        })
        
        return results
    
    def _analyze_comparison_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """比較結果を分析"""
        analysis = {
            'best_patterns': {},
            'performance_ranking': [],
            'metric_analysis': {}
        }
        
        # 各メトリクスでのベストパターンを特定
        all_metrics = set()
        valid_results = {k: v for k, v in results.items() if 'evaluation_results' in v}
        
        if not valid_results:
            return analysis
        
        for result in valid_results.values():
            all_metrics.update(result['evaluation_results'].keys())
        
        for metric in all_metrics:
            metric_scores = {}
            for pattern, result in valid_results.items():
                if metric in result['evaluation_results']:
                    metric_scores[pattern] = result['evaluation_results'][metric]
            
            if metric_scores:
                best_pattern = max(metric_scores.items(), key=lambda x: x[1])
                analysis['best_patterns'][metric] = {
                    'pattern': best_pattern[0],
                    'score': best_pattern[1]
                }
        
        return analysis

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='RAG Evaluation Framework')
    parser.add_argument('--document', '-d', required=True, help='Path to document file (PDF or text)')
    parser.add_argument('--pattern', '-p', help='Single pattern to run')
    parser.add_argument('--patterns', nargs='+', help='Multiple patterns to compare')
    parser.add_argument('--output', '-o', default='results/experiment_results.json', help='Output file path')
    parser.add_argument('--config-dir', default='config', help='Configuration directory')
    
    args = parser.parse_args()
    
    # フレームワークを初期化
    framework = RAGEvaluationFramework(args.config_dir)
    
    # 文書を読み込み
    print(f"📄 Loading document: {args.document}")
    documents = framework.load_documents(args.document)
    
    if not documents:
        print("❌ No documents loaded. Exiting.")
        return
    
    print(f"✓ Loaded {len(documents)} documents")
    
    # 実験実行
    if args.pattern:
        # 単一パターン実行
        results = framework.run_single_pattern(args.pattern, documents)
    elif args.patterns:
        # 複数パターン比較
        results = framework.run_comparison_experiment(args.patterns, documents)
    else:
        # デフォルト: 基本パターンを全て実行
        basic_patterns = ['pattern_1', 'pattern_2', 'pattern_3']
        results = framework.run_comparison_experiment(basic_patterns, documents)
    
    # 結果を保存
    save_results(results, args.output)
    print(f"\\n📊 Results saved to: {args.output}")
    
    # 簡易サマリー表示
    if 'comparison_analysis' in results:
        print("\\n🏆 Best Patterns by Metric:")
        for metric, info in results['comparison_analysis']['best_patterns'].items():
            print(f"  {metric}: {info['pattern']} (score: {info['score']:.3f})")

if __name__ == "__main__":
    main()