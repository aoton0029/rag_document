import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
from llama_index.core import Settings
from llama_index.core.response_synthesizers import (
    get_response_synthesizer, BaseSynthesizer, ResponseMode
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.base.response.schema import Response
from db.database_manager import db_manager
from configs import ProcessingConfig

class ResponseSynthesizer:
    def __init__(self, templates_dir: str = None):
        self.logger = logging.getLogger(__name__)
        
        # テンプレートディレクトリの設定
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent / "templates" / "contexts"
        
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Jinja2環境の設定
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # デフォルト変数
        self.default_variables = {
            'timestamp': datetime.now().isoformat(),
            'system_name': 'RAG System',
            'version': '1.0'
        }
    
    def create_context_from_template(self,
                                   template_name: str,
                                   query: str,
                                   retrieved_nodes: List[NodeWithScore],
                                   variables: Optional[Dict[str, Any]] = None,
                                   response_mode: str = "default") -> str:
        """テンプレートからコンテクストを生成"""
        try:
            self.logger.info(f"Creating context from template: {template_name}")
            
            # テンプレートファイル名の準備
            template_file = f"{template_name}.md"
            
            try:
                template = self.jinja_env.get_template(template_file)
            except TemplateNotFound:
                self.logger.warning(f"Template {template_file} not found, using default")
                template = self.jinja_env.get_template("default.md")
            
            # テンプレート変数の準備
            template_vars = self.default_variables.copy()
            template_vars.update({
                'query': query,
                'retrieved_nodes': self._format_nodes_for_template(retrieved_nodes),
                'response_mode': response_mode,
                'timestamp': datetime.now().isoformat(),
                'node_count': len(retrieved_nodes)
            })
            
            # ユーザー定義変数を追加
            if variables:
                template_vars.update(variables)
            
            # テンプレートをレンダリング
            context = template.render(**template_vars)
            
            self.logger.info("Context generated successfully from template")
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to create context from template: {e}")
            # フォールバック：シンプルなコンテクスト生成
            return self._create_fallback_context(query, retrieved_nodes)
    
    def _format_nodes_for_template(self, nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        """テンプレート用にノードをフォーマット"""
        formatted_nodes = []
        
        for node in nodes:
            formatted_node = {
                'text': node.node.text,
                'score': getattr(node, 'score', None),
                'metadata': getattr(node.node, 'metadata', {}),
                'node_id': getattr(node.node, 'node_id', None),
                'relationships': getattr(node.node, 'relationships', {})
            }
            formatted_nodes.append(formatted_node)
        
        return formatted_nodes
    
    def _create_fallback_context(self, query: str, nodes: List[NodeWithScore]) -> str:
        """フォールバック用のシンプルなコンテクスト"""
        context_parts = [
            f"Query: {query}",
            f"Retrieved {len(nodes)} relevant contexts:",
            ""
        ]
        
        for i, node in enumerate(nodes, 1):
            context_parts.append(f"Context {i}:")
            context_parts.append(node.node.text)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def synthesize_response(self,
                          query: str,
                          retrieved_nodes: List[NodeWithScore],
                          template_name: str = "default",
                          response_mode: str = "compact",
                          variables: Optional[Dict[str, Any]] = None) -> str:
        """レスポンスを合成"""
        try:
            self.logger.info(f"Synthesizing response for query: {query[:50]}...")
            
            # テンプレートからコンテクストを生成
            context = self.create_context_from_template(
                template_name=template_name,
                query=query,
                retrieved_nodes=retrieved_nodes,
                variables=variables,
                response_mode=response_mode
            )
            
            # LLamaIndexのレスポンスシンセサイザーを使用
            response_synthesizer = get_response_synthesizer(
                response_mode=response_mode
            )
            
            # クエリバンドルを作成
            query_bundle = QueryBundle(query_str=query)
            
            # レスポンスを生成
            response = response_synthesizer.synthesize(
                query=query_bundle,
                nodes=retrieved_nodes
            )
            
            self.logger.info("Response synthesized successfully")
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Failed to synthesize response: {e}")
            raise
    
    def synthesize_with_custom_prompt(self,
                                    query: str,
                                    retrieved_nodes: List[NodeWithScore],
                                    custom_prompt: str,
                                    variables: Optional[Dict[str, Any]] = None) -> str:
        """カスタムプロンプトでレスポンスを合成"""
        try:
            self.logger.info("Synthesizing response with custom prompt")
            
            # カスタムプロンプトをテンプレートとして処理
            template = Template(custom_prompt)
            
            # テンプレート変数の準備
            template_vars = self.default_variables.copy()
            template_vars.update({
                'query': query,
                'retrieved_nodes': self._format_nodes_for_template(retrieved_nodes),
                'context': self._create_simple_context(retrieved_nodes)
            })
            
            if variables:
                template_vars.update(variables)
            
            # プロンプトをレンダリング
            formatted_prompt = template.render(**template_vars)
            
            # LLMに直接問い合わせ
            response = Settings.llm.complete(formatted_prompt)
            
            self.logger.info("Custom prompt response generated successfully")
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Failed to synthesize with custom prompt: {e}")
            raise
    
    def _create_simple_context(self, nodes: List[NodeWithScore]) -> str:
        """シンプルなコンテクスト文字列を作成"""
        context_parts = []
        for i, node in enumerate(nodes, 1):
            context_parts.append(f"[Context {i}] {node.node.text}")
        return "\n\n".join(context_parts)
    
    def create_template(self,
                       template_name: str,
                       template_content: str,
                       overwrite: bool = False) -> bool:
        """新しいテンプレートを作成"""
        try:
            template_path = self.templates_dir / f"{template_name}.md"
            
            if template_path.exists() and not overwrite:
                self.logger.warning(f"Template {template_name} already exists")
                return False
            
            template_path.write_text(template_content, encoding='utf-8')
            
            # Jinja2環境をリロード
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            self.logger.info(f"Template {template_name} created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create template: {e}")
            return False
    
    def list_templates(self) -> List[str]:
        """利用可能なテンプレート一覧を取得"""
        try:
            templates = []
            for file_path in self.templates_dir.glob("*.md"):
                templates.append(file_path.stem)
            return sorted(templates)
            
        except Exception as e:
            self.logger.error(f"Failed to list templates: {e}")
            return []
    
    def get_template_content(self, template_name: str) -> Optional[str]:
        """テンプレートの内容を取得"""
        try:
            template_path = self.templates_dir / f"{template_name}.md"
            if template_path.exists():
                return template_path.read_text(encoding='utf-8')
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get template content: {e}")
            return None
    
    def batch_synthesize(self,
                        queries: List[str],
                        retrieved_nodes_list: List[List[NodeWithScore]],
                        template_name: str = "default",
                        variables_list: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """複数のクエリを一括でレスポンス合成"""
        try:
            self.logger.info(f"Batch synthesizing {len(queries)} responses")
            
            responses = []
            variables_list = variables_list or [{}] * len(queries)
            
            for i, (query, nodes, variables) in enumerate(zip(queries, retrieved_nodes_list, variables_list)):
                try:
                    response = self.synthesize_response(
                        query=query,
                        retrieved_nodes=nodes,
                        template_name=template_name,
                        variables=variables
                    )
                    responses.append(response)
                    self.logger.debug(f"Synthesized response {i+1}/{len(queries)}")
                except Exception as e:
                    self.logger.error(f"Failed to synthesize response {i+1}: {e}")
                    responses.append(f"Error: {str(e)}")
            
            self.logger.info("Batch synthesis completed")
            return responses
            
        except Exception as e:
            self.logger.error(f"Failed to batch synthesize: {e}")
            raise
    
    def synthesize_with_metadata_enrichment(self,
                                          query: str,
                                          retrieved_nodes: List[NodeWithScore],
                                          template_name: str = "detailed",
                                          enrich_metadata: bool = True) -> str:
        """メタデータ拡張付きでレスポンスを合成"""
        try:
            self.logger.info("Synthesizing response with metadata enrichment")
            
            # メタデータを拡張
            if enrich_metadata:
                retrieved_nodes = self._enrich_node_metadata(retrieved_nodes)
            
            # 拡張変数を準備
            variables = {
                'analysis_type': 'Detailed',
                'depth_level': 'Comprehensive',
                'include_citations': True,
                'focus_areas': self._extract_focus_areas(query),
                'metadata_enriched': enrich_metadata
            }
            
            return self.synthesize_response(
                query=query,
                retrieved_nodes=retrieved_nodes,
                template_name=template_name,
                variables=variables
            )
            
        except Exception as e:
            self.logger.error(f"Failed to synthesize with metadata enrichment: {e}")
            raise
    
    def _enrich_node_metadata(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """ノードのメタデータを拡張"""
        enriched_nodes = []
        
        for node in nodes:
            # メタデータのコピーを作成
            enriched_metadata = dict(getattr(node.node, 'metadata', {}))
            
            # 追加のメタデータを生成
            enriched_metadata.update({
                'retrieval_timestamp': datetime.now().isoformat(),
                'text_length': len(node.node.text),
                'relevance_score': getattr(node, 'score', 0.0),
                'content_hash': hash(node.node.text)
            })
            
            # 新しいノードを作成
            enriched_node_obj = TextNode(
                text=node.node.text,
                metadata=enriched_metadata,
                node_id=getattr(node.node, 'node_id', None)
            )
            
            enriched_node = NodeWithScore(
                node=enriched_node_obj,
                score=getattr(node, 'score', None)
            )
            
            enriched_nodes.append(enriched_node)
        
        return enriched_nodes
    
    def _extract_focus_areas(self, query: str) -> List[str]:
        """クエリから焦点領域を抽出"""
        # 簡単なキーワード抽出（実際の実装ではより高度な処理を行う）
        focus_keywords = ['分析', '比較', '評価', '説明', '要約', '詳細']
        focus_areas = []
        
        for keyword in focus_keywords:
            if keyword in query:
                focus_areas.append(keyword)
        
        return focus_areas if focus_areas else ['一般的な分析']