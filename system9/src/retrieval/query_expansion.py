"""
クエリ拡張モジュール
検索クエリの拡張とセマンティック強化
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple
import re
from enum import Enum

from .base_retriever import SearchQuery
from ..utils import get_logger


class ExpansionStrategy(Enum):
    """拡張戦略"""
    SYNONYM = "synonym"
    SEMANTIC = "semantic"
    ENTITY = "entity"
    CONTEXT = "context"
    HYBRID = "hybrid"


class BaseQueryExpander(ABC):
    """クエリ拡張基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"query_expander_{self.__class__.__name__}")
        
        self.max_expansions = config.get("max_expansions", 5)
        self.expansion_weight = config.get("expansion_weight", 0.7)  # 元クエリ vs 拡張語の重み
    
    @abstractmethod
    def expand_query(self, query: str) -> List[str]:
        """クエリを拡張（実装必須）"""
        pass
    
    def expand_search_query(self, search_query: SearchQuery) -> SearchQuery:
        """SearchQueryオブジェクトを拡張"""
        
        expanded_texts = self.expand_query(search_query.text)
        
        # 拡張されたクエリを結合
        if expanded_texts and len(expanded_texts) > 1:
            # 元のクエリ + 拡張語
            combined_query = search_query.text + " " + " ".join(expanded_texts[1:self.max_expansions])
        else:
            combined_query = search_query.text
        
        # 新しいSearchQueryオブジェクトを作成
        expanded_search_query = SearchQuery(
            text=combined_query,
            filters=search_query.filters,
            similarity_top_k=search_query.similarity_top_k,
            mode=search_query.mode,
            metadata={
                **(search_query.metadata or {}),
                "expanded": True,
                "original_query": search_query.text,
                "expansion_terms": expanded_texts[1:] if len(expanded_texts) > 1 else []
            }
        )
        
        return expanded_search_query


class SynonymExpander(BaseQueryExpander):
    """同義語展開器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 同義語辞書
        self.synonym_dict = config.get("synonym_dict", {})
        
        # 日本語の基本的な同義語辞書
        self.default_synonyms = {
            "研究": ["調査", "分析", "検討", "考察"],
            "手法": ["方法", "手段", "アプローチ", "技術"],
            "結果": ["成果", "効果", "出力", "アウトプット"],
            "問題": ["課題", "タスク", "イシュー", "困難"],
            "システム": ["仕組み", "機構", "構造", "フレームワーク"],
            "データ": ["情報", "資料", "記録", "統計"],
            "分析": ["解析", "評価", "検証", "診断"],
            "改善": ["向上", "最適化", "強化", "改良"],
            "効果": ["成果", "結果", "影響", "効能"],
            "評価": ["査定", "判定", "測定", "アセスメント"]
        }
        
        # 辞書をマージ
        self.synonym_dict = {**self.default_synonyms, **self.synonym_dict}
    
    def expand_query(self, query: str) -> List[str]:
        """同義語でクエリを拡張"""
        
        expanded_terms = [query]  # 元のクエリを最初に
        
        # クエリから単語を抽出
        words = self._extract_words(query)
        
        for word in words:
            synonyms = self._find_synonyms(word)
            
            # 同義語を追加（重複除去）
            for synonym in synonyms:
                expanded_query = query.replace(word, synonym)
                if expanded_query not in expanded_terms:
                    expanded_terms.append(expanded_query)
        
        return expanded_terms[:self.max_expansions]
    
    def _extract_words(self, query: str) -> List[str]:
        """クエリから重要な単語を抽出"""
        
        # 日本語の単語境界は難しいため、簡易的に空白と句読点で分割
        words = re.findall(r'[\w]+', query)
        
        # 長い単語（2文字以上）のみ抽出
        meaningful_words = [word for word in words if len(word) >= 2]
        
        return meaningful_words
    
    def _find_synonyms(self, word: str) -> List[str]:
        """単語の同義語を検索"""
        
        synonyms = []
        
        # 完全一致
        if word in self.synonym_dict:
            synonyms.extend(self.synonym_dict[word])
        
        # 部分一致（含まれる）
        for key, values in self.synonym_dict.items():
            if word in key or key in word:
                synonyms.extend(values)
        
        return list(set(synonyms))  # 重複除去


class SemanticExpander(BaseQueryExpander):
    """セマンティック拡張器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 埋め込みベースの拡張設定
        self.embedding_config = config.get("embedding", {})
        self.similarity_threshold = config.get("similarity_threshold", 0.8)
        
        # 事前定義された関連語データベース
        self.semantic_clusters = config.get("semantic_clusters", {
            "技術": ["AI", "機械学習", "深層学習", "ニューラルネット", "アルゴリズム", "プログラミング"],
            "研究": ["論文", "実験", "調査", "分析", "検証", "評価", "考察"],
            "データ": ["情報", "統計", "数値", "記録", "ログ", "メトリクス"],
            "性能": ["パフォーマンス", "効率", "速度", "精度", "品質", "最適化"],
            "システム": ["アーキテクチャ", "構造", "設計", "実装", "開発", "構築"]
        })
        
        # 埋め込みモデル（利用可能な場合）
        self.embedder = None
        try:
            from ..embedding import create_embedder
            if self.embedding_config:
                self.embedder = create_embedder(
                    self.embedding_config.get("provider", "sentence_transformers"),
                    self.embedding_config
                )
        except Exception as e:
            self.logger.warning(f"Failed to initialize embedder for semantic expansion: {e}")
    
    def expand_query(self, query: str) -> List[str]:
        """セマンティック拡張"""
        
        if self.embedder:
            return self._embedding_based_expansion(query)
        else:
            return self._cluster_based_expansion(query)
    
    def _embedding_based_expansion(self, query: str) -> List[str]:
        """埋め込みベースの拡張"""
        
        expanded_terms = [query]
        
        try:
            # クエリの埋め込みを取得
            query_embedding = self.embedder.encode_texts([query])
            
            # 候補語を埋め込み
            candidate_terms = []
            for cluster_terms in self.semantic_clusters.values():
                candidate_terms.extend(cluster_terms)
            
            candidate_embeddings = self.embedder.encode_texts(candidate_terms)
            
            # 類似度計算
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(
                query_embedding.embeddings, 
                candidate_embeddings.embeddings
            )[0]
            
            # 高類似度の語を選択
            for i, similarity in enumerate(similarities):
                if similarity > self.similarity_threshold:
                    term = candidate_terms[i]
                    expanded_query = f"{query} {term}"
                    if expanded_query not in expanded_terms:
                        expanded_terms.append(expanded_query)
            
        except Exception as e:
            self.logger.error(f"Embedding-based expansion failed: {e}")
            return self._cluster_based_expansion(query)
        
        return expanded_terms[:self.max_expansions]
    
    def _cluster_based_expansion(self, query: str) -> List[str]:
        """クラスターベースの拡張"""
        
        expanded_terms = [query]
        query_lower = query.lower()
        
        for cluster_key, cluster_terms in self.semantic_clusters.items():
            # クラスターキーがクエリに含まれているかチェック
            if cluster_key in query_lower:
                for term in cluster_terms:
                    expanded_query = f"{query} {term}"
                    if expanded_query not in expanded_terms:
                        expanded_terms.append(expanded_query)
            
            # クラスター内の語がクエリに含まれているかチェック
            for term in cluster_terms:
                if term in query_lower:
                    # 同じクラスター内の他の語を追加
                    for related_term in cluster_terms:
                        if related_term != term:
                            expanded_query = f"{query} {related_term}"
                            if expanded_query not in expanded_terms:
                                expanded_terms.append(expanded_query)
        
        return expanded_terms[:self.max_expansions]


class EntityExpander(BaseQueryExpander):
    """エンティティ展開器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # エンティティ認識設定
        self.entity_types = config.get("entity_types", ["PERSON", "ORG", "PRODUCT", "TECH"])
        
        # 事前定義されたエンティティ辞書
        self.entity_dict = config.get("entity_dict", {
            "PERSON": {
                "研究者": ["科学者", "学者", "専門家", "教授"],
                "開発者": ["エンジニア", "プログラマー", "技術者", "開発チーム"]
            },
            "ORG": {
                "大学": ["研究機関", "学術機関", "教育機関", "研究所"],
                "企業": ["会社", "組織", "団体", "法人"]
            },
            "TECH": {
                "AI": ["人工知能", "機械学習", "深層学習", "ディープラーニング"],
                "データベース": ["DB", "データストア", "データウェアハウス", "データレイク"]
            }
        })
    
    def expand_query(self, query: str) -> List[str]:
        """エンティティ展開"""
        
        expanded_terms = [query]
        
        # エンティティ検出と展開
        for entity_type, entities in self.entity_dict.items():
            for entity, variations in entities.items():
                if entity in query:
                    for variation in variations:
                        expanded_query = query.replace(entity, variation)
                        if expanded_query not in expanded_terms:
                            expanded_terms.append(expanded_query)
        
        return expanded_terms[:self.max_expansions]


class ContextExpander(BaseQueryExpander):
    """コンテキスト拡張器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # コンテキスト履歴
        self.query_history = []
        self.max_history = config.get("max_history", 10)
        
        # ドメイン知識
        self.domain_context = config.get("domain_context", {})
    
    def expand_query(self, query: str) -> List[str]:
        """コンテキスト拡張"""
        
        expanded_terms = [query]
        
        # 履歴ベースの拡張
        if self.query_history:
            # 最近のクエリから関連語を抽出
            recent_terms = self._extract_context_terms()
            
            for term in recent_terms[:3]:  # 上位3つ
                if term not in query:
                    expanded_query = f"{query} {term}"
                    expanded_terms.append(expanded_query)
        
        # ドメイン知識ベースの拡張
        if self.domain_context:
            domain_terms = self._get_domain_terms(query)
            for term in domain_terms:
                expanded_query = f"{query} {term}"
                if expanded_query not in expanded_terms:
                    expanded_terms.append(expanded_query)
        
        # 履歴に追加
        self.query_history.append(query)
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)
        
        return expanded_terms[:self.max_expansions]
    
    def _extract_context_terms(self) -> List[str]:
        """コンテキスト履歴から関連語を抽出"""
        
        term_counts = {}
        
        for past_query in self.query_history[-5:]:  # 最近5件
            words = re.findall(r'[\w]+', past_query)
            for word in words:
                if len(word) >= 2:  # 2文字以上
                    term_counts[word] = term_counts.get(word, 0) + 1
        
        # 頻度順でソート
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [term for term, count in sorted_terms]
    
    def _get_domain_terms(self, query: str) -> List[str]:
        """ドメイン知識から関連語を取得"""
        
        related_terms = []
        
        for domain, terms in self.domain_context.items():
            # ドメインキーワードがクエリに含まれているかチェック
            if any(keyword in query.lower() for keyword in terms.get("keywords", [])):
                related_terms.extend(terms.get("related_terms", []))
        
        return related_terms


class CompositeQueryExpander(BaseQueryExpander):
    """複合クエリ拡張器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 各拡張器の重み
        self.expander_weights = config.get("expander_weights", {
            "synonym": 0.3,
            "semantic": 0.4,
            "entity": 0.2,
            "context": 0.1
        })
        
        # 各拡張器を初期化
        self.expanders = {}
        
        if config.get("use_synonym", True):
            self.expanders["synonym"] = SynonymExpander(config.get("synonym_config", {}))
        
        if config.get("use_semantic", True):
            self.expanders["semantic"] = SemanticExpander(config.get("semantic_config", {}))
        
        if config.get("use_entity", True):
            self.expanders["entity"] = EntityExpander(config.get("entity_config", {}))
        
        if config.get("use_context", True):
            self.expanders["context"] = ContextExpander(config.get("context_config", {}))
    
    def expand_query(self, query: str) -> List[str]:
        """複合拡張"""
        
        all_expansions = {}  # expansion -> weight
        
        for expander_name, expander in self.expanders.items():
            try:
                expansions = expander.expand_query(query)
                weight = self.expander_weights.get(expander_name, 1.0)
                
                for expansion in expansions:
                    if expansion not in all_expansions:
                        all_expansions[expansion] = 0
                    all_expansions[expansion] += weight
                
            except Exception as e:
                self.logger.error(f"Expander {expander_name} failed: {e}")
        
        # 重みでソート
        sorted_expansions = sorted(all_expansions.items(), key=lambda x: x[1], reverse=True)
        
        return [expansion for expansion, weight in sorted_expansions[:self.max_expansions]]


class QueryAnalyzer:
    """クエリ分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger("query_analyzer")
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """クエリを分析"""
        
        analysis = {
            "original_query": query,
            "query_length": len(query),
            "word_count": len(query.split()),
            "query_type": self._classify_query_type(query),
            "complexity": self._assess_complexity(query),
            "entities": self._extract_entities(query),
            "keywords": self._extract_keywords(query),
            "intent": self._infer_intent(query)
        }
        
        return analysis
    
    def _classify_query_type(self, query: str) -> str:
        """クエリタイプを分類"""
        
        # 疑問詞チェック
        question_words = ["何", "なに", "どう", "どの", "いつ", "どこ", "なぜ", "誰", "どれ"]
        if any(word in query for word in question_words):
            return "question"
        
        # 命令形チェック
        command_words = ["して", "ください", "教えて", "説明", "比較", "分析"]
        if any(word in query for word in command_words):
            return "command"
        
        # 検索クエリ
        return "search"
    
    def _assess_complexity(self, query: str) -> str:
        """クエリの複雑さを評価"""
        
        word_count = len(query.split())
        
        if word_count <= 3:
            return "simple"
        elif word_count <= 8:
            return "medium"
        else:
            return "complex"
    
    def _extract_entities(self, query: str) -> List[Dict[str, str]]:
        """エンティティを抽出（簡易版）"""
        
        entities = []
        
        # 数値
        numbers = re.findall(r'\d+', query)
        for number in numbers:
            entities.append({"text": number, "type": "NUMBER"})
        
        # 日付パターン
        date_patterns = [r'\d{4}年', r'\d+月', r'\d+日']
        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                entities.append({"text": match, "type": "DATE"})
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """キーワードを抽出"""
        
        # 簡易的なキーワード抽出
        words = re.findall(r'[\w]+', query)
        
        # 長い単語を優先
        keywords = [word for word in words if len(word) >= 2]
        
        # 頻出語は除外（ストップワード）
        stop_words = {"これ", "それ", "この", "その", "です", "である", "ある", "する", "なる"}
        keywords = [word for word in keywords if word not in stop_words]
        
        return keywords[:5]  # 上位5つ
    
    def _infer_intent(self, query: str) -> str:
        """クエリの意図を推定"""
        
        # 比較意図
        if any(word in query for word in ["比較", "違い", "差", "対"]):
            return "comparison"
        
        # 説明意図
        if any(word in query for word in ["説明", "とは", "について", "解説"]):
            return "explanation"
        
        # 方法・手順意図
        if any(word in query for word in ["方法", "手順", "やり方", "how"]):
            return "procedure"
        
        # 原因・理由意図
        if any(word in query for word in ["なぜ", "理由", "原因", "why"]):
            return "causation"
        
        # デフォルト
        return "information_seeking"