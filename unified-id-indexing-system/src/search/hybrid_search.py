from typing import List, Dict, Any
from src.db.database_manager import db_manager

class HybridSearch:
    """ハイブリッド検索クラス - ベクトル検索とグラフ検索を組み合わせた検索機能を提供します。"""

    def __init__(self):
        self.vector_search = VectorSearch()
        self.graph_search = GraphSearch()

    def search(self, query: str) -> Dict[str, Any]:
        """クエリに基づいてハイブリッド検索を実行します。

        Args:
            query (str): 検索クエリ

        Returns:
            Dict[str, Any]: 検索結果
        """
        vector_results = self.vector_search.search(query)
        graph_results = self.graph_search.search(query)

        # 結果を統合
        combined_results = self.combine_results(vector_results, graph_results)
        return combined_results

    def combine_results(self, vector_results: List[Dict[str, Any]], graph_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ベクトル検索とグラフ検索の結果を統合します。

        Args:
            vector_results (List[Dict[str, Any]]): ベクトル検索の結果
            graph_results (List[Dict[str, Any]]): グラフ検索の結果

        Returns:
            List[Dict[str, Any]]: 統合された検索結果
        """
        # 結果を統合するロジックを実装
        combined = vector_results + graph_results
        return combined  # ここでは単純に結合していますが、必要に応じて重複排除やソートを行うことができます。