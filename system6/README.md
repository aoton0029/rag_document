以下は、llamaindex（現行API概念：Document / Node / Index / Retriever / ServiceContext 等）を使ったRAGシステムの詳細設計（ドキュメント保存フロー、RAG/類似検索フロー）です。ソースコードは含めない。運用・性能・安全面の考慮点も含む。

# 目的
- ドキュメントを取り込み、埋め込みを付与してベクトルDBに保存する。
- 保存済みデータを使って高速かつ根拠付きの回答（RAG）と類似検索を行う。

# 主要コンポーネント
- Ingest Service：ファイルアップロード／パーシング／前処理／チャンク化／埋め込み生成／インデックス登録を行う。
- **Milvus (Vector Store)**：ベクトル検索とANN検索を担う。コレクション単位でのスケーリングが可能。
- **MongoDB (Document DB)**：原文メタ（raw_documents コレクション）、チャンクデータ、バージョン管理、権限情報、生データを格納。
- **Neo4j (Graph DB)**：ドキュメント間の関係性、エンティティ間の関係、知識グラフを管理。
- **Redis (KV Store & Index Store)**：検索結果キャッシュ、セッション管理、高速インデックス、頻出クエリキャッシュを担う。
- llamaindex Index/ServiceContext：Retriever/QueryEngine を組み立て、LLM呼び出し（LLMPredictor）と結合する。
- LLM：回答生成と（必要なら）リランキングに使用。
- API 層：クエリ受付、RAGワークフロー実行、結果返却（本文＋出典）。
- Monitoring / Audit：検索ログ、コスト、品質指標を収集。

# 統合ID管理システム
## ID生成戦略
- **unified_id**: UUID4形式の統合識別子（全DB共通のプライマリキー）
- **global_sequence**: 分散環境での一意性を保証するスノーフレークID
- **correlation_id**: リクエスト単位でのトレーシング用ID

## ID生成フロー
1. ドキュメント受信時に unified_id を生成（例：`doc_uuid4_timestamp`）
2. チャンク分割時に unified_chunk_id を生成（例：`{doc_unified_id}_chunk_{index}`）
3. 各DB操作時に同一IDを使用してデータ整合性を確保

# データモデル（MongoDB コレクション設計）
## raw_documents コレクション
```json
{
  "_id": ObjectId,
  "unified_id": "uuid_string", // 全DB共通の統合ID
  "doc_id": "uuid_string", // 後方互換用（deprecated予定）
  "global_sequence": "number", // スノーフレークID
  "source_uri": "URL/パス",
  "title": "string",
  "author": "string", 
  "created_at": ISODate,
  "language": "ja|en|...",
  "original_format": "pdf|html|txt|docx",
  "checksum": "sha256_hash",
  "version": "string",
  "file_size": "number",
  "metadata": {
    "category": "string",
    "tags": ["tag1", "tag2"],
    "access_level": "public|private|restricted"
  },
  "processing_status": "pending|processing|completed|failed",
  "indexed_at": ISODate,
  "index_status": {
    "mongodb": "completed|failed",
    "milvus": "completed|failed", 
    "neo4j": "completed|failed",
    "redis": "completed|failed"
  },
  "raw_data": "生データ"
}
```

## chunks コレクション
```json
{
  "_id": ObjectId,
  "unified_chunk_id": "uuid_string", // 全DB共通のチャンク統合ID
  "chunk_id": "uuid_string", // 後方互換用（deprecated予定）
  "doc_unified_id": "uuid_string", // raw_documents への統合ID参照
  "doc_id": "uuid_string", // 後方互換用
  "text": "string",
  "chunk_index": "number",
  "offset_start": "number",
  "offset_end": "number", 
  "milvus_vector_id": "string", // Milvusでのベクトルレコード統合ID
  "neo4j_node_id": "string", // Neo4jでのノード統合ID
  "redis_cache_key": "string", // Redisでのキャッシュキー
  "metadata": {
    "section_title": "string",
    "page": "number",
    "language": "string",
    "confidence": "number",
    "channel": "string",
    "entities": ["entity1", "entity2"] // Neo4j連携用
  },
  "embedding_generated_at": ISODate,
  "created_at": ISODate,
  "index_status": {
    "mongodb": "completed|failed",
    "milvus": "completed|failed", 
    "neo4j": "completed|failed",
    "redis": "completed|failed"
  }
}
```

## distributed_index_registry コレクション（新規追加）
```json
{
  "_id": ObjectId,
  "unified_id": "uuid_string", // ドキュメントまたはチャンクの統合ID
  "entity_type": "document|chunk",
  "indexes": {
    "mongodb": {
      "collection": "string",
      "index_keys": ["unified_id", "created_at", "language"],
      "status": "creating|completed|failed",
      "created_at": ISODate
    },
    "milvus": {
      "collection": "string", 
      "index_type": "IVF_FLAT|HNSW",
      "vector_id": "string",
      "status": "creating|completed|failed",
      "created_at": ISODate
    },
    "neo4j": {
      "node_labels": ["Document", "Chunk"],
      "relationship_types": ["CONTAINS", "RELATES_TO"],
      "node_id": "string",
      "status": "creating|completed|failed", 
      "created_at": ISODate
    },
    "redis": {
      "key_pattern": "string",
      "index_name": "string", 
      "ttl": "number",
      "status": "creating|completed|failed",
      "created_at": ISODate
    }
  },
  "overall_status": "pending|creating|completed|failed",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

# 使用するライブラリ
- llama-index
- llama-index-core
- llama-index-llms-ollama
- llama-index-embeddings-ollama
- **llama-index-vector-stores-milvus** # Milvus連携
- **llama-index-storage-docstore-mongodb** # MongoDB Document Store
- **llama-index-storage-kvstore-redis** # Redis KV Store  
- **llama-index-storage-index-store-redis** # Redis Index Store
- **llama-index-graph-stores-neo4j** # Neo4j Graph Store
- **llama-index-packs-neo4j-query-engine** # Neo4j Query Engine
- llama-index-readers-file
- llama-index-readers-json
- llama-index-readers-mongodb
- llama-index-extractors-entity
- llama-index-tools-database
- llama-index-tools-neo4j
- llama-index-experimental
- llama-index-llms-huggingface
- llama-index-embeddings-ollama
- llama-index-embeddings-langchain
- llama-index-readers-database
- llama-index-readers-file
- llama-index-readers-json
- llama-index-readers-milvus
- llama-index-readers-mongodb
- llama-index-readers-obsidian
- llama-index-readers-whisper
- llama-index-readers-graphdb-cypher
- llama-index-vector-stores-milvus
- llama-index-extractors-entity
- llama-index-storage-index-store-elasticsearch
- llama-index-storage-index-store-mongodb
- llama-index-storage-index-store-redis
- llama-index-storage-docstore-elasticsearch
- llama-index-storage-docstore-mongodb
- llama-index-storage-docstore-redis
- llama-index-storage-kvstore-elasticsearch
- llama-index-storage-kvstore-mongodb
- llama-index-storage-kvstore-redis
- llama-index-tools-requests
- llama-index-tools-mcp
- llama-index-tools-neo4j
- llama-index-tools-database
- llama-index-graph-stores-neo4j
- llama-index-packs-neo4j-query-engine
- llama-index-multi-modal-llms-ollama

# ドキュメント保存フロー（統合ID + 分散インデックス）
1. 統合ID生成・受け取り
   - 統合ID生成（unified_id, global_sequence, correlation_id）
   - ソース受領（アップロード、URL、DB取り込み）
   - **MongoDB**: raw_documents に統合IDで初期レコード作成（processing_status: "pending"）
   - **Redis**: 処理状況を統合IDでキャッシュ
   - distributed_index_registry に統合IDエントリ作成

2. 前処理（統合ID継承）
   - OCR（必要時）、エンコーディング正規化
   - 言語検出、不要ノイズ除去（ヘッダ/フッタ/ナビ）
   - **MongoDB**: 統合IDで processing_status を "processing" に更新
   - PII検出/マスキング（ポリシーに基づく）
   - **Redis**: 前処理状況を統合IDでキャッシュ

3. 分割（チャンク統合ID生成）
   - unified_chunk_id 生成（{doc_unified_id}_chunk_{index}形式）
   - トークンベース／文ベースの分割（例：chunk_size=800 tokens, overlap=50）
   - セマンティック境界を保つ工夫（見出しで分割）
   - **MongoDB**: chunks コレクションに統合チャンクIDで保存
   - **Neo4j**: 統合IDでエンティティ抽出してノード・関係性を構築
   - distributed_index_registry にチャンク統合IDエントリ追加

4. 埋め込み生成（統合ID追跡）
   - 一括バッチ化で埋め込みを生成（APIコール回数最小化）
   - **Redis**: 統合IDで埋め込みキャッシュ（同一チャンクの再計算回避）
   - 埋め込みモデルはユースケースに応じて選択（OpenAI/hf/ベクトル化モデル）

5. 全DB統合保存
   - **Milvus**: 統合IDでベクトルとメタを一括インサート、コレクション管理
   - **MongoDB**: chunks に統合IDでmilvus_vector_id, neo4j_node_id を更新
   - **Neo4j**: 統合IDでノード作成、ドキュメント-チャンク関係を構築
   - **Redis**: 統合IDでチャンクデータとメタデータをキャッシュ
   - 各DB保存完了時に distributed_index_registry の該当DB status を "completed" に更新

6. 分散インデックス作成（統合ID基盤）
   - **MongoDB**: 統合IDベースの複合インデックス作成
     ```javascript
     db.raw_documents.createIndex({"unified_id": 1, "created_at": -1})
     db.chunks.createIndex({"unified_chunk_id": 1, "doc_unified_id": 1})
     ```
   - **Milvus**: 統合IDメタデータ付きベクトルインデックス構築
     ```python
     index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
     collection.create_index(field_name="vector", index_params=index_params)
     ```
   - **Neo4j**: 統合IDベースのノード・関係インデックス作成
     ```cypher
     CREATE INDEX unified_doc_index FOR (d:Document) ON (d.unified_id)
     CREATE INDEX unified_chunk_index FOR (c:Chunk) ON (c.unified_chunk_id)
     ```
   - **Redis**: 統合IDベースの検索インデックス作成
     ```python
     FT.CREATE idx:unified ON HASH PREFIX 1 unified: SCHEMA 
       unified_id TEXT SORTABLE 
       doc_unified_id TEXT 
       text TEXT
     ```

7. インデックス状態管理・検証
   - distributed_index_registry で全DBインデックス状態を一元管理
   - 各DBインデックス作成完了時に該当status を "completed" に更新
   - **MongoDB**: 統合IDで processing_status を "completed" に更新
   - StorageContext を MongoDB/Milvus/Neo4j/Redis 統合IDで構成
   - サンプリングで統合ID検索精度確認、重複チェック、ログ出力
   - overall_status を "completed" に更新

8. 完了・監査ログ
   - **MongoDB**: query_logs に統合IDでの処理履歴を保存
   - **Redis**: 統合IDでの処理結果をキャッシュ
   - 全DB横断での統合ID一貫性チェック

## RAG（質問応答）フロー（統合ID活用）
1. 入力クエリの正規化
   - correlation_id 生成でリクエスト追跡
   - **Redis**: セッション情報とクエリ履歴を統合IDベースで取得
   - トークン化、言語判定、簡易前処理（不要文字除去）

2. 統合キャッシュ確認
   - **Redis**: 統合IDベースで類似クエリの結果をチェック
   - ヒットした場合は統合IDトレース付きで高速レスポンス

3. クエリ埋め込み生成
   - **Redis**: 埋め込み結果を統合IDで関連付けてキャッシュ
   - 検索用埋め込みを生成（同じ埋め込みモデルを使用）

4. 統合マルチDB検索
   - **Milvus**: 統合IDメタデータ付きベクトル類似検索で top_k を取得
   - **MongoDB**: 統合IDでメタデータフィルタリング（日付/ドメイン絞り込み）
   - **Neo4j**: 統合IDで関連エンティティ・知識グラフ情報を補強
   - **Redis**: 統合ID検索結果をキャッシュ

5. 統合ID再ランキング（任意）
   - LLM ベースの relevance re-ranker で統合ID上位 N を再評価
   - **Redis**: 統合IDリランキング結果をキャッシュ

6. 統合コンテキスト構築
   - **MongoDB**: unified_chunk_id から完全なテキストとメタデータを取得
   - **Neo4j**: 統合IDで関連する知識グラフ情報を付加
   - プロンプトテンプレートに統合ID情報を埋め込み（長さ管理）

7. LLM 呼び出し・統合ID追跡
   - 指示の明確化（出典必須、推測は明示）
   - correlation_id でLLM呼び出しを追跡

8. 統合ポストプロセス・記録
   - **MongoDB**: query_logs に統合IDでの詳細ログを保存
   - **Redis**: 統合IDレスポンスをキャッシュ
   - 出典整形（unified_chunk_id -> doc_unified_id mapping、ページ情報）

# 類似検索フロー（統合ID連携）
1. **Redis**: 統合IDクエリキャッシュ確認
2. **Milvus**: 統合IDベクトル類似検索実行
3. **MongoDB**: 統合IDメタデータフィルタとテキスト補完
4. **Neo4j**: 統合IDで関連文書の関係性情報を付加
5. **Redis**: 統合ID結果をキャッシュして返却

# 統合ID管理・検索パラメータ
## 統合ID設計原則
- unified_id: 全DB共通プライマリキー（UUID4 + timestamp）
- correlation_id: リクエスト追跡用（分散トレーシング）
- 各DBネイティブIDとの相互参照維持
- 統合IDでのクロスDB結合クエリ最適化

## Milvus 統合ID対応
- vector_id として統合IDを使用
- partition_key として doc_unified_id を活用
- メタデータに unified_chunk_id を含める

## MongoDB 統合ID対応  
- unified_id でのシャーディングキー設計
- 統合ID複合インデックス戦略
- 集約パイプラインでの統合ID活用

## Neo4j 統合ID対応
- ノードプロパティとして unified_id を設定
- 統合IDベースのCypher クエリ最適化
- 関係性において統合ID参照を維持

## Redis 統合ID対応
- 統合IDベースのキー設計パターン
- 統合ID検索インデックス構築
- TTL設定での統合ID一貫性管理

# データ一貫性・整合性管理（統合ID基盤）
## 統合ID整合性チェック
- 定期的な全DB横断統合ID存在確認
- 孤立データ（統合IDが他DBに存在しない）の検出・修復
- distributed_index_registry での状態不整合検出

## 障害時統合ID復旧
- 統合IDベースでの部分復旧戦略
- 各DB障害時の統合ID情報保持
- 統合IDログでの操作履歴追跡

## バックアップ・リストア（統合ID対応）
- 統合IDベースのポイントインタイム復旧
- 各DBバックアップでの統合ID情報保持
- クロスDB復旧時の統合ID整合性確保

# 運用・監視（統合ID統合）
## 統合ID追跡・監視
- 統合IDベースの分散トレーシング
- 各DB操作での統合ID性能監視
- 統合ID重複・衝突の検出・アラート

## 統合メトリクス
- 統合IDベースのレイテンシ測定
- 統合ID検索成功率・品質指標
- distributed_index_registry ベースの健全性監視

# 結論（統合ID + 分散インデックス設計のポイント）
- **統合ID管理**: 全DB共通の統合IDで一貫性とトレーサビリティを確保
- **分散インデックス**: 各DB最適化インデックスを統合IDベースで構築
- **状態管理**: distributed_index_registry での一元的インデックス状態管理
- **整合性保証**: 統合IDベースでの全DB横断整合性チェック機構
- **障害回復**: 統合ID追跡による高精度な部分復旧・修復戦略
- **運用効率**: 統合IDベースの監視・トレーシングで運用負荷軽減
