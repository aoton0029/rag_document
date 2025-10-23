# 概要
llama_indexを用いたRAGシステムの評価フレームワーク。
チャンキングから評価までを一つの試行として、チャンキングインデクシング手法、AdvancedRAG手法の組み合わせによる試行パターンを作り評価する。
テストパターンごとの評価結果を比較することで、チャンキングインデクシングの最適手法、AdvancedRAGの精度向上を目指すと共に最適手法を見つける。

# 言語
日本語

# 対象
論文PDF、製品マニュアルPDF

# プロジェクト構造
```
rag-evaluation-framework/
├── README.md
├── requirements.txt
├── config/
│   ├── chunking_configs.yaml
│   ├── embedding_configs.yaml
│   ├── llm_configs.yaml
│   ├── tokenizer.yaml
│   ├── evaluation_configs.yaml
│   ├── domain_configs.yaml
│   └── test_patterns.yaml
├── src/
│   ├── __init__.py
│   ├── chunking/
│   ├── indexing/
│   ├── embedding/
│   ├── retrieval/
│   ├── evaluation/
│   ├── query/
│   ├── responsesynthesizer/
│   ├── data_generation/
│   ├── monitoring/
│   └── utils/
├── tests/
├── data/
└── results/
```


# LLM Model
- Qwen/Qwen3-32B (vLLM)
- openai/gpt-oss-120b (vLLM)

# Embedding Model
- qwen3-embedding:8b (Ollama)

# StorageContext
- vector_store: Milvus
  - from llama_index.vector_stores.milvus import MilvusVectorStore
- docstore: Mongodb
  - from llama_index.docstores.mongodb_docstore import MongoDBDocumentStore
- index_store: Redis
  - from llama_index.index_stores.redis_index_store import RedisIndexStore
- graph_store: Neo4j
  - from llama_index.graph_stores.neo4j_graph_store import Neo4jGraphStore

# requirements.txt
```txt
python-dotenv
setuptools
llama-index
llama-index-core
llama-index-llms-ollama
llama-index-llms-vllm
llama-index-embeddings-ollama
llama-index-embeddings-langchain
llama-index-vector-stores-milvus
llama-index-graph-stores-neo4j
llama-index-readers-file
llama-index-readers-json
llama-index-readers-mongodb
llama-index-readers-database
llama-index-readers-milvus
llama-index-readers-obsidian
llama-index-readers-whisper
llama-index-readers-graphdb-cypher
llama-index-storage-index-store-redis
llama-index-storage-docstore-mongodb
llama-index-storage-kvstore-redis
llama-index-tools-database
llama-index-tools-neo4j
llama-index-multi-modal-llms-ollama
llama-index-packs-neo4j-query-engine
llama-index-extractors-entity
llama-index-experimental
transformers
sentence-transformers
ragas
datasets
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
pyyaml
pytest
tqdm
evaluate
rouge-score
bert-score
tiktoken
langchain
nest-asyncio
```
