## 改良版プロジェクト構造
```
rag-evaluation-framework-v2/
├── README.md
├── requirements.txt
├── config/
│   ├── chunking_configs.yaml
│   ├── embedding_configs.yaml
│   ├── evaluation_configs.yaml
│   └── domain_configs.yaml
├── src/
│   ├── __init__.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── strategies.py
│   │   └── quality_assessor.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── similarity_analyzer.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── strategies.py
│   │   └── coverage_analyzer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── component_evaluator.py
│   │   ├── system_evaluator.py
│   │   ├── human_evaluator.py
│   │   ├── production_evaluator.py
│   │   ├── statistical_validator.py
│   │   ├── llm_judge.py
│   │   └── comprehensive_runner.py
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── adaptive_generator.py
│   │   └── adversarial_generator.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── continuous_pipeline.py
│   │   └── alert_system.py
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── ragas_integration.py
│       └── domain_adapter.py
├── tests/
├── data/
├── results/
└── notebooks/
```

## 改良版主要ファイル

### `requirements.txt`
```txt
llama-index==0.10.57
python-dotenv
setuptools
llama-index
llama-index-core
llama-index-llms-ollama
llama-index-embeddings-ollama
llama-index-vector-stores-milvus
llama-index-storage-docstore-mongodb
llama-index-storage-kvstore-redis
llama-index-storage-index-store-redis
llama-index-graph-stores-neo4j
llama-index-packs-neo4j-query-engine
llama-index-readers-file
llama-index-readers-json
llama-index-readers-mongodb
llama-index-extractors-entity
llama-index-tools-database
llama-index-tools-neo4j
llama-index-experimental
llama-index-llms-huggingface
llama-index-embeddings-langchain
llama-index-readers-database
llama-index-readers-milvus
llama-index-readers-obsidian
llama-index-readers-whisper
llama-index-readers-graphdb-cypher
llama-index-extractors-entity
llama-index-storage-index-store-elasticsearch
llama-index-storage-index-store-mongodb
llama-index-storage-docstore-elasticsearch
llama-index-storage-docstore-redis
llama-index-storage-kvstore-elasticsearch
llama-index-storage-kvstore-mongodb
llama-index-tools-requests
llama-index-tools-mcp
llama-index-multi-modal-llms-ollama
ragas==0.1.7
datasets==2.14.7
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
scipy==1.11.4
pyyaml==6.0.1
pytest==7.4.3
jupyter==1.0.0
tqdm==4.66.1
evaluate==0.4.1
rouge-score==0.1.2
bert-score==0.3.13
openai==1.3.0
tiktoken==0.5.2
langchain==0.1.0
nest-asyncio==1.5.8
```

