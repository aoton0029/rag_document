import logging
from typing import List
from llama_index.core import Settings
from db.database_utils import verify_stored_data
from main import initialize_system
from configs.processing_config import ProcessingConfig
from services import DocumentIngestionService, IndexingService, QueryService, ChunkingService, RetrieverService
logger = logging.getLogger(__name__)

def from_dir(directory_path:str):
    # サービスの初期化
    ingestion_service = DocumentIngestionService()
    chunking_service = ChunkingService()
    indexing_service = IndexingService()
    query_service = QueryService()

    logging.info("Starting Llama PDF Sample Process")

    ingest_res = ingestion_service.ingest_from_directory("sample_data/pdf")
    if not ingest_res.success:
        logging.error(f"Ingestion failed: {ingest_res.error_message}")
        return
    
    for doc in ingest_res.documents:
        logging.info(f"Document metadata: {doc.metadata}")
        chunk_res = chunking_service.chunk_document(doc)
        vec_index = indexing_service.create_vector_store_index(chunk_res.nodes)
        tree_index = indexing_service.create_tree_index(chunk_res.nodes)
        summary_index = indexing_service.create_summary_index(chunk_res.nodes)
        query_service.create_router_query_engine([vec_index, tree_index, summary_index])


def from_file(file_path):
    config = ProcessingConfig()
    # サービスの初期化
    ingestion_service = DocumentIngestionService(config)
    chunking_service = ChunkingService(config)
    indexing_service = IndexingService()
    query_service = QueryService()
    retriever_service = RetrieverService()

    logger.info("Starting Llama PDF Sample Process")

    ingest_res = ingestion_service.ingest_from_file_path(file_path)
    if not ingest_res.success:
        logging.error(f"Ingestion failed: {ingest_res.error_message}")
        return

    logger.info(f"Document metadata: {ingest_res.documents[0].metadata}")
    chunk_res = chunking_service.chunk_documents(ingest_res.documents)
    logger.info(f'Chunk: {chunk_res.nodes}')
    vec_index = indexing_service.create_vector_store_index(chunk_res.nodes)
    tree_index = indexing_service.create_tree_index(chunk_res.nodes)
    summary_index = indexing_service.create_summary_index(chunk_res.nodes)
    indexes = [vec_index, tree_index, summary_index]
    
    verify_stored_data()

    query_engine = query_service.create_retriever_query_engine(retriever_type='fusion', retrievers=[idx.as_retriever() for idx in indexes])
    
    query = 'Transmedia Storytellingとは何か'
    response = query_engine.query(query)

    logger.info(f'実行クエリ : {query}')
    logger.info(f'応答 : {response}')



def main():
    try:
        initialize_system()
        from_file("sample_data/pdf/Storytelling Across Worlds Transmedia for Creatives.pdf")
        logger.info("Llama PDF Sample Process Completed")
    except Exception as e:
        logger.error(f"エラー : {e}")

main()