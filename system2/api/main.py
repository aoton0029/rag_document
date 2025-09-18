from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import aiofiles
import os
import tempfile

from services.ingest_service import DocumentIngestService
from services.rag_service import RAGService
from services.similarity_service import SimilaritySearchService
from core.database import db_manager

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG System API", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# サービスインスタンス
ingest_service = DocumentIngestService()
rag_service = RAGService()
similarity_service = SimilaritySearchService()

# リクエスト/レスポンスモデル
class RAGQuery(BaseModel):
    query: str
    user_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

class SimilarityQuery(BaseModel):
    query: str
    top_k: Optional[int] = 10
    threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None

class RAGResponse(BaseModel):
    query_id: str
    response: str
    sources: List[Dict[str, Any]]
    processing_time_ms: float

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    try:
        db_manager.initialize_connections()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy", "message": "RAG System is running"}

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """ドキュメントアップロードエンドポイント"""
    try:
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # メタデータの解析
        doc_metadata = {}
        if metadata:
            import json
            doc_metadata = json.loads(metadata)
        
        # ドキュメント処理
        doc_id = await ingest_service.process_document(temp_file_path, doc_metadata)
        
        # 一時ファイルを削除
        os.unlink(temp_file_path)
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "status": "uploaded",
            "message": "Document uploaded and processing started"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query", response_model=RAGResponse)
async def rag_query(query: RAGQuery):
    """RAG検索エンドポイント"""
    try:
        result = await rag_service.query(
            query_text=query.query,
            user_id=query.user_id,
            filters=query.filters
        )
        
        return RAGResponse(
            query_id=result["query_id"],
            response=result["response"],
            sources=result["sources"],
            processing_time_ms=result["processing_time_ms"]
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity/search")
async def similarity_search(query: SimilarityQuery):
    """類似検索エンドポイント"""
    try:
        results = await similarity_service.search_similar(
            query_text=query.query,
            top_k=query.top_k,
            filters=query.filters,
            threshold=query.threshold
        )
        
        return {
            "query": query.query,
            "results": results,
            "total_count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}/metadata")
async def get_document_metadata(doc_id: str):
    """ドキュメントメタデータ取得エンドポイント"""
    try:
        metadata = await similarity_service.get_document_metadata(doc_id)
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to get metadata for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
