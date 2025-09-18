import os
import logging
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
import tempfile
import json
import numpy as np
from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_image
from yomitoku import TextRecognizer
from yomitoku import OCR
from llama_index.core import Document

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """文字の境界ボックス情報"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    text: str


@dataclass
class OCRConfig:
    """OCR設定クラス"""
    # Yomitoku設定
    model_name: str = "yomitoku"  # デフォルトモデル
    device: str = "cpu"  # "cpu" or "cuda"
    batch_size: int = 1
    
    # 画像前処理設定
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    enhance_contrast: bool = True
    denoise: bool = True
    
    # OCR処理設定
    confidence_threshold: float = 0.5
    merge_boxes: bool = True
    preserve_layout: bool = True
    
    # 言語設定
    languages: List[str] = field(default_factory=lambda: ["ja", "en"])
    
    # 出力設定
    include_bounding_boxes: bool = True
    include_confidence: bool = True
    
    # パフォーマンス設定
    max_image_size: int = 4096 * 4096  # 最大画像サイズ
    timeout_seconds: int = 300  # タイムアウト


@dataclass
class OCRResult:
    """OCR結果クラス"""
    success: bool
    text: str = ""
    confidence: float = 0.0
    language_detected: Optional[str] = None
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    processing_time: float = 0.0
    error_message: Optional[str] = None
    
    # メタデータ
    page_count: int = 1
    image_width: int = 0
    image_height: int = 0
    
    # 構造化情報
    lines: List[str] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)


class YomitokuOCR:
    """
    Yomitoku OCRサービス
    画像やPDFからテキストを抽出し、Document形式で返す
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.ocr_engine = None
        self.text_recognizer = None
        self.document_analyzer = None
        
        # Yomitokuエンジンの初期化
        self._initialize_engines()
        
        logger.info(f"YomitokuOCR initialized with config: {self.config}")
    
    def _initialize_engines(self):
        """Yomitokuエンジンを初期化"""
        try:
            # メインOCRエンジン
            self.ocr_engine = OCR(
                device=self.config.device,
                model_name=self.config.model_name
            )
            
            # テキスト認識エンジン
            self.text_recognizer = TextRecognizer(
                device=self.config.device
            )
            
            # ドキュメント解析エンジン
            self.document_analyzer = DocumentAnalyzer(
                device=self.config.device
            )
            
            logger.info("Yomitoku engines initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Yomitoku engines: {e}")
            raise RuntimeError(f"OCR engine initialization failed: {e}")
    
    def extract_text_from_image(
        self, 
        image_path: Union[str, Path, BinaryIO]
    ) -> OCRResult:
        """
        画像からテキストを抽出
        
        Args:
            image_path: 画像ファイルのパスまたはBinaryIO
            
        Returns:
            OCRResult: OCR処理結果
        """
        start_time = datetime.now()
        
        try:
            # 画像を読み込み
            if isinstance(image_path, (str, Path)):
                image = load_image(str(image_path))
                image_path_str = str(image_path)
            else:
                # BinaryIOの場合、一時ファイルに保存
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    tmp_file.write(image_path.read())
                    tmp_file_path = tmp_file.name
                
                image = load_image(tmp_file_path)
                image_path_str = tmp_file_path
                
                # 一時ファイルを削除
                os.unlink(tmp_file_path)
            
            # 画像サイズチェック
            if hasattr(image, 'size'):
                width, height = image.size
                if width * height > self.config.max_image_size:
                    # 画像をリサイズ
                    from PIL import Image
                    scale = (self.config.max_image_size / (width * height)) ** 0.5
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = image.resize((new_width, new_height))
            
            # OCR実行
            result = self._perform_ocr(image, image_path_str)
            
            # 処理時間計算
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            logger.info(f"OCR completed for image: {len(result.text)} characters extracted in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"OCR failed for image: {e}"
            logger.error(error_msg)
            return OCRResult(
                success=False,
                error_message=error_msg,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def extract_text_from_pdf(
        self, 
        pdf_path: Union[str, Path]
    ) -> OCRResult:
        """
        PDFからテキストを抽出（各ページを画像として処理）
        
        Args:
            pdf_path: PDFファイルのパス
            
        Returns:
            OCRResult: OCR処理結果
        """
        start_time = datetime.now()
        
        try:
            import pymupdf as fitz
            
            pdf_path = Path(pdf_path)
            
            # PDFを開く
            doc = fitz.open(str(pdf_path))
            
            all_text = []
            all_bounding_boxes = []
            total_confidence = 0.0
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc[page_num]
                
                # ページを画像として取得
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2倍解像度
                img_data = pix.tobytes("png")
                
                # 一時ファイルに保存
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    tmp_file.write(img_data)
                    tmp_file_path = tmp_file.name
                
                try:
                    # 画像からOCR実行
                    image = load_image(tmp_file_path)
                    page_result = self._perform_ocr(image, f"{pdf_path}#page{page_num+1}")
                    
                    if page_result.success:
                        all_text.append(f"=== Page {page_num + 1} ===\n{page_result.text}")
                        all_bounding_boxes.extend(page_result.bounding_boxes)
                        total_confidence += page_result.confidence
                        
                        logger.debug(f"Page {page_num + 1}: {len(page_result.text)} characters extracted")
                    
                finally:
                    # 一時ファイルを削除
                    os.unlink(tmp_file_path)
            
            doc.close()
            
            # 全体の結果をまとめる
            combined_text = "\n\n".join(all_text)
            average_confidence = total_confidence / page_count if page_count > 0 else 0.0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = OCRResult(
                success=True,
                text=combined_text,
                confidence=average_confidence,
                bounding_boxes=all_bounding_boxes,
                processing_time=processing_time,
                page_count=page_count
            )
            
            logger.info(f"PDF OCR completed: {len(combined_text)} characters from {page_count} pages in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"PDF OCR failed: {e}"
            logger.error(error_msg)
            return OCRResult(
                success=False,
                error_message=error_msg,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _perform_ocr(self, image, source_path: str) -> OCRResult:
        """
        実際のOCR処理を実行
        
        Args:
            image: 処理対象の画像
            source_path: ソースパス（ログ用）
            
        Returns:
            OCRResult: OCR結果
        """
        try:
            # ドキュメント解析でレイアウト検出
            layout_result = self.document_analyzer.analyze(image)
            
            # テキスト認識実行
            ocr_result = self.ocr_engine.recognize(image)
            
            # 結果を解析
            extracted_text = ""
            bounding_boxes = []
            confidence_scores = []
            
            if hasattr(ocr_result, 'text_lines') and ocr_result.text_lines:
                for line in ocr_result.text_lines:
                    if hasattr(line, 'text') and line.text:
                        extracted_text += line.text + "\n"
                        
                        # 境界ボックス情報を取得
                        if hasattr(line, 'bbox') and self.config.include_bounding_boxes:
                            bbox = line.bbox
                            confidence = getattr(line, 'confidence', 0.5)
                            confidence_scores.append(confidence)
                            
                            if confidence >= self.config.confidence_threshold:
                                bounding_boxes.append(BoundingBox(
                                    x1=float(bbox[0]),
                                    y1=float(bbox[1]),
                                    x2=float(bbox[2]),
                                    y2=float(bbox[3]),
                                    confidence=float(confidence),
                                    text=line.text
                                ))
            
            # 代替方法：直接text属性がある場合
            elif hasattr(ocr_result, 'text'):
                extracted_text = ocr_result.text
                if hasattr(ocr_result, 'confidence'):
                    confidence_scores.append(ocr_result.confidence)
            
            # さらなる代替方法：結果を文字列に変換
            else:
                extracted_text = str(ocr_result)
            
            # 平均信頼度計算
            average_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
            
            # 言語検出（簡易版）
            detected_language = self._detect_language(extracted_text)
            
            # 構造化処理
            lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
            paragraphs = self._group_into_paragraphs(lines)
            
            return OCRResult(
                success=True,
                text=extracted_text.strip(),
                confidence=float(average_confidence),
                language_detected=detected_language,
                bounding_boxes=bounding_boxes,
                lines=lines,
                paragraphs=paragraphs,
                image_width=getattr(image, 'width', 0),
                image_height=getattr(image, 'height', 0)
            )
            
        except Exception as e:
            logger.error(f"OCR processing failed for {source_path}: {e}")
            return OCRResult(
                success=False,
                error_message=str(e)
            )
    
    def _detect_language(self, text: str) -> Optional[str]:
        """簡易言語検出"""
        if not text:
            return None
        
        # 日本語文字の検出
        japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u309F' or 
                           '\u30A0' <= char <= '\u30FF' or 
                           '\u4E00' <= char <= '\u9FAF')
        
        # 英語文字の検出
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return None
        
        japanese_ratio = japanese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if japanese_ratio > 0.1:
            return "ja"
        elif english_ratio > 0.5:
            return "en"
        else:
            return "unknown"
    
    def _group_into_paragraphs(self, lines: List[str]) -> List[str]:
        """行を段落にグループ化"""
        if not lines:
            return []
        
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            if not line.strip():
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line.strip())
        
        # 最後の段落を追加
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return paragraphs
    
    def convert_to_documents(
        self, 
        ocr_result: OCRResult, 
        source_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        OCR結果をDocument形式に変換（埋め込みサービス用）
        
        Args:
            ocr_result: OCR処理結果
            source_path: ソースファイルパス
            chunk_size: チャンクサイズ
            chunk_overlap: チャンクオーバーラップ
            
        Returns:
            List[Document]: Document形式のリスト
        """
        if not ocr_result.success or not ocr_result.text:
            return []
        
        documents = []
        
        # 基本メタデータ
        base_metadata = {
            "source_path": source_path,
            "source_type": "ocr",
            "ocr_confidence": ocr_result.confidence,
            "ocr_language": ocr_result.language_detected,
            "ocr_processing_time": ocr_result.processing_time,
            "page_count": ocr_result.page_count,
            "extraction_method": "yomitoku",
            "timestamp": datetime.now().isoformat()
        }
        
        # テキストをチャンクに分割
        text_chunks = self._split_text_into_chunks(
            ocr_result.text, 
            chunk_size, 
            chunk_overlap
        )
        
        for i, chunk in enumerate(text_chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "chunk_size": len(chunk),
                "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()
            })
            
            # 該当する境界ボックス情報を追加
            if ocr_result.bounding_boxes:
                chunk_boxes = self._find_bounding_boxes_for_chunk(
                    chunk, ocr_result.bounding_boxes
                )
                if chunk_boxes:
                    chunk_metadata["bounding_boxes"] = [
                        {
                            "x1": box.x1, "y1": box.y1, 
                            "x2": box.x2, "y2": box.y2,
                            "confidence": box.confidence,
                            "text": box.text
                        } for box in chunk_boxes
                    ]
            
            document = Document(
                text=chunk,
                metadata=chunk_metadata
            )
            documents.append(document)
        
        logger.info(f"Converted OCR result to {len(documents)} documents for {source_path}")
        
        return documents
    
    def _split_text_into_chunks(
        self, 
        text: str, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> List[str]:
        """テキストをチャンクに分割"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 単語境界で分割
            if end < len(text):
                # 改行を優先的に探す
                newline_pos = text.rfind('\n', start, end)
                if newline_pos != -1:
                    end = newline_pos + 1
                else:
                    # スペースを探す
                    space_pos = text.rfind(' ', start, end)
                    if space_pos != -1:
                        end = space_pos + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # オーバーラップを考慮した次の開始位置
            start = end - chunk_overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def _find_bounding_boxes_for_chunk(
        self, 
        chunk: str, 
        all_boxes: List[BoundingBox]
    ) -> List[BoundingBox]:
        """チャンクに対応する境界ボックスを検索"""
        chunk_words = set(chunk.lower().split())
        matching_boxes = []
        
        for box in all_boxes:
            box_words = set(box.text.lower().split())
            if chunk_words & box_words:  # 交集合がある場合
                matching_boxes.append(box)
        
        return matching_boxes
    
    async def extract_text_from_image_async(
        self, 
        image_path: Union[str, Path, BinaryIO]
    ) -> OCRResult:
        """画像からテキストを非同期で抽出"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.extract_text_from_image, 
            image_path
        )
    
    async def extract_text_from_pdf_async(
        self, 
        pdf_path: Union[str, Path]
    ) -> OCRResult:
        """PDFからテキストを非同期で抽出"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.extract_text_from_pdf, 
            pdf_path
        )
    
    def get_engine_info(self) -> Dict[str, Any]:
        """エンジン情報を取得"""
        return {
            "engine": "yomitoku",
            "config": {
                "model_name": self.config.model_name,
                "device": self.config.device,
                "languages": self.config.languages,
                "confidence_threshold": self.config.confidence_threshold
            },
            "status": "ready" if self.ocr_engine else "not_initialized"
        }


# 使用例とテスト用の関数
async def test_yomitoku_ocr():
    """テスト関数"""
    ocr_config = OCRConfig(
        device="cpu",
        confidence_threshold=0.3,
        languages=["ja", "en"]
    )
    
    ocr_service = YomitokuOCR(ocr_config)
    
    # 画像ファイルのテスト
    test_image = "test_image.png"
    if Path(test_image).exists():
        result = await ocr_service.extract_text_from_image_async(test_image)
        print(f"Image OCR result: {result.success}")
        print(f"Extracted text: {result.text[:200]}...")
        
        # Document形式に変換
        documents = ocr_service.convert_to_documents(result, test_image)
        print(f"Generated {len(documents)} documents")
    
    # PDFファイルのテスト
    test_pdf = "test_document.pdf"
    if Path(test_pdf).exists():
        result = await ocr_service.extract_text_from_pdf_async(test_pdf)
        print(f"PDF OCR result: {result.success}")
        print(f"Pages processed: {result.page_count}")
        print(f"Average confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    asyncio.run(test_yomitoku_ocr())

