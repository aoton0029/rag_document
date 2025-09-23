import os
from typing import List, Dict, Any
import logging
from llama_index.core import Document
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.readers.file import DocxReader, PDFReader, EpubReader, ImageReader, PandasExcelReader, VideoAudioReader, MarkdownReader
from llama_index.readers.database import DatabaseReader
from llama_index.readers.whisper import WhisperReader
from llama_index.readers.json import JSONReader
from llama_index.readers.obsidian import ObsidianReader
from llama_index.core.readers.string_iterable import StringIterableReader

class DocumentLoader:
    def __init__(self):        
        self.logger = logging.getLogger(__name__)

    def load(self, source: str, source_type: str = "auto", **kwargs) -> List[Document]:
        """
        様々なソースからドキュメントを読み込み、List[Document]に変換する
        
        Args:
            source: ファイルパス、ディレクトリパス、またはデータソース
            source_type: ソースタイプ ("auto", "directory", "docx", "pdf", "epub", "image", 
                        "excel", "video_audio", "markdown", "database", "whisper", 
                        "json", "obsidian", "string")
            **kwargs: 各Readerに渡す追加パラメータ
        
        Returns:
            List[Document]: 読み込まれたドキュメントのリスト
        """
        try:
            if source_type == "auto":
                return self._auto_detect_and_load(source, **kwargs)
            elif source_type == "directory":
                return self._load_from_directory(source, **kwargs)
            elif source_type == "docx":
                return self._load_docx(source, **kwargs)
            elif source_type == "pdf":
                return self._load_pdf(source, **kwargs)
            elif source_type == "epub":
                return self._load_epub(source, **kwargs)
            elif source_type == "image":
                return self._load_image(source, **kwargs)
            elif source_type == "excel":
                return self._load_excel(source, **kwargs)
            elif source_type == "video_audio":
                return self._load_video_audio(source, **kwargs)
            elif source_type == "markdown":
                return self._load_markdown(source, **kwargs)
            elif source_type == "database":
                return self._load_database(source, **kwargs)
            elif source_type == "whisper":
                return self._load_whisper(source, **kwargs)
            elif source_type == "json":
                return self._load_json(source, **kwargs)
            elif source_type == "obsidian":
                return self._load_obsidian(source, **kwargs)
            elif source_type == "string":
                return self._load_string(source, **kwargs)
            else:
                raise ValueError(f"Unsupported source_type: {source_type}")
                
        except Exception as e:
            self.logger.error(f"Error loading documents from {source}: {str(e)}")
            raise

    def _auto_detect_and_load(self, source: str, **kwargs) -> List[Document]:
        """ファイル拡張子から自動的にReaderを選択"""
        if os.path.isdir(source):
            return self._load_from_directory(source, **kwargs)
        
        ext = os.path.splitext(source)[1].lower()
        if ext in ['.docx', '.doc']:
            return self._load_docx(source, **kwargs)
        elif ext == '.pdf':
            return self._load_pdf(source, **kwargs)
        elif ext == '.epub':
            return self._load_epub(source, **kwargs)
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return self._load_image(source, **kwargs)
        elif ext in ['.xlsx', '.xls']:
            return self._load_excel(source, **kwargs)
        elif ext in ['.mp4', '.avi', '.mov', '.mp3', '.wav']:
            return self._load_video_audio(source, **kwargs)
        elif ext in ['.md', '.markdown']:
            return self._load_markdown(source, **kwargs)
        elif ext == '.json':
            return self._load_json(source, **kwargs)
        else:
            # デフォルトはSimpleDirectoryReaderを使用
            return self._load_from_directory(source, **kwargs)

    def _load_from_directory(self, directory_path: str, **kwargs) -> List[Document]:
        """ディレクトリから複数ファイルを読み込み"""
        reader = SimpleDirectoryReader(input_dir=directory_path, **kwargs)
        return reader.load_data()

    def _load_docx(self, file_path: str, **kwargs) -> List[Document]:
        """DOCXファイルを読み込み"""
        reader = DocxReader(**kwargs)
        return reader.load_data(file=file_path)

    def _load_pdf(self, file_path: str, **kwargs) -> List[Document]:
        """PDFファイルを読み込み"""
        reader = PDFReader(**kwargs)
        return reader.load_data(file=file_path)

    def _load_epub(self, file_path: str, **kwargs) -> List[Document]:
        """EPUBファイルを読み込み"""
        reader = EpubReader(**kwargs)
        return reader.load_data(file=file_path)

    def _load_image(self, file_path: str, **kwargs) -> List[Document]:
        """画像ファイルを読み込み"""
        reader = ImageReader(**kwargs)
        return reader.load_data(file=file_path)

    def _load_excel(self, file_path: str, **kwargs) -> List[Document]:
        """Excelファイルを読み込み"""
        reader = PandasExcelReader(**kwargs)
        return reader.load_data(file=file_path)

    def _load_video_audio(self, file_path: str, **kwargs) -> List[Document]:
        """動画・音声ファイルを読み込み"""
        reader = VideoAudioReader(**kwargs)
        return reader.load_data(file=file_path)

    def _load_markdown(self, file_path: str, **kwargs) -> List[Document]:
        """Markdownファイルを読み込み"""
        reader = MarkdownReader(**kwargs)
        return reader.load_data(file=file_path)

    def _load_database(self, connection_string: str, **kwargs) -> List[Document]:
        """データベースから読み込み"""
        reader = DatabaseReader(**kwargs)
        # データベース接続とクエリの設定が必要
        query = kwargs.get('query', 'SELECT * FROM documents')
        return reader.load_data(query=query)

    def _load_whisper(self, file_path: str, **kwargs) -> List[Document]:
        """Whisperを使用して音声ファイルを読み込み"""
        reader = WhisperReader(**kwargs)
        return reader.load_data(file=file_path)

    def _load_json(self, file_path: str, **kwargs) -> List[Document]:
        """JSONファイルを読み込み"""
        reader = JSONReader(**kwargs)
        return reader.load_data(input_file=file_path)

    def _load_obsidian(self, vault_path: str, **kwargs) -> List[Document]:
        """Obsidianボルトから読み込み"""
        reader = ObsidianReader(input_dir=vault_path, **kwargs)
        return reader.load_data()

    def _load_string(self, strings: List[str], **kwargs) -> List[Document]:
        """文字列リストからドキュメントを作成"""
        reader = StringIterableReader(**kwargs)
        return reader.load_data(texts=strings)


    def load_directory_explicit(self, 
                               input_dir: str,
                               input_files: List[str] = None,
                               exclude: List[str] = None,
                               exclude_hidden: bool = True,
                               errors: str = 'ignore',
                               recursive: bool = True,
                               encoding: str = 'utf-8',
                               filename_as_id: bool = False,
                               required_exts: List[str] = None,
                               file_extractor: Dict = None,
                               num_files_limit: int = None,
                               file_metadata: callable = None) -> List[Document]:
        """ディレクトリから明示的なパラメータでファイルを読み込み"""
        reader = SimpleDirectoryReader(
            input_dir=input_dir,
            input_files=input_files,
            exclude=exclude,
            exclude_hidden=exclude_hidden,
            errors=errors,
            recursive=recursive,
            encoding=encoding,
            filename_as_id=filename_as_id,
            required_exts=required_exts,
            file_extractor=file_extractor,
            num_files_limit=num_files_limit,
            file_metadata=file_metadata
        )
        return reader.load_data()

    def load_docx_explicit(self, 
                          file_path: str,
                          extra_info: Dict = None) -> List[Document]:
        """DOCXファイルを明示的なパラメータで読み込み"""
        reader = DocxReader()
        documents = reader.load_data(file=file_path)
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
        
        return documents

    def load_pdf_explicit(self, 
                         file_path: str,
                         extra_info: Dict = None,
                         return_full_document: bool = False) -> List[Document]:
        """PDFファイルを明示的なパラメータで読み込み"""
        reader = PDFReader(return_full_document=return_full_document)
        documents = reader.load_data(file=file_path)
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
        
        return documents

    def load_epub_explicit(self, 
                          file_path: str,
                          extra_info: Dict = None) -> List[Document]:
        """EPUBファイルを明示的なパラメータで読み込み"""
        reader = EpubReader()
        documents = reader.load_data(file=file_path)
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
        
        return documents

    def load_image_explicit(self, 
                           file_path: str,
                           extra_info: Dict = None,
                           text_type: str = "text",
                           parse_text: bool = False,
                           keep_image: bool = False) -> List[Document]:
        """画像ファイルを明示的なパラメータで読み込み"""
        reader = ImageReader(
            text_type=text_type,
            parse_text=parse_text,
            keep_image=keep_image
        )
        documents = reader.load_data(file=file_path)
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
        
        return documents

    def load_excel_explicit(self, 
                           file_path: str,
                           sheet_name: str = None,
                           pandas_config: Dict = None,
                           extra_info: Dict = None,
                           concat_rows: bool = True,
                           row_joiner: str = "\n") -> List[Document]:
        """Excelファイルを明示的なパラメータで読み込み"""
        reader = PandasExcelReader(
            concat_rows=concat_rows,
            row_joiner=row_joiner,
            pandas_config=pandas_config or {}
        )
        documents = reader.load_data(file=file_path, sheet_name=sheet_name)
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
        
        return documents

    def load_video_audio_explicit(self, 
                                 file_path: str,
                                 model_version: str = "base",
                                 extra_info: Dict = None) -> List[Document]:
        """動画・音声ファイルを明示的なパラメータで読み込み"""
        reader = VideoAudioReader(model_version=model_version)
        documents = reader.load_data(file=file_path)
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
        
        return documents

    def load_markdown_explicit(self, 
                              file_path: str,
                              remove_hyperlinks: bool = True,
                              remove_images: bool = True,
                              extra_info: Dict = None) -> List[Document]:
        """Markdownファイルを明示的なパラメータで読み込み"""
        reader = MarkdownReader(
            remove_hyperlinks=remove_hyperlinks,
            remove_images=remove_images
        )
        documents = reader.load_data(file=file_path)
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
        
        return documents

    def load_database_explicit(self, 
                              query: str,
                              uri: str = None,
                              scheme: str = None,
                              host: str = None,
                              port: int = None,
                              user: str = None,
                              password: str = None,
                              dbname: str = None,
                              extra_info: Dict = None) -> List[Document]:
        """データベースから明示的なパラメータで読み込み"""
        reader = DatabaseReader(
            uri=uri,
            scheme=scheme,
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname
        )
        documents = reader.load_data(query=query)
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
        
        return documents

    def load_whisper_explicit(self, 
                             file_path: str,
                             model_name: str = "base",
                             language: str = None,
                             temperature: float = 0,
                             extra_info: Dict = None) -> List[Document]:
        """Whisperを使用して音声ファイルを明示的なパラメータで読み込み"""
        reader = WhisperReader(
            model_name=model_name,
            language=language,
            temperature=temperature
        )
        documents = reader.load_data(file=file_path)
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
        
        return documents

    def load_json_explicit(self, 
                          file_path: str,
                          levels_back: int = 0,
                          collapse_length: int = None,
                          ensure_ascii: bool = False,
                          is_jsonl: bool = False,
                          clean_json: bool = True,
                          extra_info: Dict = None) -> List[Document]:
        """JSONファイルを明示的なパラメータで読み込み"""
        reader = JSONReader(
            levels_back=levels_back,
            collapse_length=collapse_length,
            ensure_ascii=ensure_ascii,
            is_jsonl=is_jsonl,
            clean_json=clean_json
        )
        documents = reader.load_data(input_file=file_path)
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
        
        return documents

    def load_obsidian_explicit(self, 
                              input_dir: str,
                              encoding: str = "UTF-8",
                              extra_info: Dict = None) -> List[Document]:
        """Obsidianボルトから明示的なパラメータで読み込み"""
        reader = ObsidianReader(
            input_dir=input_dir,
            encoding=encoding
        )
        documents = reader.load_data()
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
        
        return documents

    def load_string_explicit(self, 
                            texts: List[str],
                            extra_info: Dict = None,
                            separator: str = "\n\n") -> List[Document]:
        """文字列リストから明示的なパラメータでドキュメントを作成"""
        reader = StringIterableReader()
        documents = reader.load_data(texts=texts)
        
        if extra_info:
            for doc in documents:
                doc.metadata.update(extra_info)
                
        # テキストを指定のセパレータで結合
        if len(documents) > 1 and separator:
            combined_text = separator.join([doc.text for doc in documents])
            combined_doc = Document(
                text=combined_text,
                metadata=documents[0].metadata if documents else {}
            )
            return [combined_doc]
        
        return documents
    