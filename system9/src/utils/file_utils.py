"""
ファイル操作ユーティリティ
PDFファイルやその他のドキュメント処理
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
import hashlib
import json
from datetime import datetime
import mimetypes


class FileManager:
    """ファイル管理クラス"""
    
    def __init__(self, base_path: Union[str, Path] = "."):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def ensure_dir(self, path: Union[str, Path]) -> Path:
        """ディレクトリの存在を確認し、なければ作成"""
        dir_path = Path(path)
        if not dir_path.is_absolute():
            dir_path = self.base_path / dir_path
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def list_files(self, directory: Union[str, Path], 
                   extensions: Optional[List[str]] = None,
                   recursive: bool = True) -> List[Path]:
        """指定ディレクトリ内のファイル一覧を取得"""
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.base_path / dir_path
        
        if not dir_path.exists():
            return []
        
        files = []
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                if extensions is None or file_path.suffix.lower() in extensions:
                    files.append(file_path)
        
        return sorted(files)
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """ファイル情報を取得"""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_path / path
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        stat = path.stat()
        
        return {
            "path": str(path),
            "name": path.name,
            "stem": path.stem,
            "suffix": path.suffix,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "mime_type": mimetypes.guess_type(str(path))[0],
            "hash": self.calculate_file_hash(path)
        }
    
    def calculate_file_hash(self, file_path: Union[str, Path], 
                           algorithm: str = "md5") -> str:
        """ファイルのハッシュ値を計算"""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_path / path
        
        hash_func = getattr(hashlib, algorithm)()
        
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def copy_file(self, src: Union[str, Path], 
                  dst: Union[str, Path],
                  create_dirs: bool = True) -> Path:
        """ファイルをコピー"""
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.is_absolute():
            src_path = self.base_path / src_path
        if not dst_path.is_absolute():
            dst_path = self.base_path / dst_path
        
        if create_dirs:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src_path, dst_path)
        return dst_path
    
    def move_file(self, src: Union[str, Path], 
                  dst: Union[str, Path],
                  create_dirs: bool = True) -> Path:
        """ファイルを移動"""
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.is_absolute():
            src_path = self.base_path / src_path
        if not dst_path.is_absolute():
            dst_path = self.base_path / dst_path
        
        if create_dirs:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(src_path), str(dst_path))
        return dst_path
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """ファイルを削除"""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_path / path
        
        try:
            if path.exists():
                path.unlink()
                return True
        except Exception:
            pass
        return False
    
    def cleanup_empty_dirs(self, directory: Union[str, Path]) -> int:
        """空のディレクトリを削除"""
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.base_path / dir_path
        
        deleted_count = 0
        
        # 深い階層から順に処理
        for root, dirs, files in os.walk(str(dir_path), topdown=False):
            root_path = Path(root)
            if not files and not dirs:
                try:
                    root_path.rmdir()
                    deleted_count += 1
                except OSError:
                    pass
        
        return deleted_count
    
    def batch_rename(self, directory: Union[str, Path],
                     pattern: str,
                     replacement: str) -> List[Dict[str, str]]:
        """ファイルの一括リネーム"""
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.base_path / dir_path
        
        renamed_files = []
        
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                old_name = file_path.name
                new_name = old_name.replace(pattern, replacement)
                
                if old_name != new_name:
                    new_path = file_path.parent / new_name
                    file_path.rename(new_path)
                    renamed_files.append({
                        "old_name": old_name,
                        "new_name": new_name,
                        "path": str(new_path)
                    })
        
        return renamed_files


class DocumentProcessor:
    """ドキュメント処理クラス"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.json': 'application/json',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """サポートされているファイル形式かチェック"""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_formats
    
    def extract_text_from_file(self, file_path: Union[str, Path]) -> str:
        """ファイルからテキストを抽出（基本実装）"""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.txt' or suffix == '.md':
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            raise NotImplementedError(f"Text extraction not implemented for {suffix}")
    
    def validate_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """ドキュメントの妥当性をチェック"""
        path = Path(file_path)
        
        if not path.exists():
            return {"valid": False, "error": "File not found"}
        
        if not self.is_supported_format(path):
            return {"valid": False, "error": f"Unsupported format: {path.suffix}"}
        
        try:
            # 基本的な読み込みテスト
            if path.suffix.lower() in ['.txt', '.md']:
                with open(path, 'r', encoding='utf-8') as f:
                    f.read(100)  # 最初の100文字だけ読み取りテスト
            
            return {"valid": True, "format": path.suffix.lower()}
        
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def get_document_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """ドキュメントのメタデータを取得"""
        file_manager = FileManager()
        file_info = file_manager.get_file_info(file_path)
        
        # ドキュメント固有の情報を追加
        path = Path(file_path)
        
        if path.suffix.lower() in ['.txt', '.md']:
            try:
                content = self.extract_text_from_file(file_path)
                file_info.update({
                    "character_count": len(content),
                    "word_count": len(content.split()),
                    "line_count": content.count('\n') + 1
                })
            except Exception:
                pass
        
        return file_info


def batch_process_files(directory: Union[str, Path],
                       processor_func: callable,
                       extensions: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
    """ファイルの一括処理"""
    file_manager = FileManager()
    doc_processor = DocumentProcessor()
    
    files = file_manager.list_files(directory, extensions)
    
    for file_path in files:
        try:
            if doc_processor.is_supported_format(file_path):
                result = processor_func(file_path)
                yield {
                    "file_path": str(file_path),
                    "success": True,
                    "result": result
                }
            else:
                yield {
                    "file_path": str(file_path),
                    "success": False,
                    "error": "Unsupported format"
                }
        except Exception as e:
            yield {
                "file_path": str(file_path),
                "success": False,
                "error": str(e)
            }