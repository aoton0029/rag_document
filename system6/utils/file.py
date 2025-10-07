import tempfile
from typing import Dict, Any
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
from typing import Optional
from pydantic import BaseModel

@contextmanager
def temp_file(file_path: Path, suffix: str) -> str:
    """一時ファイルにバイトデータを保存し、そのパスを返す"""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp_file.write(file_path.read_bytes())
        tmp_file.flush()
        tmp_file.close()
        yield tmp_file.name
    finally:
        Path(tmp_file.name).unlink(missing_ok=True)


def pydantic_field_info(model: BaseModel, field_name: str) -> Dict[str, Any]:
    """指定モデルの field_name に対する pydantic Field 情報を抽出します。"""
    mf = model.__pydantic_fields__.get(field_name)
    if mf is None:
        return {}
    fi = mf.field_info
    return {
        "max_length": getattr(fi, "max_length", None),
        "description": fi.description ,
        "alias": fi.alias
    }
