"""
システムセットアップチェッカー
必要な依存関係とサービスが正しく設定されているかチェックします
"""

import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Pythonバージョンをチェック"""
    print("\n=== Pythonバージョン ===")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9以上が必要です")
        return False
    else:
        print("✅ Pythonバージョン OK")
        return True


def check_package(package_name, import_name=None):
    """パッケージがインストールされているかチェック"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} がインストールされていません")
        return False


def check_python_packages():
    """必要なPythonパッケージをチェック"""
    print("\n=== Pythonパッケージ ===")
    
    packages = [
        ("llama-index", "llama_index"),
        ("llama-index-core", "llama_index.core"),
        ("llama-index-llms-ollama", "llama_index.llms.ollama"),
        ("llama-index-embeddings-ollama", "llama_index.embeddings.ollama"),
        ("pymilvus", "pymilvus"),
        ("pymongo", "pymongo"),
        ("redis", "redis"),
        ("neo4j", "neo4j"),
        ("pyyaml", "yaml"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
    ]
    
    results = []
    for pkg_name, import_name in packages:
        results.append(check_package(pkg_name, import_name))
    
    return all(results)


def check_service(name, host, port):
    """サービスが起動しているかチェック"""
    import socket
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ {name} ({host}:{port})")
            return True
        else:
            print(f"❌ {name} ({host}:{port}) - 接続できません")
            return False
    except Exception as e:
        print(f"❌ {name} ({host}:{port}) - エラー: {e}")
        return False


def check_databases():
    """データベースサービスをチェック"""
    print("\n=== データベースサービス（オプション） ===")
    
    services = [
        ("Milvus", "localhost", 19530),
        ("MongoDB", "localhost", 27017),
        ("Redis", "localhost", 6379),
        ("Neo4j", "localhost", 7687),
    ]
    
    results = []
    for name, host, port in services:
        results.append(check_service(name, host, port))
    
    if not any(results):
        print("\n⚠️  データベースが起動していません")
        print("   docker-compose up -d で起動できます")
        print("   または、StorageContextを使用しない設定で実行できます")
    
    return results


def check_ollama():
    """Ollamaサービスをチェック"""
    print("\n=== Ollama（オプション） ===")
    
    # Ollamaサーバーチェック
    if not check_service("Ollama Server", "localhost", 11434):
        print("   ollama serve で起動してください")
        return False
    
    # Ollamaモデルチェック
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("\n利用可能なOllamaモデル:")
            print(result.stdout)
            
            # 推奨モデルのチェック
            models = result.stdout.lower()
            if "qwen3" in models or "qwen2" in models:
                print("✅ 推奨モデルがインストールされています")
                return True
            else:
                print("⚠️  推奨モデル（qwen3:32b, qwen3-embedding:8b）がインストールされていません")
                print("   ollama pull qwen3:32b")
                print("   ollama pull qwen3-embedding:8b")
                return False
        else:
            print("❌ Ollamaコマンドの実行に失敗しました")
            return False
            
    except FileNotFoundError:
        print("❌ Ollamaコマンドが見つかりません")
        print("   https://ollama.ai/ からインストールしてください")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def check_directories():
    """必要なディレクトリをチェック"""
    print("\n=== ディレクトリ構造 ===")
    
    directories = [
        "config",
        "src",
        "data",
        "results",
        "tests",
        "examples"
    ]
    
    all_exist = True
    for dir_name in directories:
        path = Path(dir_name)
        if path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ が存在しません")
            all_exist = False
    
    return all_exist


def check_config_files():
    """設定ファイルをチェック"""
    print("\n=== 設定ファイル ===")
    
    config_files = [
        "config/chunking_configs.yaml",
        "config/embedding_configs.yaml",
        "config/llm_configs.yaml",
        "config/evaluation_configs.yaml",
        "config/test_patterns.yaml"
    ]
    
    all_exist = True
    for file_path in config_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} が存在しません")
            all_exist = False
    
    return all_exist


def main():
    """メインチェック処理"""
    print("="*60)
    print("RAG評価フレームワーク セットアップチェッカー")
    print("="*60)
    
    results = {}
    
    # 各種チェック実行
    results['python'] = check_python_version()
    results['packages'] = check_python_packages()
    results['directories'] = check_directories()
    results['configs'] = check_config_files()
    results['databases'] = check_databases()
    results['ollama'] = check_ollama()
    
    # 結果サマリー
    print("\n" + "="*60)
    print("チェック結果サマリー")
    print("="*60)
    
    essential_checks = ['python', 'packages', 'directories', 'configs']
    optional_checks = ['databases', 'ollama']
    
    essential_ok = all(results[k] for k in essential_checks if k in results)
    
    if essential_ok:
        print("\n✅ 必須要件: すべて満たしています")
        print("   基本的な実験を実行できます")
    else:
        print("\n❌ 必須要件: 一部満たしていません")
        print("   不足している項目をインストール/設定してください")
    
    if any(results[k] for k in optional_checks if k in results):
        print("\n✅ オプション要件: 一部利用可能")
    else:
        print("\n⚠️  オプション要件: 利用できません")
        print("   データベースやOllamaを使用しない設定で実行してください")
    
    # 推奨事項
    print("\n" + "="*60)
    print("次のステップ")
    print("="*60)
    
    if not essential_ok:
        print("\n1. 必須パッケージをインストール:")
        print("   pip install -r requirements.txt")
    else:
        print("\n1. データを準備:")
        print("   mkdir -p data/documents")
        print("   # PDFファイルを data/documents/ に配置")
        
        print("\n2. シンプルな例を実行:")
        print("   python examples/simple_example.py")
        
        print("\n3. メイン実験を実行:")
        print("   python main.py --data data/documents --pattern baseline_001")
    
    if not any(results[k] for k in optional_checks if k in results):
        print("\n（オプション）データベースとOllamaをセットアップ:")
        print("   docker-compose up -d")
        print("   ollama serve")
        print("   ollama pull qwen3:32b")
        print("   ollama pull qwen3-embedding:8b")


if __name__ == "__main__":
    main()
