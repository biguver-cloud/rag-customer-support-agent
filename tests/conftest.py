import sys
from pathlib import Path

# プロジェクトルートをパスに追加（rag.query などのインポートを有効にする）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
