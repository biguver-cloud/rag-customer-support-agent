# ------------------------------------------------------------
# 1) data/ 配下のPDFを読み込む（サブフォルダも対象）
# 2) 文書を分割してEmbedding
# 3) Chroma(storage/chroma) に保存
# ------------------------------------------------------------
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# ------------------------------------------------------------
# 追加: source からカテゴリ(company/customer/service)を付与する
# ------------------------------------------------------------
def infer_category_from_source(source: str) -> str:
    s = source.replace("\\", "/")
    if "/company/" in s:
        return "company"
    if "/customer/" in s:
        return "customer"
    if "/service/" in s:
        return "service"
    return "unknown"


def main():
    # ------------------------------------------------------------
    # 0) APIキー確認
    # ------------------------------------------------------------
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が .env に設定されていません")

    base_dir = Path(__file__).parent
    pdf_dir = base_dir / "data"
    persist_dir = base_dir / "storage" / "chroma"

    if not pdf_dir.exists():
        raise RuntimeError(f"PDFフォルダがありません: {pdf_dir}")

    # ------------------------------------------------------------
    # 1) PDF読み込み（data/配下をまとめて読む）
    # ------------------------------------------------------------
    loader = PyPDFDirectoryLoader(str(pdf_dir))
    docs = loader.load()

    if not docs:
        raise RuntimeError("PDFが1件も読み込めませんでした。data/配下にPDFがあるか確認してください。")

    # docs を読み込んだ直後にカテゴリを付与
    for d in docs:
        src = d.metadata.get("source", "")
        d.metadata["category"] = infer_category_from_source(src)

    # ------------------------------------------------------------
    # 2) 分割
    # ------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    splits = splitter.split_documents(docs)

    # ------------------------------------------------------------
    # 3) Chromaへ保存（混在防止で一度コレクションを消す）
    # ------------------------------------------------------------
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        collection_name="docs",
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    # 同名コレクションに追記されるのを避ける（課題では毎回作り直し推奨）
    try:
        db.delete_collection()
        db = Chroma(
            collection_name="docs",
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
    except Exception:
        pass

    db.add_documents(splits)

    print("インデックス作成完了")
    print(f"読み込みPDF数: {len(docs)}")
    print(f"分割チャンク数: {len(splits)}")
    print(f"保存先: {persist_dir}")


if __name__ == "__main__":
    main()
