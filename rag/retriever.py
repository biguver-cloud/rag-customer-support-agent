from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def get_retriever(
    base_dir: Path,
    collection_name: str = "lumidesk_docs",
    k: int = 3,
):
    """
    storage/chroma に保存された Chroma を開いて retriever を返す
    """
    persist_dir = base_dir / "storage" / "chroma"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    return vectordb.as_retriever(search_kwargs={"k": k})
