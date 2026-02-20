from pathlib import Path
from typing import List, Tuple, Dict, Optional

from langchain_openai import OpenAIEmbeddings
from config import RETRIEVER_K, RETRIEVER_K_DEFAULT
from langchain_chroma import Chroma
from langchain_core.documents import Document


def get_retriever(
    base_dir: Path,
    collection_name: str = "lumidesk_docs",
    k: int = RETRIEVER_K,  # config.py から読み込み
):
    """
    storage/chroma に保存された Chroma を開いて retriever を返す
    
    Args:
        base_dir: プロジェクトのベースディレクトリ
        collection_name: Chromaのコレクション名
        k: 取得するドキュメント数
        
    Returns:
        Chromaのretrieverオブジェクト
    """
    persist_dir = base_dir / "storage" / "chroma"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    return vectordb.as_retriever(search_kwargs={"k": k})


def get_vectorstore(
    persist_dir: Path,
    collection_name: str = "docs",
) -> Chroma:
    """
    Chromaベクトルストアを開く
    
    Args:
        persist_dir: Chromaの永続化ディレクトリ
        collection_name: コレクション名
        
    Returns:
        Chromaベクトルストアオブジェクト
        
    Raises:
        Exception: ベクトルストアのロードに失敗した場合
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectordb = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        return vectordb
    except Exception as e:
        raise Exception(f"Chromaベクトルストアの読み込みに失敗しました: {str(e)}")


def retrieve_documents_with_score(
    vectorstore: Chroma,
    query: str,
    k: int = RETRIEVER_K_DEFAULT,  # config.py から読み込み
    category: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """
    ベクトルストアから類似度スコア付きでドキュメントを検索する
    
    Args:
        vectorstore: Chromaベクトルストアオブジェクト
        query: 検索クエリ
        k: 取得するドキュメント数（デフォルト: 4）
        category: カテゴリフィルタ（Noneまたは"unknown"の場合はフィルタなし）
        
    Returns:
        List[Tuple[Document, float]]: (ドキュメント, スコア)のタプルのリスト
        スコアは類似度を表すfloat値（Chromaの場合、値が小さいほど類似度が高い）
        検索結果がない場合は空のリストを返す
        
    Raises:
        ValueError: queryが空文字列の場合
        Exception: Chroma検索中にエラーが発生した場合
        
    Examples:
        >>> results = retrieve_documents_with_score(db, "解約方法", k=3)
        >>> for doc, score in results:
        ...     print(f"Score: {score}, Content: {doc.page_content[:50]}")
    """
    if not query or not query.strip():
        raise ValueError("検索クエリが空です")
    
    try:
        # カテゴリフィルタの設定
        search_kwargs = {}
        if category and category != "unknown":
            search_kwargs["filter"] = {"category": category}
        
        # スコア付き検索を実行
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            **search_kwargs
        )
        
        return results
        
    except Exception as e:
        raise Exception(f"ドキュメント検索中にエラーが発生しました: {str(e)}")


def retrieve_documents(
    vectorstore: Chroma,
    query: str,
    k: int = RETRIEVER_K_DEFAULT,  # config.py から読み込み
    category: Optional[str] = None,
) -> List[Document]:
    """
    ベクトルストアからドキュメントを検索する（スコアなし）
    
    既存コードとの互換性のための関数。
    スコアが不要な場合に使用。
    
    Args:
        vectorstore: Chromaベクトルストアオブジェクト
        query: 検索クエリ
        k: 取得するドキュメント数（デフォルト: 4）
        category: カテゴリフィルタ（Noneまたは"unknown"の場合はフィルタなし）
        
    Returns:
        List[Document]: ドキュメントのリスト
        検索結果がない場合は空のリストを返す
        
    Raises:
        ValueError: queryが空文字列の場合
        Exception: Chroma検索中にエラーが発生した場合
    """
    results_with_score = retrieve_documents_with_score(
        vectorstore=vectorstore,
        query=query,
        k=k,
        category=category
    )
    
    # スコアを除去してDocumentのみを返す
    return [doc for doc, _ in results_with_score]


def retrieve_with_metadata(
    vectorstore: Chroma,
    query: str,
    k: int = RETRIEVER_K_DEFAULT,  # config.py から読み込み
    category: Optional[str] = None,
) -> List[Dict]:
    """
    ベクトルストアから検索し、ドキュメントとスコアを辞書形式で返す
    
    UI側で扱いやすいように、dict形式で返す。
    
    Args:
        vectorstore: Chromaベクトルストアオブジェクト
        query: 検索クエリ
        k: 取得するドキュメント数（デフォルト: 4）
        category: カテゴリフィルタ（Noneまたは"unknown"の場合はフィルタなし）
        
    Returns:
        List[dict]: 以下の形式の辞書のリスト
        [
            {
                "document": Document,
                "score": float,
                "content": str,  # page_content
                "metadata": dict,  # source, page, category など
            },
            ...
        ]
        検索結果がない場合は空のリストを返す
        
    Raises:
        ValueError: queryが空文字列の場合
        Exception: Chroma検索中にエラーが発生した場合
        
    Examples:
        >>> results = retrieve_with_metadata(db, "返金条件", k=5, category="service")
        >>> for item in results:
        ...     print(f"Score: {item['score']:.4f}")
        ...     print(f"Source: {item['metadata'].get('source')}")
    """
    results_with_score = retrieve_documents_with_score(
        vectorstore=vectorstore,
        query=query,
        k=k,
        category=category
    )
    
    # dict形式に変換
    formatted_results = []
    for doc, score in results_with_score:
        formatted_results.append({
            "document": doc,
            "score": score,
            "content": doc.page_content,
            "metadata": doc.metadata,
        })
    
    return formatted_results
