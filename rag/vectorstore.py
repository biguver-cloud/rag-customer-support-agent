import re
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def open_vectorstore(persist_dir: Path) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="docs",
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

def retrieve_with_score(db: Chroma, query: str, k: int = 4, category: str = "unknown"):
    kwargs = {}
    if category != "unknown":
        kwargs["filter"] = {"category": category}

    results = db.similarity_search_with_score(query, k=k, **kwargs)
    if not results:
        return "", [], None

    docs = [d for d, _ in results]
    scores = [s for _, s in results]
    best_score = min(scores) if scores else None

    context = "\n\n---\n\n".join([d.page_content for d in docs])

    citations = []
    for d in docs:
        src = d.metadata.get("source", "")
        page = d.metadata.get("page", None)
        cat = d.metadata.get("category", category)
        text = d.page_content.strip().replace("\n", " ")
        quote = text[:400] + ("..." if len(text) > 400 else "")  # メール等が切れないよう長め

        citations.append(
            {"category": cat, "source": src, "page": (page + 1) if isinstance(page, int) else None, "quote": quote}
        )

    return context, citations, best_score


def _vector_only_search(
    db: Chroma,
    query: str,
    k: int,
    category: str,
) -> list[tuple[Document, float]]:
    """ベクトル検索のみで結果を返す（フォールバック用）。"""
    vec_kwargs = {}
    if category and category != "unknown":
        vec_kwargs["filter"] = {"category": category}
    return db.similarity_search_with_score(query, k=k, **vec_kwargs)


def hybrid_retrieve_with_score(
    db: Chroma,
    query: str,
    k: int = 4,
    category: str = "unknown",
    rrf_k: int = 60,
) -> list[tuple[Document, float]]:
    """BM25 + ベクトル検索を RRF で統合するハイブリッド検索。
    BM25 が利用できない場合はベクトル検索のみにフォールバックする。

    Returns:
        (Document, distance) のリスト。distance は小さいほど良い（ベクトル距離ベース）。
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[hybrid_retrieve] rank_bm25 not found, falling back to vector search")
        return _vector_only_search(db, query, k, category)

    try:
        # Chromaから全ドキュメントを取得
        all_data = db.get(include=["documents", "metadatas"])
        all_contents: list[str] = all_data.get("documents") or []
        all_metadatas: list[dict] = all_data.get("metadatas") or []

        if not all_contents:
            return _vector_only_search(db, query, k, category)

        # カテゴリフィルタ（Pythonレベル）
        if category and category != "unknown":
            pairs = [(c, m) for c, m in zip(all_contents, all_metadatas)
                     if m and m.get("category") == category]
            if pairs:
                all_contents, all_metadatas = zip(*pairs)
                all_contents, all_metadatas = list(all_contents), list(all_metadatas)
            else:
                # カテゴリが見つからない場合はフィルタなしで再検索
                return _vector_only_search(db, query, k, category)

        n = len(all_contents)

        # BM25 検索（日本語: Janome 形態素解析によるトークナイズ）
        try:
            from janome.tokenizer import Tokenizer as JanomeTokenizer
            _jt = JanomeTokenizer()
            def _tokenize(text: str) -> list[str]:
                return [t.surface for t in _jt.tokenize(text)]
        except ImportError:
            def _tokenize(text: str) -> list[str]:
                return re.findall(r'\w+', text)

        tokenized_corpus = [_tokenize(doc) for doc in all_contents]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(_tokenize(query))
        bm25_rank = {
            all_contents[idx]: rank
            for rank, idx in enumerate(sorted(range(n), key=lambda i: bm25_scores[i], reverse=True))
        }

        # ベクトル検索
        vec_kwargs = {}
        if category and category != "unknown":
            vec_kwargs["filter"] = {"category": category}
        vector_results = db.similarity_search_with_score(query, k=n, **vec_kwargs)
        vec_data = {doc.page_content: (rank, score) for rank, (doc, score) in enumerate(vector_results)}

        # RRF スコア計算（高いほど良い）
        rrf_scores = {}
        for content in all_contents:
            bm25_r = bm25_rank.get(content, n)
            vec_r = vec_data.get(content, (n, 1.0))[0]
            rrf_scores[content] = 1 / (rrf_k + bm25_r) + 1 / (rrf_k + vec_r)

        # 上位 k 件を取得
        top_contents = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)[:k]

        content_to_meta = dict(zip(all_contents, all_metadatas))
        results = []
        for content in top_contents:
            doc = Document(page_content=content, metadata=content_to_meta.get(content, {}))
            _, vec_dist = vec_data.get(content, (n, 0.15))
            results.append((doc, vec_dist))

        return results

    except Exception as e:
        print(f"[hybrid_retrieve] BM25 error: {e}, falling back to vector search")
        return _vector_only_search(db, query, k, category)
