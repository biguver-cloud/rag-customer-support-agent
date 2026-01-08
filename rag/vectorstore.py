import re
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

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
