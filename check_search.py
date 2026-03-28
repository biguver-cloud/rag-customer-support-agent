from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
db = Chroma(collection_name="docs", persist_directory="storage/chroma", embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))

results = db.similarity_search_with_score("料金プラン", k=5)
print("=== 検索結果 ===")
for doc, score in results:
    print(f"スコア: {score:.4f} | カテゴリ: {doc.metadata.get('category')} | ソース: {doc.metadata.get('source', '')}")
    print(f"  内容: {doc.page_content[:100]}")
    print()
