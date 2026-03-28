from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
db = Chroma(collection_name="docs", persist_directory="storage/chroma", embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
all_data = db.get(include=["metadatas"])
sources = set(m.get("source", "") for m in all_data["metadatas"])
print("インデックス済みファイル:")
for s in sorted(sources):
    print(" ", s)
print("総チャンク数:", len(all_data["metadatas"]))
