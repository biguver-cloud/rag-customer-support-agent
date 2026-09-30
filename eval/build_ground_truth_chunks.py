"""
各質問(eval/dataset.json)に対して、正解チャンク（ChromaDB内のどのチャンクに
答えが書かれているか）を自動ラベリングするスクリプト。

expected_answer と各チャンク本文の文字レベル類似度（difflib）を計算し、
最も一致度の高いチャンクを正解候補とする。LLMは使用しない。

1問につき最大2件までを正解候補として残す（1つの答えが複数チャンクに
分かれているケースを考慮）。

使い方:
    python eval/build_ground_truth_chunks.py
"""
import json
import re
import sys
import difflib
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.vectorstore import open_vectorstore

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "storage" / "chroma"
DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "ground_truth_chunks.json"

TOP_N = 2  # 1問あたり最大何件を正解候補とするか
MIN_SIMILARITY = 0.3  # これ未満の類似度は正解候補としない


def coverage_similarity(expected: str, chunk: str) -> float:
    """
    「期待回答の文字が、チャンクの中にどれだけ含まれているか」を測る。

    difflib.SequenceMatcher.ratio() は2つの文字列の長さの差に弱く、
    短い期待回答と長いチャンク（見出し・key_facts等を含む）を比較すると
    内容が完全一致していても低いスコアになってしまう。
    そこで、一致文字数を「期待回答側の長さ」だけで正規化する。
    """
    a = re.sub(r'\s+', '', expected)
    b = re.sub(r'\s+', '', chunk)
    if not a or not b:
        return 0.0
    matcher = difflib.SequenceMatcher(None, a, b)
    matched_chars = sum(block.size for block in matcher.get_matching_blocks())
    return matched_chars / len(a)


def main():
    load_dotenv()

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    db = open_vectorstore(PERSIST_DIR)
    chunks = db.get(include=["documents", "metadatas"])
    chunk_ids = chunks["ids"]
    chunk_docs = chunks["documents"]
    chunk_metas = chunks["metadatas"]

    print(f"質問数: {len(dataset)} 件 / チャンク数: {len(chunk_ids)} 件")

    result = {}
    low_confidence = []  # スポットチェック候補として、類似度が低いものを記録

    for item in dataset:
        qid = item["id"]
        category = item.get("category", "unknown")
        expected = item["expected_answer"]

        # カテゴリが一致するチャンクのみを候補にする（誤マッチ防止）
        candidates = [
            (cid, doc, meta)
            for cid, doc, meta in zip(chunk_ids, chunk_docs, chunk_metas)
            if meta.get("category") == category
        ]
        if not candidates:
            # カテゴリ不一致の場合は全チャンクを対象にフォールバック
            candidates = list(zip(chunk_ids, chunk_docs, chunk_metas))

        scored = [
            (cid, coverage_similarity(expected, doc), meta)
            for cid, doc, meta in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        top = [
            {
                "chunk_id": cid,
                "similarity": round(sim, 3),
                "source": meta.get("source", ""),
                "page": meta.get("page"),
            }
            for cid, sim, meta in scored[:TOP_N]
            if sim >= MIN_SIMILARITY
        ]

        result[qid] = top

        if not top or top[0]["similarity"] < 0.3:
            low_confidence.append((qid, top[0]["similarity"] if top else 0.0))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 正解チャンクのラベリング完了 → {OUTPUT_PATH}")
    print(f"\n類似度が低く要確認の質問: {len(low_confidence)} 件")
    for qid, sim in sorted(low_confidence, key=lambda x: x[1])[:20]:
        print(f"  {qid}: 最高類似度 {sim:.3f}")


if __name__ == "__main__":
    main()
