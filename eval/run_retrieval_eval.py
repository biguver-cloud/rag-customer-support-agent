"""
検索単体の精度評価スクリプト（Recall@5 / MRR）。

eval/ground_truth_chunks.json（正解チャンク）を使い、ベクトル検索・
ハイブリッド検索それぞれの上位5件に正解チャンクが含まれているかを
LLMを使わずに機械的に判定する。

生成・LLM Judgeを一切使わないため、run_eval.pyよりも大幅に安く・速く・
決定的に実行できる。

使い方:
    python eval/run_retrieval_eval.py
    python eval/run_retrieval_eval.py --rewrite
    python eval/run_retrieval_eval.py --no-janome
    python eval/run_retrieval_eval.py --dataset dataset_colloquial.json
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import MODEL_NAME, TOP_K
from rag.vectorstore import open_vectorstore, hybrid_retrieve_with_score, _vector_only_search
from rag.query import rewrite_query_for_search

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "storage" / "chroma"
DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth_chunks.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

K = 5  # Recall@Kのk

CSV_HEADERS = [
    "id", "category", "question",
    "gt_chunk_ids",
    "vector_hit", "vector_rank", "vector_top_ids",
    "hybrid_hit", "hybrid_rank", "hybrid_top_ids",
]


def build_content_to_id_map(db):
    """チャンク本文 -> chroma id の対応表を作る（検索結果からidを逆引きするため）。"""
    data = db.get(include=["documents"])
    return dict(zip(data["documents"], data["ids"]))


def rank_of_first_hit(retrieved_ids: list[str], gt_ids: set[str]) -> int:
    """正解チャンクが最初に現れた順位（1始まり）を返す。無ければ0。"""
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gt_ids:
            return rank
    return 0


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite", action="store_true", help="クエリリライトを有効にする")
    parser.add_argument("--dataset", type=str, default=None, help="使用するデータセットファイル名（eval/配下）")
    parser.add_argument("--no-janome", action="store_true", help="Janome形態素解析を無効にする")
    args = parser.parse_args()
    use_rewrite = args.rewrite
    use_janome = not args.no_janome

    load_dotenv()

    dataset_path = Path(__file__).resolve().parent / args.dataset if args.dataset else DATASET_PATH

    print("=" * 55)
    print("🎯 検索単体評価：Recall@5 / MRR（LLM不使用）")
    print(f"   クエリリライト: {'あり' if use_rewrite else 'なし'}")
    print(f"   Janome形態素解析: {'あり' if use_janome else 'なし（正規表現）'}")
    print(f"   データセット: {dataset_path.name}")
    print("=" * 55)

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        ground_truth = json.load(f)

    valid = [d for d in dataset if d.get("expected_answer", "").strip() and ground_truth.get(d["id"])]
    print(f"\n評価問題数: {len(valid)} 件（正解チャンク未設定の問題は除外）\n")

    db = open_vectorstore(PERSIST_DIR)
    content_to_id = build_content_to_id_map(db)

    # クエリリライトのみLLMを使う（検索そのものはLLM不使用）
    rewrite_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0) if use_rewrite else None

    RESULTS_DIR.mkdir(exist_ok=True)
    rewrite_label = "_rewrite" if use_rewrite else ""
    dataset_label = f"_{dataset_path.stem}" if args.dataset else ""
    janome_label = "_nojanome" if not use_janome else ""
    results_path = RESULTS_DIR / f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}{dataset_label}{janome_label}{rewrite_label}.csv"

    rows = []
    for i, item in enumerate(valid, 1):
        qid = item["id"]
        question = item["question"]
        category = item.get("category", "unknown")
        gt_ids = {c["chunk_id"] for c in ground_truth[qid]}

        print(f"[{i}/{len(valid)}] {question}")

        search_query = rewrite_query_for_search(question, llm=rewrite_llm) if use_rewrite else question

        vec_results = _vector_only_search(db, search_query, k=K, category=category)
        vec_ids = [content_to_id.get(doc.page_content, "") for doc, _ in vec_results]

        hyb_results = hybrid_retrieve_with_score(db, search_query, k=K, category=category, use_janome=use_janome)
        hyb_ids = [content_to_id.get(doc.page_content, "") for doc, _ in hyb_results]

        vec_rank = rank_of_first_hit(vec_ids, gt_ids)
        hyb_rank = rank_of_first_hit(hyb_ids, gt_ids)

        rows.append({
            "id": qid,
            "category": category,
            "question": question,
            "gt_chunk_ids": ",".join(gt_ids),
            "vector_hit": "○" if vec_rank else "×",
            "vector_rank": vec_rank,
            "vector_top_ids": ",".join(vec_ids),
            "hybrid_hit": "○" if hyb_rank else "×",
            "hybrid_rank": hyb_rank,
            "hybrid_top_ids": ",".join(hyb_ids),
        })

    with open(results_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    vec_recall = sum(1 for r in rows if r["vector_hit"] == "○") / n
    hyb_recall = sum(1 for r in rows if r["hybrid_hit"] == "○") / n
    vec_mrr = sum(1 / r["vector_rank"] for r in rows if r["vector_rank"]) / n
    hyb_mrr = sum(1 / r["hybrid_rank"] for r in rows if r["hybrid_rank"]) / n

    print("=" * 55)
    print("🎯 検索単体評価サマリー")
    print("=" * 55)
    print(f"{'手法':<16} {'Recall@5':<12} {'MRR'}")
    print(f"{'ベクトル検索':<16} {f'{vec_recall:.1%}':<12} {vec_mrr:.3f}")
    print(f"{'ハイブリッド検索':<16} {f'{hyb_recall:.1%}':<12} {hyb_mrr:.3f}")
    print(f"\n差分: Recall@5 {hyb_recall - vec_recall:+.1%}  MRR {hyb_mrr - vec_mrr:+.3f}")
    print(f"\n📄 詳細結果: {results_path}")


if __name__ == "__main__":
    run()
