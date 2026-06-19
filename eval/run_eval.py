"""
ベクトル検索 vs ハイブリッド検索 の精度評価スクリプト。

dataset.json の各質問に対して両方の検索方法で回答を生成し、
② LLM as a Judge（○/×）と ③ 文字類似度 で採点して CSV に出力する。

使い方:
    # 1. 先にデータセットを生成（初回のみ）
    python eval/generate_dataset.py

    # 2. 評価を実行
    python eval/run_eval.py
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import MODEL_NAME, TEMPERATURE, TOP_K, AGENT_ROUNDS
from rag.vectorstore import open_vectorstore, hybrid_retrieve_with_score, _vector_only_search
from rag.agent import agent_answer
from eval.metrics import text_similarity, llm_judge

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "storage" / "chroma"
DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CSV_HEADERS = [
    "id", "category", "question", "expected_answer",
    "vector_answer", "hybrid_answer",
    "vector_judge", "hybrid_judge",
    "vector_similarity", "hybrid_similarity",
    "vector_judge_reason", "hybrid_judge_reason",
]


def _generate_answer(search_results: list, question: str, llm) -> str:
    """検索結果からRAG回答を生成する。"""
    if not search_results:
        return "資料に記載がありません。"
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in search_results)
    result = agent_answer(llm, question, context, rounds=AGENT_ROUNDS)
    return result["answer"]


def run():
    load_dotenv()

    print("=" * 55)
    print("📊 RAG 精度評価：ベクトル検索 vs ハイブリッド検索")
    print("=" * 55)

    # データセット読み込み
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    # expected_answer が空のものを除外
    valid = [d for d in dataset if d.get("expected_answer", "").strip()]
    if not valid:
        print("\n⚠️  dataset.json に expected_answer が設定されていません。")
        print("   先に generate_dataset.py を実行してください：")
        print("   python eval/generate_dataset.py\n")
        return

    print(f"\n評価問題数: {len(valid)} 件\n")

    db = open_vectorstore(PERSIST_DIR)
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    rows = []

    for i, item in enumerate(valid, 1):
        qid      = item["id"]
        question = item["question"]
        expected = item["expected_answer"]
        category = item.get("category", "unknown")

        print(f"[{i}/{len(valid)}] {question}")

        # ── ベクトル検索 ──────────────────────────────────
        print("  🔍 ベクトル検索...")
        vec_results = _vector_only_search(db, question, k=TOP_K, category=category)
        vec_answer  = _generate_answer(vec_results, question, llm)

        # ── ハイブリッド検索 ──────────────────────────────
        print("  🔍 ハイブリッド検索...")
        hyb_results = hybrid_retrieve_with_score(db, question, k=TOP_K, category=category)
        hyb_answer  = _generate_answer(hyb_results, question, llm)

        # ── ② LLM as a Judge ─────────────────────────────
        print("  🤖 LLM評価...")
        vec_judge = llm_judge(question, expected, vec_answer, llm)
        hyb_judge = llm_judge(question, expected, hyb_answer, llm)

        # ── ③ 文字類似度 ──────────────────────────────────
        vec_sim = text_similarity(expected, vec_answer)
        hyb_sim = text_similarity(expected, hyb_answer)

        print(f"  ベクトル   : {vec_judge['judgment']}  類似度 {vec_sim:.1%}")
        print(f"  ハイブリッド: {hyb_judge['judgment']}  類似度 {hyb_sim:.1%}\n")

        rows.append({
            "id":                   qid,
            "category":             category,
            "question":             question,
            "expected_answer":      expected,
            "vector_answer":        vec_answer,
            "hybrid_answer":        hyb_answer,
            "vector_judge":         vec_judge["judgment"],
            "hybrid_judge":         hyb_judge["judgment"],
            "vector_similarity":    f"{vec_sim:.3f}",
            "hybrid_similarity":    f"{hyb_sim:.3f}",
            "vector_judge_reason":  vec_judge["reason"],
            "hybrid_judge_reason":  hyb_judge["reason"],
        })

    # ── CSV 出力 ───────────────────────────────────────────
    with open(results_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    # ── サマリー ───────────────────────────────────────────
    n = len(rows)
    vec_accuracy = sum(1 for r in rows if r["vector_judge"] == "○") / n
    hyb_accuracy = sum(1 for r in rows if r["hybrid_judge"] == "○") / n
    vec_avg_sim  = sum(float(r["vector_similarity"]) for r in rows) / n
    hyb_avg_sim  = sum(float(r["hybrid_similarity"]) for r in rows) / n

    print("=" * 55)
    print("📊 評価結果サマリー")
    print("=" * 55)
    print(f"{'手法':<16} {'正解率(LLM judge)':<20} {'平均類似度'}")
    print(f"{'ベクトル検索':<16} {f'{vec_accuracy:.1%}':<20} {vec_avg_sim:.1%}")
    print(f"{'ハイブリッド検索':<16} {f'{hyb_accuracy:.1%}':<20} {hyb_avg_sim:.1%}")

    diff_acc = hyb_accuracy - vec_accuracy
    diff_sim = hyb_avg_sim - vec_avg_sim
    if diff_acc > 0:
        verdict = "ハイブリッド検索が優位"
    elif diff_acc < 0:
        verdict = "ベクトル検索が優位"
    elif diff_sim > 0:
        verdict = "正解率同率・類似度でハイブリッドがわずかに優位"
    elif diff_sim < 0:
        verdict = "正解率同率・類似度でベクトルがわずかに優位"
    else:
        verdict = "両者同等"
    print(f"\n差分: 正解率 {diff_acc:+.1%}  類似度 {diff_sim:+.1%}  → {verdict}")
    print(f"\n📄 詳細結果: {results_path}")


if __name__ == "__main__":
    run()
