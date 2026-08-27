"""
LLM Judgeの安定性検証用：run_eval.py の結果CSVから無作為にサンプルを抽出し、
人手判定を書き込むための空欄付きCSVを作成する。

使い方:
    # 最新の eval/results/*.csv から25件抽出
    python eval/sample_for_human_review.py

    # ファイル・件数・乱数シードを指定
    python eval/sample_for_human_review.py --file eval/results/eval_xxx.csv --n 30 --seed 1
"""
import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

OUTPUT_HEADERS = [
    "id", "category", "method", "question", "expected_answer", "answer",
    "llm_judgment", "llm_judge_reason", "llm_judge_agreement",
    "human_judgment", "human_reason",
]


def _latest_results_csv() -> Path:
    candidates = sorted(RESULTS_DIR.glob("eval_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("eval/results/ に eval_*.csv が見つかりません。先に run_eval.py を実行してください。")
    return candidates[-1]


def _rows_from_result(row: dict) -> list[dict]:
    """1件の評価結果行から vector/hybrid 2件分のレビュー対象行を作る。"""
    out = []
    for method in ("vector", "hybrid"):
        out.append({
            "id": row["id"],
            "category": row["category"],
            "method": method,
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "answer": row[f"{method}_answer"],
            "llm_judgment": row[f"{method}_judge"],
            "llm_judge_reason": row[f"{method}_judge_reason"],
            "llm_judge_agreement": row.get(f"{method}_judge_agreement", ""),
            "human_judgment": "",
            "human_reason": "",
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="対象の eval/results/*.csv（省略時は最新ファイル）")
    parser.add_argument("--n", type=int, default=25, help="抽出件数（デフォルト25）")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（再現性のため固定）")
    args = parser.parse_args()

    src_path = Path(args.file) if args.file else _latest_results_csv()
    print(f"対象ファイル: {src_path}")

    with open(src_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    all_candidates = []
    for row in rows:
        all_candidates.extend(_rows_from_result(row))

    if args.n > len(all_candidates):
        raise RuntimeError(f"抽出件数({args.n})が候補件数({len(all_candidates)})を超えています。")

    random.seed(args.seed)
    sample = random.sample(all_candidates, args.n)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"human_review_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        writer.writerows(sample)

    print(f"{len(sample)}件抽出しました → {out_path}")
    print("human_judgment 列に ○/× を、human_reason 列に理由を記入したうえで")
    print("compare_human_judge.py に渡してください。")


if __name__ == "__main__":
    main()
