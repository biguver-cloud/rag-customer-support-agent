"""
sample_for_human_review.py で作成し、human_judgment 列を人手で埋めたCSVを読み込み、
LLM Judgeの判定と人手判定の一致率・Cohen's kappa を集計する。

使い方:
    python eval/compare_human_judge.py --file eval/results/human_review_sample_xxx.csv
"""
import argparse
import csv
from pathlib import Path


def _cohens_kappa(llm_labels: list[str], human_labels: list[str]) -> float:
    """2値（○/×）のCohen's kappaを外部ライブラリなしで計算する。"""
    n = len(llm_labels)
    if n == 0:
        return 0.0

    po = sum(1 for l, h in zip(llm_labels, human_labels) if l == h) / n

    llm_maru = sum(1 for l in llm_labels if l == "○") / n
    llm_batsu = 1 - llm_maru
    human_maru = sum(1 for h in human_labels if h == "○") / n
    human_batsu = 1 - human_maru
    pe = (llm_maru * human_maru) + (llm_batsu * human_batsu)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="human_judgment記入済みのCSVファイル")
    args = parser.parse_args()

    path = Path(args.file)
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    unfilled = [r for r in rows if r["human_judgment"].strip() not in ("○", "×")]
    if unfilled:
        print(f"⚠️  human_judgment が未記入または不正な行が{len(unfilled)}件あります（○ か × のみ有効）。")
        rows = [r for r in rows if r["human_judgment"].strip() in ("○", "×")]

    if not rows:
        print("集計対象の行がありません。human_judgment 列を ○/× で埋めてください。")
        return

    llm_labels = [r["llm_judgment"].strip() for r in rows]
    human_labels = [r["human_judgment"].strip() for r in rows]

    n = len(rows)
    agree = sum(1 for l, h in zip(llm_labels, human_labels) if l == h)
    kappa = _cohens_kappa(llm_labels, human_labels)

    # 混同行列
    tt = sum(1 for l, h in zip(llm_labels, human_labels) if l == "○" and h == "○")
    tf = sum(1 for l, h in zip(llm_labels, human_labels) if l == "○" and h == "×")
    ft = sum(1 for l, h in zip(llm_labels, human_labels) if l == "×" and h == "○")
    ff = sum(1 for l, h in zip(llm_labels, human_labels) if l == "×" and h == "×")

    print("=" * 50)
    print("📊 LLM Judge vs 人手判定 一致率")
    print("=" * 50)
    print(f"対象件数: {n}")
    print(f"一致率  : {agree}/{n} ({agree / n:.1%})")
    print(f"Cohen's kappa: {kappa:.3f}")
    print()
    print("混同行列（行=LLM Judge, 列=人手判定）")
    print(f"{'':<12}{'人手○':<10}{'人手×':<10}")
    print(f"{'LLM ○':<12}{tt:<10}{tf:<10}")
    print(f"{'LLM ×':<12}{ft:<10}{ff:<10}")

    disagreements = [r for l, h, r in zip(llm_labels, human_labels, rows) if l != h]
    if disagreements:
        print(f"\n不一致 {len(disagreements)}件:")
        for r in disagreements:
            print(f"  [{r['id']}/{r['method']}] LLM={r['llm_judgment']} 人手={r['human_judgment']}  Q: {r['question'][:40]}")


if __name__ == "__main__":
    main()
