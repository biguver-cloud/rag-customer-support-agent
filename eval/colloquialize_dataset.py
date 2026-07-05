"""
既存のdataset.jsonの質問文を口語・ノイズ入りに変換するスクリプト。

クリーンな質問文を実際のユーザーが入力するような表現に変換し、
クエリリライトの効果を検証するためのデータセットを生成する。

使い方:
    python eval/colloquialize_dataset.py
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import MODEL_NAME

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "dataset.json"
OUTPUT_PATH = BASE_DIR / "dataset_colloquial.json"


def colloquialize(question: str, llm) -> str:
    prompt = f"""以下の質問文を、実際のユーザーがカスタマーサポートに送るような
口語・ノイズ入りの表現に書き換えてください。

ルール：
- 敬語・丁寧語を使う（〜なんですけど、〜したいんですが、〜でしょうか）
- 質問の意図は変えない（意味は同じに保つ）
- 不必要な前置きや背景を少し加える（例：「急に〜」「先日〜したんですが」）
- 質問文を1〜2文で収める
- 元の質問文よりも曖昧・口語的にする

元の質問：{question}

口語化した質問（1文のみ出力）："""

    try:
        result = llm.invoke([{"role": "user", "content": prompt}])
        return result.content.strip()
    except Exception as e:
        print(f"  ⚠️ エラー: {e}")
        return question


def main():
    load_dotenv()
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)

    with open(INPUT_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"口語化開始: {len(dataset)} 件")

    colloquial_dataset = []
    for i, item in enumerate(dataset, 1):
        original = item["question"]
        colloquial = colloquialize(original, llm)
        print(f"[{i}/{len(dataset)}] {original[:30]}...")
        print(f"         → {colloquial[:50]}...")

        colloquial_dataset.append({
            **item,
            "question_original": original,
            "question": colloquial,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(colloquial_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(colloquial_dataset)} 件を口語化 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
