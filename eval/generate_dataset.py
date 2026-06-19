"""
PDFから評価用Q&Aデータセットを自動生成するスクリプト。

各PDFの内容をLLMに読み込ませ、「良い質問と正解」のペアを生成して
eval/dataset.json に上書き保存する。

使い方:
    python eval/generate_dataset.py
"""
import json
import sys
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import MODEL_NAME, TEMPERATURE

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = Path(__file__).resolve().parent / "dataset.json"

# カテゴリとPDFの対応
CATEGORY_MAP = {
    "unknown": [
        DATA_DIR / "company" / "解約・返金ポリシー.pdf",
    ],
    "service": [
        DATA_DIR / "service" / "料金プラン.pdf",
        DATA_DIR / "service" / "請求について.pdf",
        DATA_DIR / "service" / "アカウント.pdf",
        DATA_DIR / "service" / "利用開始ガイド.pdf",
        DATA_DIR / "service" / "サービス機能概要.pdf",
    ],
    "company": [
        DATA_DIR / "company" / "会社概要.pdf",
        DATA_DIR / "company" / "問い合わせ対応方針.pdf",
    ],
    "customer": [
        DATA_DIR / "customer" / "カスタマープロフィール.pdf",
    ],
}


def extract_qa_from_pdf(pdf_path: Path, category: str, llm) -> list[dict]:
    """PDFの内容からQ&Aペアを生成する。"""
    print(f"  読み込み中: {pdf_path.name}")
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
    except Exception as e:
        print(f"  ⚠️ 読み込みエラー: {e}")
        return []

    content = "\n".join(p.page_content for p in pages)[:3000]

    prompt = f"""以下のPDF内容から、カスタマーサポートで実際に問い合わせが来そうな
質問と、その正確な回答を2〜3セット生成してください。

ルール：
- 質問はPDFに書かれていることのみに基づく
- 回答はPDFの記載内容を忠実にまとめた100〜200文字程度の文章
- 推測や創作は禁止

[PDF内容]
{content}

以下のJSON配列のみ出力してください（前後の説明文不要）：
[
  {{"question": "質問文", "expected_answer": "正解の回答"}},
  ...
]"""

    try:
        response = llm.invoke([{"role": "user", "content": prompt}]).content
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            pairs = json.loads(match.group())
            return [
                {
                    "category": category,
                    "question": p.get("question", ""),
                    "expected_answer": p.get("expected_answer", ""),
                }
                for p in pairs
                if p.get("question") and p.get("expected_answer")
            ]
    except Exception as e:
        print(f"  ⚠️ Q&A生成エラー: {e}")

    return []


def main():
    load_dotenv()
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)

    all_pairs = []
    q_id = 1

    for category, pdf_paths in CATEGORY_MAP.items():
        print(f"\n📂 カテゴリ: {category}")
        for pdf_path in pdf_paths:
            if not pdf_path.exists():
                print(f"  ⚠️ ファイルが見つかりません: {pdf_path}")
                continue
            pairs = extract_qa_from_pdf(pdf_path, category, llm)
            for pair in pairs:
                pair["id"] = f"q{q_id:03d}"
                q_id += 1
                all_pairs.append(pair)
                print(f"  ✅ Q{pair['id']}: {pair['question'][:40]}...")

    # dataset.json に保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(all_pairs)} 件のQ&Aを生成しました → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
