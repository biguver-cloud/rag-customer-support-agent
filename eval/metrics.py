"""
評価指標モジュール

② LLM as a Judge：生成回答が正解と一致しているか ○/× で判定
③ 文字類似度：正解と生成回答の文字レベルの一致率を計算
"""
import re
import json
import difflib


def text_similarity(expected: str, generated: str) -> float:
    """
    文字レベルの類似度を計算する（difflib.SequenceMatcher）。

    空白・改行を除去してから比較することで、
    表現の違いではなく内容の一致度を測る。

    Returns:
        0.0（完全不一致）〜 1.0（完全一致）
    """
    a = re.sub(r'\s+', '', expected)
    b = re.sub(r'\s+', '', generated)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def llm_judge(question: str, expected: str, generated: str, llm) -> dict:
    """
    LLM as a Judge：生成回答の品質を ○/× で判定する。

    完全一致ではなく「重要な情報が含まれているか」で判定するため、
    表現が違っても内容が正しければ ○ になる。

    Returns:
        {"judgment": "○" or "×", "reason": str}
    """
    prompt = f"""あなたは回答品質を評価する厳格な審査員です。

「質問」「期待する回答」「生成された回答」を比較し、
生成された回答が質問に対して適切かどうかを判定してください。

【判定基準】
○：生成回答が質問の要点に答えており、期待回答の重要な情報を含んでいる
×：重要な情報が欠けている、または内容が明らかに誤っている

[質問]
{question}

[期待する回答]
{expected}

[生成された回答]
{generated}

以下のJSON形式のみで回答してください（説明文不要）：
{{"judgment": "○ または ×", "reason": "判定理由を1文で"}}"""

    try:
        response = llm.invoke([{"role": "user", "content": prompt}]).content
        match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            judgment = "○" if "○" in str(data.get("judgment", "")) else "×"
            return {"judgment": judgment, "reason": data.get("reason", "")}
    except Exception as e:
        print(f"  [LLM Judge] エラー: {e}")

    return {"judgment": "×", "reason": "評価エラー"}
