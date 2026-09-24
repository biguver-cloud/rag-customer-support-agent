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


JUDGE_PROMPT_TEMPLATE = """あなたは回答品質を評価する厳格な審査員です。

「質問」「期待する回答」「生成された回答」を比較し、
生成された回答が質問に対して適切かどうかを判定してください。

【判定手順（必ずこの順で行うこと）】
1. 期待回答から、判定に使う核心的事実（数値、期限、手順、固有名詞、条件など）を
   箇条書きで洗い出す
2. 生成された回答の**全文を最初から最後まで読んだ上で**、各事実について
   「生成された回答のどこに書かれているか」を具体的に探す
   （句読点・語順・言い換え・文体の違いは無視して、同じ内容が書かれていれば
   見つかったものとして扱う）
3. 「欠落している」と判定する前に、本当に生成された回答のどこにも
   書かれていないかを、もう一度読み直して再確認する。読み直した結果
   見つかった場合は、欠落ではなく含まれているものとして扱う
4. 上記の確認結果をもとに、最終判定を行う

【判定基準】
○にする条件（すべて満たす場合）：
  - 質問が求めている要点に直接答えている
  - 期待回答に含まれる重要な事実が、手順2〜3の確認の結果、
    過不足なく見つかった

×にする条件（いずれか1つでも該当する場合）：
  - 手順2〜3で再確認しても、期待回答にある重要な事実が本当に見つからない、
    または数値・期限・条件などが誤っている
  - 質問と無関係な内容にすり替わっている、もしくは論点をはぐらかしている
  - 資料に記載がない旨を回答しているが、期待回答には具体的な答えが存在する
  - 断定を避けた曖昧な表現（例：「〜の場合があります」を多用するなど）に終始し、
    期待回答が示す具体的な結論に到達していない

【判定しないこと】
  - 文体・敬語・語順・句読点の有無など表現上の違いだけで減点しない
  - 期待回答にない補足情報が追加されているだけでは減点しない

[質問]
{question}

[期待する回答]
{expected}

[生成された回答]
{generated}

以下のJSON形式のみで回答してください（説明文不要）：
{{"judgment": "○ または ×", "reason": "×の場合は、再確認した上でどの事実が生成された回答のどこにも見つからなかったかを具体的に1文で。○の場合は判定根拠を1文で"}}"""


def _invoke_judge(question: str, expected: str, generated: str, llm) -> dict:
    """LLMを1回呼び出し、判定結果を1件返す内部ヘルパー。"""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, expected=expected, generated=generated)
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


def llm_judge(question: str, expected: str, generated: str, llm) -> dict:
    """
    LLM as a Judge：生成回答の品質を ○/× で判定する（1回判定）。

    完全一致ではなく「重要な情報が含まれているか」で判定するため、
    表現が違っても内容が正しければ ○ になる。

    Returns:
        {"judgment": "○" or "×", "reason": str}
    """
    return _invoke_judge(question, expected, generated, llm)


def llm_judge_majority(question: str, expected: str, generated: str, llm, n_votes: int = 3) -> dict:
    """
    LLM as a Judge：同じ判定を n_votes 回実行し、多数決で最終判定する。

    1回だけの判定はプロンプトへの解釈ゆれで結果がぶれることがあるため、
    複数回判定した多数決を採用して安定性を高める。

    Args:
        llm: 判定用のLLM。多数決に意味を持たせるため、呼び出し側で
             temperature > 0 のインスタンスを渡すこと（temperature=0だと
             毎回同じ結果になり多数決が機能しない）。
        n_votes: 判定を実行する回数（デフォルト3、奇数推奨）。

    Returns:
        {
            "judgment": "○" or "×"（多数決の結果）,
            "reason": str（多数決側の判定のうち最初の理由）,
            "votes": ["○", "×", "○"] のような各回の判定,
            "agreement": "2/3" のような多数決側の一致率,
        }
    """
    votes = [_invoke_judge(question, expected, generated, llm) for _ in range(n_votes)]
    judgments = [v["judgment"] for v in votes]

    maru_count = judgments.count("○")
    batsu_count = judgments.count("×")
    majority = "○" if maru_count >= batsu_count else "×"
    majority_votes = maru_count if majority == "○" else batsu_count

    reason = next((v["reason"] for v in votes if v["judgment"] == majority), "")

    return {
        "judgment": majority,
        "reason": reason,
        "votes": judgments,
        "agreement": f"{majority_votes}/{n_votes}",
    }
