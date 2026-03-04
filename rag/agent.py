import json
import re
import time
from typing import Callable, Optional
import tiktoken
from .prompts import SYSTEM_PROMPT


def _self_evaluate(llm, question: str, context_excerpt: str, answer: str) -> dict:
    """
    LLMに回答の正確性・網羅性を自己評価させ、0〜100のスコアを返す。

    Returns:
        {"accuracy": int, "completeness": int}  ※失敗時は {"accuracy": 0, "completeness": 0}
    """
    eval_prompt = f"""あなたは回答品質を評価するレビュアーです。
以下の質問・参照資料・回答のセットを評価し、JSONのみを出力してください。

[質問]
{question}

[参照した資料（抜粋）]
{context_excerpt}

[生成した回答]
{answer}

評価基準:
- accuracy  (正確性 0〜100): 回答が参照資料の内容に忠実か。資料に根拠のある記述が多いほど高い。
- completeness (網羅性 0〜100): 質問に対して必要な情報を過不足なく含んでいるか。

出力形式（JSON のみ、説明文は不要）:
{{"accuracy": <整数>, "completeness": <整数>}}"""

    try:
        response = llm.invoke([{"role": "user", "content": eval_prompt}]).content
        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group())
            return {
                "accuracy": max(0, min(100, int(data.get("accuracy", 0)))),
                "completeness": max(0, min(100, int(data.get("completeness", 0)))),
                "_prompt": eval_prompt,
            }
    except Exception as e:
        print(f"[Agent] 自己評価失敗: {e}")

    return {"accuracy": 0, "completeness": 0, "_prompt": eval_prompt}


def summarize_context(llm, context: str, question: str) -> str:
    """
    contextを要点抽出し、短縮版を返す。
    回答に必要な情報のみを箇条書きにまとめる。
    """
    summary_prompt = f"""次のコンテキストから、質問に答えるために必要な情報のみを箇条書きで抽出してください。

必須ルール:
- 回答に必要な根拠となる情報のみを列挙する
- 引用元や識別情報があれば含める
- 不要な詳細は省略し、要点のみを簡潔に記載する
- 300トークン以内に収める

[質問]
{question}

[コンテキスト]
{context}

[要点抽出(箇条書き)]
"""
    context_slim = llm.invoke(
        [{"role": "system", "content": "あなたは情報を簡潔にまとめる専門家です。"},
         {"role": "user", "content": summary_prompt}]
    ).content
    return context_slim


def agent_answer(
    llm,
    question: str,
    context: str,
    rounds: int = 0,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """
    高速化版agent_answer:
    - contextが短い場合（1500トークン未満）は要約をスキップ
    - 改善ラウンドはデフォルト0（本番では無効化推奨）
    - 改善時はcontextを再送せず会話履歴のみ使用
    - LLM呼び出し回数を最小化（1〜2回で完結）

    Returns:
        dict: {answer, loops, tokens}
    """
    start_time = time.time()
    total_tokens = 0

    # contextの長さをトークン数で判定
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        context_tokens = len(encoding.encode(context))
    except Exception:
        encoding = None
        context_tokens = len(context) // 2

    def _tok(text: str) -> int:
        """テキストのトークン数を推定する。"""
        if encoding is not None:
            try:
                return len(encoding.encode(text))
            except Exception:
                pass
        return len(text) // 4

    # 1500トークン未満なら要約をスキップ
    needs_summary = context_tokens > 1500

    # ステップ数: [圧縮?] + 初回回答 + 改善rounds + 自己評価
    total_steps = (1 if needs_summary else 0) + 1 + rounds + 1
    step = 0

    def tick(label: str, elapsed: float = None):
        nonlocal step
        step += 1
        if elapsed is not None:
            label = f"{label} ({elapsed:.2f}秒)"
        if progress:
            progress(label, step, total_steps)

    # Step 1: contextを短縮（必要な場合のみ）
    if needs_summary:
        step_start = time.time()
        tick("コンテキストを圧縮中...")
        context_slim = summarize_context(llm, context, question)
        elapsed = time.time() - step_start
        slim_tokens = _tok(context_slim)
        total_tokens += context_tokens + slim_tokens
        print(f"[Agent] コンテキスト圧縮: {elapsed:.2f}秒, {context_tokens}→{slim_tokens}トークン")
    else:
        context_slim = context
        print(f"[Agent] コンテキスト圧縮スキップ: {context_tokens}トークン")

    # Step 2: 初回回答を作成
    step_start = time.time()
    tick("初回回答を作成中...")
    base_prompt = f"""[コンテキスト]
{context_slim}

[質問]
{question}

[回答]
"""
    total_tokens += _tok(SYSTEM_PROMPT) + _tok(base_prompt)
    answer = llm.invoke(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": base_prompt}]
    ).content
    total_tokens += _tok(answer)
    elapsed = time.time() - step_start
    print(f"[Agent] 初回回答: {elapsed:.2f}秒")

    # Step 3: 改善ラウンド(rounds回) - contextは再送しない
    for i in range(rounds):
        step_start = time.time()
        tick(f"回答を改善中...({i+1}/{rounds})")

        unified_prompt = f"""次の回答を自己レビューし、改善点を見つけて書き直してください。

必須ルール:
- 「曖昧表現がないか」「手順が具体的か」を確認
- 前回のコンテキストに基づいた情報のみ使用
- 可能なら手順を箇条書きで具体化する
- レビューコメントは出力せず、改善後の回答のみを出力

[質問]
{question}

[現在の回答]
{answer}

[改善後の回答]
"""
        total_tokens += _tok(SYSTEM_PROMPT) + _tok(unified_prompt)
        answer = llm.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": unified_prompt}]
        ).content
        total_tokens += _tok(answer)
        elapsed = time.time() - step_start
        print(f"[Agent] 改善ラウンド{i+1}: {elapsed:.2f}秒")

    # 自己評価ステップ
    step_start = time.time()
    tick("回答の品質を自己評価中...")
    context_excerpt = context_slim[:800]  # 評価用に先頭800字を使用
    eval_result = _self_evaluate(llm, question, context_excerpt, answer)
    eval_prompt = eval_result.pop("_prompt", "")
    total_tokens += _tok(eval_prompt) + 20  # 出力JSONは短いので固定で加算
    elapsed = time.time() - step_start
    print(f"[Agent] 自己評価: {elapsed:.2f}秒, accuracy={eval_result['accuracy']}, completeness={eval_result['completeness']}")

    total_time = time.time() - start_time
    print(f"[Agent] 合計処理時間: {total_time:.2f}秒, 推定トークン: {total_tokens}")

    return {
        "answer": answer,
        "loops": rounds,
        "tokens": total_tokens,
        "accuracy": eval_result["accuracy"],
        "completeness": eval_result["completeness"],
    }
