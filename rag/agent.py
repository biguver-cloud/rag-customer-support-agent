import time
from typing import Callable, Optional
import tiktoken
from .prompts import SYSTEM_PROMPT


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
) -> str:
    """
    高速化版agent_answer:
    - contextが短い場合（1500トークン未満）は要約をスキップ
    - 改善ラウンドはデフォルト0（本番では無効化推奨）
    - 改善時はcontextを再送せず会話履歴のみ使用
    - LLM呼び出し回数を最小化（1〜2回で完結）
    """
    start_time = time.time()
    
    # contextの長さをトークン数で判定
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        context_tokens = len(encoding.encode(context))
    except:
        # フォールバック: 1文字=約0.5トークンと仮定
        context_tokens = len(context) // 2
    
    # 1500トークン未満なら要約をスキップ
    needs_summary = context_tokens > 1500
    
    total_steps = (1 if needs_summary else 0) + 1 + rounds
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
        print(f"[Agent] コンテキスト圧縮: {elapsed:.2f}秒, {context_tokens}→{len(encoding.encode(context_slim))}トークン")
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
    answer = llm.invoke(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": base_prompt}]
    ).content
    elapsed = time.time() - step_start
    print(f"[Agent] 初回回答: {elapsed:.2f}秒")

    # Step 3: 改善ラウンド(rounds回) - contextは再送しない
    for i in range(rounds):
        step_start = time.time()
        tick(f"回答を改善中...({i+1}/{rounds})")
        
        # 改善プロンプト（contextは送らず、初回の要点を参照）
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
        answer = llm.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": unified_prompt}]
        ).content
        elapsed = time.time() - step_start
        print(f"[Agent] 改善ラウンド{i+1}: {elapsed:.2f}秒")

    total_time = time.time() - start_time
    print(f"[Agent] 合計処理時間: {total_time:.2f}秒")
    
    return answer
