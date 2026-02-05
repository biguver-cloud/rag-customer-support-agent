import time
from typing import Callable, Optional
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
    rounds: int = 1,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> str:
    """
    改善版agent_answer:
    - contextを最初に1回だけ短縮
    - レビューと改善を1回のLLM呼び出しに統合
    - 速度計測を実装
    """
    start_time = time.time()
    total_steps = 1 + 1 + rounds  # 短縮 + 初回回答 + 改善ラウンド
    step = 0

    def tick(label: str, elapsed: float = None):
        nonlocal step
        step += 1
        if elapsed is not None:
            label = f"{label} ({elapsed:.2f}秒)"
        if progress:
            progress(label, step, total_steps)

    # Step 1: contextを短縮
    step_start = time.time()
    tick("コンテキストを圧縮中...")
    context_slim = summarize_context(llm, context, question)
    elapsed = time.time() - step_start
    print(f"[Agent] コンテキスト圧縮: {elapsed:.2f}秒")

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

    # Step 3: 統合プロンプトで改善(rounds回)
    for i in range(rounds):
        step_start = time.time()
        tick(f"回答を改善中...({i+1}/{rounds})")
        
        # レビューと改善を1つのプロンプトに統合
        unified_prompt = f"""次の回答を自己レビューし、改善点を見つけて書き直してください。

必須ルール:
- 「コンテキストに基づいているか」「曖昧表現がないか」「手順が具体的か」を確認
- 追加情報は「コンテキスト」に書かれていることのみ使用
- コンテキストに無いことは「資料に記載がありません」と明記
- 可能なら手順を箇条書きで具体化する
- レビューコメントは出力せず、改善後の回答のみを出力

[コンテキスト]
{context_slim}

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
