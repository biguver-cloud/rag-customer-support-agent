from typing import Callable, Optional
from .prompts import SYSTEM_PROMPT

def agent_answer(
    llm,
    question: str,
    context: str,
    rounds: int = 2,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> str:
    total_steps = 1 + rounds * 2
    step = 0

    def tick(label: str):
        nonlocal step
        step += 1
        if progress:
            progress(label, step, total_steps)

    tick("回答案を作成中...")
    base_prompt = f"""[コンテキスト]
{context}

[質問]
{question}

[回答]
"""
    answer = llm.invoke(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": base_prompt}]
    ).content

    for i in range(rounds):
        tick(f"自己レビュー中...（{i+1}/{rounds}）")
        critique_prompt = f"""あなたは厳しい品質レビュアーです。
次の「回答」をレビューし、改善点を短く箇条書きで指摘してください。

必須ルール:
- 指摘は「コンテキストに基づいているか」「曖昧表現がないか」「手順が具体的か」を中心にする
- コンテキストに無い情報を追加する指示はしない

[コンテキスト]
{context}

[質問]
{question}

[回答]
{answer}

[指摘(箇条書き)]
"""
        critique = llm.invoke(
            [{"role": "system", "content": "あなたはレビュー担当です。"},
             {"role": "user", "content": critique_prompt}]
        ).content

        tick(f"回答を改善中...（{i+1}/{rounds}）")
        improve_prompt = f"""次の「指摘」を必ず反映して、回答を書き直してください。

必須ルール:
- 追加情報は「コンテキスト」に書かれていることだけ
- コンテキストに無いことは「資料に記載がありません」と書く
- 可能なら手順を箇条書きで具体化する

[コンテキスト]
{context}

[質問]
{question}

[現在の回答]
{answer}

[指摘]
{critique}

[改善後の回答]
"""
        answer = llm.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": improve_prompt}]
        ).content

    return answer
