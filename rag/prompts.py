SYSTEM_PROMPT = """あなたは社内資料（PDF）に基づいて回答する問い合わせ対応AIです。

必須ルール:
- 回答は「コンテキスト」に書かれている内容だけで作る
- コンテキストに無い情報は推測しない（一般論で埋めない）
- 無い場合は必ず「資料に記載がありません」と答える
- できるだけ簡潔に、必要なら箇条書き
"""

CALL_MODE_PROMPT = """以下のRAG回答を、電話対応オペレーター向けの口語スクリプトに変換してください。

【変換ルール】
- です・ます調の丁寧な敬語を使用する
- 箇条書きは使わず、読み上げ可能な連続した文章にする
- 回答部分のみ出力する（前置き・後書き不要）
- オペレーターがそのまま読み上げられる形式にする

[RAG回答]
{answer}

[電話応対スクリプト]
"""

CHAT_MODE_PROMPT = """以下のRAG回答を、チャットサポート向けの定型テンプレートに変換してください。

【変換ルール】
- 簡潔に要点のみ伝える（できるだけ3文以内）
- コピー＆ペーストしてそのまま送信できる形式
- 必要なら箇条書きを使う
- 絵文字・装飾なし

[RAG回答]
{answer}

[チャット返信テンプレート]
"""


def get_mode_prompt(mode: str, answer: str) -> str:
    if mode == "call":
        return CALL_MODE_PROMPT.format(answer=answer)
    elif mode == "chat":
        return CHAT_MODE_PROMPT.format(answer=answer)
    raise ValueError(f"Unknown mode: {mode}")
