import re


def guess_category(question: str, llm=None) -> str:
    q = question.lower()

    if any(k in q for k in ["プロフィール", "顧客", "カスタマー", "ユーザー情報"]):
        return "customer"

    if any(k in q for k in ["解約", "返金", "請求", "支払い", "アカウント", "ログイン", "料金", "プラン", "機能", "利用開始"]):
        return "service"

    if any(k in q for k in ["会社", "概要", "所在地", "沿革", "企業", "問い合わせ対応方針"]):
        return "company"

    # キーワードで判定できない場合はLLMにフォールバック
    if llm is not None:
        try:
            prompt = (
                "以下の質問のカテゴリを次の4つから1つだけ回答してください。\n"
                "customer（顧客情報）/ service（サービス・解約・料金）/ company（会社情報）/ unknown（不明）\n\n"
                f"質問: {question}\n\nカテゴリ（1語のみ）:"
            )
            result = llm.invoke(prompt)
            cat = result.content.strip().lower()
            if cat in ("customer", "service", "company"):
                return cat
        except Exception:
            pass

    return "unknown"


def rewrite_query_for_search(question: str, llm=None) -> str:
    """LLMでキーワード抽出し、失敗時は正規表現にフォールバック。"""
    if llm is not None:
        try:
            prompt = (
                "以下の質問から、PDF文書の全文検索に使う検索キーワードを抽出してください。\n"
                "理由・背景・敬語は不要です。名詞や動詞のキーワードのみを短く出力してください。\n\n"
                f"質問: {question}\n\nキーワード:"
            )
            result = llm.invoke(prompt)
            keyword = result.content.strip()
            if keyword:
                return keyword
        except Exception:
            pass

    # フォールバック: 正規表現ベース
    q = question.strip()
    q = re.sub(r"^.+?(?:ので|から|ため(?:に)?)[、,\s]*", "", q)
    q = re.sub(r"(教えて|知りたい|できますか|お願いします|方法は\?|方法|について|したいです|したい|ください|下さい|の際)", "", q)
    q = q.replace("。", "").replace("？", "").replace("?", "").strip()
    q = re.sub(r"[をがはもに]$", "", q).strip()
    return q if q else question
