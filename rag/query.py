import re

def guess_category(question: str) -> str:
    q = question.lower()

    if any(k in q for k in ["プロフィール", "顧客", "カスタマー", "ユーザー情報"]):
        return "customer"

    if any(k in q for k in ["解約", "返金", "請求", "支払い", "アカウント", "ログイン", "料金", "プラン", "機能", "利用開始"]):
        return "service"

    if any(k in q for k in ["会社", "概要", "所在地", "沿革", "企業", "問い合わせ対応方針"]):
        return "company"

    return "unknown"


def rewrite_query_for_search(question: str) -> str:
    q = question.strip()
    q = re.sub(r"(教えて|知りたい|できますか|お願いします|方法は\?|方法|について)", "", q)
    q = q.replace("？", "").replace("?", "").strip()
    return q if q else question
