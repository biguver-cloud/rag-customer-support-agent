import re
from pathlib import Path
import streamlit as st
from rag.config import SCORE_TYPE, SHOW_RAW_SCORE


def convert_score_to_rating_10(score: float, score_type: str = "distance") -> float:
    """
    スコアを10点満点の評価に変換する。
    
    Args:
        score: 元のスコア値
        score_type: "similarity" (0〜1で大きいほど良い) または "distance" (0に近いほど良い)
    
    Returns:
        10点満点の評価値（小数1桁）
    """
    if score_type == "similarity":
        # 類似度の場合: 単純に10倍
        rating = round(score * 10, 1)
    else:  # distance
        # 距離の場合: 0に近いほど高得点
        rating = round(10 / (1 + score), 1)
    
    # 0〜10の範囲に収める
    return max(0.0, min(10.0, rating))

def render_citations(citations: list[dict]):
    if not citations:
        return

    with st.expander("根拠（参照した資料）"):
        for i, c in enumerate(citations, start=1):
            src = Path(c.get("source", "")).name
            page = c.get("page", None)
            cat = c.get("category", "unknown")
            quote = c.get("quote", "")
            score = c.get("score", None)  # スコアを取得

            title = f"[{i}] ({cat}) {src}"
            
            # メタ情報にスコアを追加
            meta_parts = []
            if page:
                meta_parts.append(f"ページ: {page}")
            else:
                meta_parts.append("ページ: 不明")
            
            if score is not None:
                # スコアを10点満点に変換して表示
                rating_10 = convert_score_to_rating_10(score, SCORE_TYPE)
                if SHOW_RAW_SCORE:
                    meta_parts.append(f"類似度: {rating_10}/10（raw: {score:.3f}）")
                else:
                    meta_parts.append(f"類似度: {rating_10}/10")
            
            meta = " / ".join(meta_parts)

            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(meta)
                st.markdown(f"> {quote}")

def extract_contact_info_from_citations(citations: list[dict]) -> list[dict]:
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    emails = []
    seen = set()

    for c in citations:
        quote = c.get("quote", "") or ""
        src = Path(c.get("source", "")).name
        page = c.get("page", None)

        for m in email_re.findall(quote):
            if m in seen:
                continue
            seen.add(m)
            emails.append({"value": m, "source": src, "page": page})

    return emails

def render_contact_guidance(user_text: str, citations: list[dict]):
    trigger_keywords = ["返金", "申請", "請求", "解約", "アカウント", "不具合", "障害", "サポート", "問い合わせ"]
    if not any(k in user_text for k in trigger_keywords):
        return

    emails = extract_contact_info_from_citations(citations)
    if not emails:
        return

    st.divider()
    st.subheader("問い合わせ・申請の連絡先")

    with st.container(border=True):
        st.markdown("以下の方法でお問い合わせください（資料に記載のある範囲）。")
        for e in emails:
            page = f"p.{e['page']}" if e["page"] else "ページ不明"
            st.markdown(f"- **メール**：`{e['value']}`（出典：{e['source']} / {page}）")
