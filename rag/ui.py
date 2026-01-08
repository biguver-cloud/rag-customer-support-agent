import re
from pathlib import Path
import streamlit as st

def render_citations(citations: list[dict]):
    if not citations:
        return

    with st.expander("根拠（参照した資料）"):
        for i, c in enumerate(citations, start=1):
            src = Path(c.get("source", "")).name
            page = c.get("page", None)
            cat = c.get("category", "unknown")
            quote = c.get("quote", "")

            title = f"[{i}] ({cat}) {src}"
            meta = f"ページ: {page}" if page else "ページ: 不明"

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
