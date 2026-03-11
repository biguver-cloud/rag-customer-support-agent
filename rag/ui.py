import re
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
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

def render_copy_button(text: str) -> None:
    """チャットモード用のクリップボードコピーボタンを描画する。"""
    safe_text = json.dumps(text)  # JS変数に安全に埋め込む（クォート衝突を回避）
    components.html(
        f"""
        <style>
        button {{
            background-color: #f0f2f6;
            border: 1px solid #d0d2d6;
            border-radius: 6px;
            padding: 6px 16px;
            cursor: pointer;
            font-size: 13px;
            color: #333;
        }}
        button:hover {{ background-color: #e0e2e6; }}
        </style>
        <button id="copyBtn">📋 文章をコピー</button>
        <script>
        var copyText = {safe_text};
        document.getElementById('copyBtn').addEventListener('click', function() {{
            navigator.clipboard.writeText(copyText).then(function() {{
                document.getElementById('copyBtn').innerText = '✅ コピーしました';
                setTimeout(function() {{
                    document.getElementById('copyBtn').innerText = '📋 文章をコピー';
                }}, 2000);
            }});
        }});
        </script>
        """,
        height=50,
    )


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

def render_agent_log(agent_log: dict | None) -> None:
    """
    サイドバーのエージェント思考ログパネルを描画する。

    agent_log の構造:
        {
            "steps": [{"icon": str, "label": str, "status": "pending"|"running"|"done"}],
            "is_processing": bool,
            "self_eval": {"accuracy": int, "completeness": int},   # 0-100
            "exec_meta": {"loops": int, "tokens": int},
        }
    """
    st.markdown("### 🧠 推論ステータス")

    if agent_log is None:
        st.caption("質問を入力すると、AIの推論プロセスが表示されます。")
        return

    st.divider()

    # --- 思考ステップ ---
    for step in agent_log.get("steps", []):
        status = step.get("status", "pending")
        label = step.get("label", "")
        icon = step.get("icon", "⬜")

        if status == "done":
            st.markdown(f"✅&nbsp; {label}")
        elif status == "running":
            st.markdown(f"⏳&nbsp; **{label}**")
        else:
            st.markdown(
                f"<span style='color:#999'>⬜&nbsp; {label}</span>",
                unsafe_allow_html=True,
            )

    if agent_log.get("is_processing", True):
        return  # 処理中はスコアを非表示

    # --- 自己評価スコア ---
    self_eval = agent_log.get("self_eval")
    if self_eval is not None:
        st.divider()
        st.markdown("### 📊 自己評価スコア")

        accuracy = self_eval.get("accuracy", 0)
        completeness = self_eval.get("completeness", 0)

        st.caption("正確性 (Accuracy)")
        st.progress(accuracy / 100, text=f"{accuracy}%")
        st.caption("網羅性 (Completeness)")
        st.progress(completeness / 100, text=f"{completeness}%")

    # --- 実行メタデータ ---
    exec_meta = agent_log.get("exec_meta")
    if exec_meta is not None:
        st.divider()
        st.markdown("### ⚙️ 実行メタデータ")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("反復回数", f"{exec_meta.get('loops', 0)} 回")
        with col2:
            st.metric("トークン数", f"~{exec_meta.get('tokens', 0):,}")


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
