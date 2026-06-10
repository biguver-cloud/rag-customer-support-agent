import os
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.config import MODEL_NAME, TEMPERATURE
from rag.prompts import get_mode_prompt
from rag.ui import render_citations, render_agent_log, render_copy_button

API_URL = os.getenv("API_URL", "http://localhost:8000")

THINKING_STEPS: list[tuple[str, str]] = [
    ("🔍", "質問の意図を分析中..."),
    ("📄", "関連ドキュメントを検索中..."),
    ("⚖️", "回答の妥当性を自己検閲中..."),
    ("✅", "最終回答を生成しました"),
]


def _log_path() -> Path:
    filename = datetime.now().strftime("chat_log_%Y_%m_%d.csv")
    return Path(__file__).resolve().parent / "logs" / filename


def _make_steps(done: int, running: int | None = None) -> list[dict]:
    result = []
    for i, (icon, label) in enumerate(THINKING_STEPS):
        if i < done:
            result.append({"icon": "✅", "label": label, "status": "done"})
        elif i == running:
            result.append({"icon": icon, "label": label, "status": "running"})
        else:
            result.append({"icon": icon, "label": label, "status": "pending"})
    return result


def _show_sidebar(
    placeholder,
    done: int,
    running: int | None = None,
    is_processing: bool = True,
    self_eval: dict | None = None,
    exec_meta: dict | None = None,
) -> None:
    data: dict = {"steps": _make_steps(done, running), "is_processing": is_processing}
    if self_eval is not None:
        data["self_eval"] = self_eval
    if exec_meta is not None:
        data["exec_meta"] = exec_meta
    with placeholder.container():
        render_agent_log(data)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


def safe_avatar(icon_path: Path) -> str | None:
    if icon_path and icon_path.exists():
        return str(icon_path)
    if "avatar_warning_shown" not in st.session_state:
        st.session_state.avatar_warning_shown = True
        st.warning(f"⚠️ アバター画像が見つかりません: {icon_path.name if icon_path else 'None'}（デフォルトアイコンで表示します）")
    return None


def call_chat_api(question: str) -> dict:
    try:
        resp = httpx.post(
            f"{API_URL}/api/chat",
            json={"question": question},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = e.response.text
        raise RuntimeError(f"API エラー ({e.response.status_code}): {detail}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"API への接続に失敗しました: {e}") from e


def main():
    st.set_page_config(page_title="問い合わせ対応支援RAGエージェント", layout="wide")

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY が .env に設定されていません")
        st.stop()
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()

    BASE_DIR = Path(__file__).resolve().parent
    user_icon_path = BASE_DIR / "images" / "User_アイコン.png"
    ai_icon_path = BASE_DIR / "images" / "AI_アイコン.png"

    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "display_mode" not in st.session_state:
        st.session_state.display_mode = None
    if "formatted_answers" not in st.session_state:
        st.session_state.formatted_answers = {}
    if "agent_log" not in st.session_state:
        st.session_state.agent_log = None

    def _set_call_mode():
        st.session_state.display_mode = "call"

    def _set_chat_mode():
        st.session_state.display_mode = "chat"

    # ── サイドバー ────────────────────────────────────────────────────────────
    with st.sidebar:
        agent_log_placeholder = st.empty()
        with agent_log_placeholder.container():
            render_agent_log(st.session_state.agent_log)

        st.divider()
        st.markdown("**質問例**")
        st.markdown("- 解約したい")
        st.markdown("- 返金条件を教えて")
        st.markdown("- 請求内容を確認したい")

        st.divider()
        st.markdown("**ログ出力**")
        log_path = _log_path()
        if log_path.exists():
            with open(log_path, "rb") as f:
                st.download_button(
                    label="📥 CSVダウンロード",
                    data=f,
                    file_name=log_path.name,
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.caption("まだログがありません")
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown("<h1 style='text-align:center; margin-bottom: 0.2em;'>問い合わせ対応支援RAGエージェント</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: #666; margin-top: -10px; margin-bottom: 20px;'><b>社内資料・PDFを知識源として問い合わせ回答を支援するAI</b></p>", unsafe_allow_html=True)
    st.info("📋 資料に基づいた正確な回答をお届けします。解約・返金・請求など、よくある質問にすぐに対応いたします。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    MAX_MESSAGES = 20

    if st.session_state.display_mode and st.session_state.last_answer:
        mode = st.session_state.display_mode
        if mode not in st.session_state.formatted_answers:
            llm = get_llm()
            prompt = get_mode_prompt(mode, st.session_state.last_answer)
            with st.spinner("整形中..."):
                formatted = llm.invoke([{"role": "user", "content": prompt}]).content
            st.session_state.formatted_answers[mode] = formatted
            label = "📞 コールモード" if mode == "call" else "💬 チャットモード"
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"**【{label}】**\n\n{formatted}",
                "mode": mode,
                "formatted_text": formatted,
            })
            st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
        st.session_state.display_mode = None
        st.rerun()

    messages_to_show = st.session_state.messages[-MAX_MESSAGES:]
    last_rag_idx = next(
        (i for i, m in reversed(list(enumerate(messages_to_show)))
         if m["role"] == "assistant" and "citations" in m),
        None,
    )
    for i, m in enumerate(messages_to_show):
        avatar = safe_avatar(user_icon_path) if m["role"] == "user" else safe_avatar(ai_icon_path)
        with st.chat_message(m["role"], avatar=avatar):
            if m.get("mode") == "call":
                st.markdown("**【📞 コールモード】**")
                formatted_html = m["formatted_text"].replace("\n", "<br>")
                st.markdown(
                    f'<div style="font-size:0.875rem; line-height:1.8; color:inherit">{formatted_html}</div>',
                    unsafe_allow_html=True,
                )
            elif m.get("mode") == "chat":
                st.markdown(m["content"])
                render_copy_button(m["formatted_text"])
            else:
                st.markdown(m["content"])
            if i == last_rag_idx:
                render_citations(m.get("citations", []))

    if st.session_state.last_answer:
        col1, col2 = st.columns(2)
        with col1:
            st.button("📞 コールモード", use_container_width=True, on_click=_set_call_mode)
        with col2:
            st.button("💬 チャットモード", use_container_width=True, on_click=_set_chat_mode)

    user_text = st.chat_input("ここに質問内容をご入力ください")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

    with st.chat_message("user", avatar=safe_avatar(user_icon_path)):
        st.markdown(user_text)

    # サイドバー: 処理中状態を表示
    _show_sidebar(agent_log_placeholder, done=0, running=0, is_processing=True)

    with st.chat_message("assistant", avatar=safe_avatar(ai_icon_path)):
        with st.spinner("PDFから検索して回答中..."):
            try:
                data = call_chat_api(user_text)
            except RuntimeError as e:
                st.error(str(e))
                return

        answer = data["answer"]
        citations = [dict(c) for c in data.get("citations", [])]
        accuracy = data.get("accuracy", 0)
        completeness = data.get("completeness", 0)
        agent_loops = data.get("agent_loops", 0)
        agent_tokens = data.get("agent_tokens", 0)

        st.markdown(answer)

    # サイドバー: 全ステップ完了 → 最終状態に更新
    final_log = {
        "steps": _make_steps(len(THINKING_STEPS)),
        "is_processing": False,
        "self_eval": {"accuracy": accuracy, "completeness": completeness},
        "exec_meta": {"loops": agent_loops, "tokens": agent_tokens},
    }
    with agent_log_placeholder.container():
        render_agent_log(final_log)
    st.session_state.agent_log = final_log

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "user_text": user_text,
    })
    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

    st.session_state.last_answer = answer
    st.session_state.formatted_answers = {}
    st.rerun()


if __name__ == "__main__":
    main()
