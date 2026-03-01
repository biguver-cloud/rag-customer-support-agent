import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.config import MODEL_NAME, TEMPERATURE, TOP_K, WEAK_SCORE_THRESHOLD, AGENT_ROUNDS
from rag.prompts import SYSTEM_PROMPT, get_mode_prompt
from rag.query import guess_category, rewrite_query_for_search
from rag.vectorstore import open_vectorstore
from rag.retriever import retrieve_documents_with_score
from rag.agent import agent_answer
from rag.ui import render_citations, render_contact_guidance


@st.cache_resource(show_spinner=False)
def get_db(persist_dir: Path):
    # persist_dir は Path のままでOK（内部で str 化されてもよい）
    return open_vectorstore(persist_dir)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


def safe_avatar(icon_path: Path) -> str | None:
    """
    アバター画像のパスが存在する場合のみ文字列パスを返す。
    存在しない場合はNoneを返してアプリが落ちないようにする。
    
    Args:
        icon_path: 画像ファイルのPathオブジェクト
    
    Returns:
        存在する場合は str(path)、存在しない場合は None
    """
    if icon_path and icon_path.exists():
        return str(icon_path)
    else:
        # 画像が見つからない場合の警告を1回だけ出す
        if "avatar_warning_shown" not in st.session_state:
            st.session_state.avatar_warning_shown = True
            st.warning(f"⚠️ アバター画像が見つかりません: {icon_path.name if icon_path else 'None'}（デフォルトアイコンで表示します）")
        return None


def build_followup_questions(user_text: str) -> str:
    return f"""資料だけでは特定できませんでした。次のどれに近いですか？

1) 解約したい（いつ解約が有効になるか知りたい）
2) 返金できるか知りたい（返金条件を確認したい）
3) 請求・支払いについて知りたい
4) アカウント/ログインについて知りたい
5) その他（状況をもう少し具体的に教えてください）

たとえば「2) 返金。契約開始日が○月○日で、今○日目です」のように書いてください。
"""


def main():
    st.set_page_config(page_title="問い合わせ対応自動化AIエージェント", layout="wide")

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY が .env に設定されていません")
        st.stop()

    # BASE_DIRを絶対パスで固定（相対パスのずれを防ぐ）
    BASE_DIR = Path(__file__).resolve().parent
    persist_dir = BASE_DIR / "storage" / "chroma"
    user_icon_path = BASE_DIR / "images" / "User_アイコン.png"
    ai_icon_path = BASE_DIR / "images" / "AI_アイコン.png"

    if not persist_dir.exists():
        st.error("ベクトルDBがありません。先に `python build_index.py` を実行してください。")
        st.stop()

    # モード管理の session_state 初期化
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "display_mode" not in st.session_state:
        st.session_state.display_mode = None
    if "formatted_answers" not in st.session_state:
        st.session_state.formatted_answers = {}

    # モードボタンのコールバック
    def _set_call_mode():
        st.session_state.display_mode = "call"

    def _set_chat_mode():
        st.session_state.display_mode = "chat"

    # サイドバー
    st.sidebar.markdown("## AIエージェント機能")
    agent_mode = st.sidebar.selectbox("利用有無", ["利用する", "利用しない"], index=0)

    with st.sidebar.expander("⚙️ 詳細", expanded=False):
        st.markdown("**AIエージェントとは**  \n自己評価・改善を繰り返して、より正確な回答を生成する機能です。\n\n⚠️ 処理時間が長くなる可能性があります。")

    # メイン
    st.markdown("<h1 style='text-align:center; margin-bottom: 0.2em;'>問い合わせ対応自動化AIエージェント</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: #666; margin-top: -10px; margin-bottom: 20px;'><b>社内資料・PDFから問い合わせ対応を自動化するAI</b></p>", unsafe_allow_html=True)
    st.info("📋 資料に基づいた正確な回答をお届けします。解約・返金・請求など、よくある質問にすぐに対応いたします。")

    # サンプル質問の表示（初回のみ）
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**質問例**")
    st.sidebar.markdown("- 解約したい")
    st.sidebar.markdown("- 返金条件を教えて")
    st.sidebar.markdown("- 請求内容を確認したい")

    # チャット履歴（無制限に増えるとメモリを食うので上限をつける）
    MAX_MESSAGES = 20

    # モードボタンが押された場合：整形処理 → メッセージに追加 → 再描画
    if st.session_state.display_mode and st.session_state.last_answer:
        mode = st.session_state.display_mode
        if mode not in st.session_state.formatted_answers:
            llm = get_llm()
            prompt = get_mode_prompt(mode, st.session_state.last_answer)
            with st.spinner("整形中..."):
                formatted = llm.invoke([
                    {"role": "user", "content": prompt}
                ]).content
            st.session_state.formatted_answers[mode] = formatted
            label = "📞 コールモード" if mode == "call" else "💬 チャットモード"
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"**【{label}】**\n\n{formatted}",
            })
            st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
        st.session_state.display_mode = None
        st.rerun()

    # 表示は直近だけ
    messages_to_show = st.session_state.messages[-MAX_MESSAGES:]
    # citations を持つ最後の assistant メッセージのインデックスを特定
    last_rag_idx = next(
        (i for i, m in reversed(list(enumerate(messages_to_show)))
         if m["role"] == "assistant" and "citations" in m),
        None
    )
    for i, m in enumerate(messages_to_show):
        avatar = safe_avatar(user_icon_path) if m["role"] == "user" else safe_avatar(ai_icon_path)
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if i == last_rag_idx:
                render_citations(m.get("citations", []))
                render_contact_guidance(m.get("user_text", ""), m.get("citations", []))

    # モードボタン（チャット入力欄の直上に固定表示）
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
    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]  # ここが重要

    with st.chat_message("user", avatar=safe_avatar(user_icon_path)):
        st.markdown(user_text)

    # 新しい質問は履歴に追加済みなので履歴ループで表示される

    with st.chat_message("assistant", avatar=safe_avatar(ai_icon_path)):
        with st.spinner("PDFから検索して回答中..."):
            try:
                # 毎回ロードしない（ここが一番効く）
                db = get_db(persist_dir)

                search_query = rewrite_query_for_search(user_text)
                category = guess_category(user_text)

                # retriever.pyの関数を使ってスコア付き検索
                search_results = retrieve_documents_with_score(
                    vectorstore=db,
                    query=search_query,
                    k=TOP_K,
                    category=category if category != "unknown" else None
                )

                # コンテキストとcitationsを構築
                if not search_results:
                    context = ""
                    citations = []
                    best_score = None
                else:
                    # Documentとスコアを分離
                    docs = [doc for doc, _ in search_results]
                    scores = [score for _, score in search_results]
                    best_score = min(scores) if scores else None

                    # LLMに渡すコンテキスト（page_contentを結合）
                    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

                    # UI表示用のcitations（スコアを含める）
                    citations = []
                    for doc, score in search_results:
                        src = doc.metadata.get("source", "")
                        page = doc.metadata.get("page", None)
                        cat = doc.metadata.get("category", category if category != "unknown" else "unknown")
                        text = doc.page_content.strip().replace("\n", " ")
                        quote = text[:400] + ("..." if len(text) > 400 else "")

                        citations.append({
                            "category": cat,
                            "source": src,
                            "page": (page + 1) if isinstance(page, int) else None,
                            "quote": quote,
                            "score": score  # スコアを追加
                        })

            except Exception as e:
                st.error(f"検索中にエラーが発生しました: {str(e)}")
                context = ""
                citations = []
                best_score = None

            # LLM も毎回作らない
            llm = get_llm()

            if not context.strip():
                answer = "資料に記載がありません。該当するPDF名や用語（例：解約、返金、請求など）を少し具体的に教えてください。"
                citations = []
            elif best_score is not None and best_score < WEAK_SCORE_THRESHOLD:
                answer = build_followup_questions(user_text)
            else:
                if agent_mode == "利用する":
                    status = st.status("AIエージェント実行中...", expanded=True)
                    prog = st.progress(0)

                    def on_progress(label: str, step: int, total: int):
                        status.update(label=label, state="running")
                        prog.progress(int(step / total * 100))

                    answer = agent_answer(llm, user_text, context, rounds=AGENT_ROUNDS, progress=on_progress)
                    status.update(label="完了", state="complete")
                else:
                    prompt = f"""[コンテキスト]
{context}

[質問]
{user_text}

[回答]
"""
                    answer = llm.invoke(
                        [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}]
                    ).content

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "user_text": user_text,
    })
    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

    # 新規回答を last_answer に保存し、整形キャッシュをリセット
    st.session_state.last_answer = answer
    st.session_state.formatted_answers = {}
    # ボタンを即座に表示するために再描画
    st.rerun()


if __name__ == "__main__":
    main()
