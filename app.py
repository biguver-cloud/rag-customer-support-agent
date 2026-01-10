import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.config import MODEL_NAME, TEMPERATURE, TOP_K, WEAK_SCORE_THRESHOLD, AGENT_ROUNDS
from rag.prompts import SYSTEM_PROMPT
from rag.query import guess_category, rewrite_query_for_search
from rag.vectorstore import open_vectorstore, retrieve_with_score
from rag.agent import agent_answer
from rag.ui import render_citations, render_contact_guidance


@st.cache_resource(show_spinner=False)
def get_db(persist_dir: Path):
    # persist_dir は Path のままでOK（内部で str 化されてもよい）
    return open_vectorstore(persist_dir)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


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

    base_dir = Path(__file__).parent
    persist_dir = base_dir / "storage" / "chroma"
    user_icon_path = str(base_dir / "images" / "User_アイコン.png")
    ai_icon_path = str(base_dir / "images" / "AI_アイコン.png")

    if not persist_dir.exists():
        st.error("ベクトルDBがありません。先に `python build_index.py` を実行してください。")
        st.stop()

    # サイドバー
    st.sidebar.markdown("## AIエージェント機能の利用有無")
    agent_mode = st.sidebar.selectbox(" ", ["利用する", "利用しない"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 【AIエージェントとは】")
    st.sidebar.info("質問に対して適切と考えられる回答を生成できるまで、生成AIが自己評価・改善を繰り返す機能です。")
    st.sidebar.warning("AIエージェント機能を利用する場合、回答生成により多くの時間を要する可能性が高いです。")

    # メイン
    st.markdown("<h1 style='text-align:center; margin-bottom: 0.2em;'>問い合わせ対応自動化AIエージェント</h1>", unsafe_allow_html=True)
    st.success("こちらは弊社に関する質問にお答えする生成AIチャットボットです。AIエージェントの利用有無を選択し、画面下部のチャット欄から質問してください。")
    st.warning("具体的に入力したほうが期待通りの回答を得やすいです。")

    # チャット履歴（無制限に増えるとメモリを食うので上限をつける）
    MAX_MESSAGES = 20
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 表示は直近だけ
    for m in st.session_state.messages[-MAX_MESSAGES:]:
        with st.chat_message(m["role"], avatar=user_icon_path if m["role"] == "user" else ai_icon_path):
            st.markdown(m["content"])

    user_text = st.chat_input("こちらからメッセージを送信してください。")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]  # ここが重要

    with st.chat_message("user", avatar=user_icon_path):
        st.markdown(user_text)

    with st.chat_message("assistant", avatar=ai_icon_path):
        with st.spinner("PDFから検索して回答中..."):
            # 毎回ロードしない（ここが一番効く）
            db = get_db(persist_dir)

            search_query = rewrite_query_for_search(user_text)
            category = guess_category(user_text)

            context, citations, best_score = retrieve_with_score(
                db, search_query, k=TOP_K, category=category
            )

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

        st.markdown(answer)
        render_citations(citations)
        render_contact_guidance(user_text, citations)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]  # ここも重要


if __name__ == "__main__":
    main()
