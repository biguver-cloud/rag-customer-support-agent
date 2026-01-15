# RAG Customer Support Agent (Streamlit)

## はじめに

このリポジトリは、
「実務で通用するRAG構成」を、
**設計・実装・説明まで一貫してできること**を示すためのポートフォリオです。

PDFや社内資料をそのまま活かし、
検索（Retriever）と生成（LLM）を組み合わせた構成で、
**再現性のある実装**と**業務に寄せた回答設計**を重視しました。

---

## 🌐 公開デモ

Streamlit Cloud 上で動作するデモはこちら

👉 [https://rag-customer-support-agent-afisjncrkgflsn7dusecdd.streamlit.app](https://rag-customer-support-agent-afisjncrkgflsn7dusecdd.streamlit.app)

---

## 📌 概要

本ツールは、  
**社内資料やPDFを知識源として、問い合わせ対応を自動化・効率化するAIエージェント**です。

**解約・返金・請求などの定型的な問い合わせ**に対し、  
資料を検索した上で **根拠に基づいた案内レベルの回答** を行います。

### ■ 目的

本ツールの目的は、  
**人が資料を確認しながら対応している問い合わせの一次対応をAIに任せ、  
業務効率と回答品質を両立させること**です。

LLM単体ではなく検索＋生成（RAG）構成を採用し、  
実務で安全に使えることを前提に設計しています。

### ■ 解決できる課題

- **問い合わせ対応のたびに資料確認に時間がかかる**
- **担当者ごとに回答内容がブレる**
- **FAQでは質問の表現ゆれに対応できない**

本ツールにより、  
**対応時間の短縮・回答品質の統一・担当者負荷の軽減** を実現します。

---

## 📂 ディレクトリ構成

```
.
├── app.py            # Streamlit アプリ本体
├── build_index.py    # PDF → ベクトルDB作成
├── requirements.txt
├── .gitignore
├── data/
│   ├── company/      # 会社情報（架空）
│   ├── customer/     # カスタマープロフィール（架空）
│   └── service/      # 料金・解約・利用ガイド等（架空）
├── rag/
│   ├── loader.py
│   ├── retriever.py
│   └── prompt.py
├── storage/
│   └── chroma/       # ChromaDB 永続化データ
└── images/           # README用画像
```

※ `data/` 配下のPDFは **すべて架空データ** です。

---

## 🎬 デモ・実際の画面

### デモ動画

* ▶️ [https://YOUR-DEMO-VIDEO-URL](https://YOUR-DEMO-VIDEO-URL)

### 実際の画面

#### トップ画面

<img width="1280" height="700" alt="実際のツール画面" src="https://github.com/user-attachments/assets/c1b5bf38-338c-414e-b674-e177d7e0234a" />

#### 質問入力と回答例

<img width="1004" height="623" alt="image" src="https://github.com/user-attachments/assets/c315135f-4033-487e-b18d-37012feca3ad" />

#### 検索結果と回答根拠

<img width="1019" height="584" alt="image" src="https://github.com/user-attachments/assets/34d83557-42ec-42d3-be38-4dcbc700dafe" />


---

## 🧠 システム構成

* **UI**：Streamlit
* **LLM**：OpenAI API（via LangChain）
* **Embedding / Vector DB**：ChromaDB
* **Document Loader**：PDF（pypdf）
* **検索方式**：Similarity Search + RAG

---

## 🖥 使用環境

* OS：Windows
* Python：3.11
* フレームワーク：Streamlit
* LLM：OpenAI API（LangChain経由）
* ベクトルDB：ChromaDB
* 主なライブラリ：langchain, chromadb, pypdf, streamlit
* デプロイ：Streamlit Cloud

---

## 🧩 拡張予定機能

* 回答の根拠PDFを画面上に明示（引用表示）
* カテゴリ別検索の精度向上
* 管理者向けログ・評価画面の追加
* 多言語対応（日本語 / 英語）
* 認証・利用制限機能の追加

---

##  セットアップ手順

### 1. リポジトリをクローン

```bash
git clone https://github.com/biguver-cloud/rag-customer-support-agent.git
cd rag-customer-support-agent
```

### 2. 仮想環境の作成（任意）

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\\Scripts\\activate   # Windows
```

### 3. 依存関係をインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

`.env` ファイルを作成し、OpenAI APIキーを設定してください。

```env
OPENAI_API_KEY=your_api_key_here
```

---

## 📄 PDFインデックスの作成

PDFを差し替えた場合や、初回起動時は以下を実行します。

```bash
python build_index.py
```

成功すると `storage/chroma` にベクトルDBが作成されます。

---

## 🖥 アプリ起動

```bash
streamlit run app.py
```

ブラウザで以下にアクセスします。

```
http://localhost:8501
```

---

## 💬 動作イメージ

* ユーザーが問い合わせを入力
* 関連するPDF内容を検索
* 根拠に基づいた回答を生成
* 判断が必要な内容は「案内」に留める

---

## ⚠️ 注意事項

* 本プロジェクトは **学習・ポートフォリオ目的** です
* 実在の企業・人物・サービスは含まれていません

**実運用時に必要な対応**

* 認証・認可
* ログ管理
* 個人情報マスキング
* プロンプト・回答制御の強化

---

## 🧩 今後の改善案

* 回答根拠PDFの明示（引用表示）
* カテゴリ別検索制御
* 管理者向けログ・評価UI
* マルチ言語対応
* デプロイ（Streamlit Cloud / Render / Cloud Run 等）

---

## 👤 Author

GitHub: https://github.com/biguser-cloud  

Purpose:  
社内資料を活用した問い合わせ対応AIの設計・実装を通じて、  
**RAGを用いた業務向けAIエージェント開発スキルを示すためのポートフォリオ**

---

## 📄 License

This project is for educational and demonstration purposes only.

---

## おわりに

本プロジェクトは、
「技術を作る」だけでなく、
「実際の業務でどう使われるか」
「クライアントや利用者にとって何が嬉しいか」
を常に意識しながら設計しました。

RAG・生成AI・業務自動化は、
まだ“正解の形”が固まっていない分野です。
だからこそ、
小さく作って、動かして、改善していくことに
大きな価値があると考えています。

このREADMEやデモをご覧いただいた方が、
「こんな使い方ができそう」
「うちの業務にも応用できそう」
と感じていただけたなら、とても嬉しく思います。

最後までご覧いただき、ありがとうございました。
