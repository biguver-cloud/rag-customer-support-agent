import pytest
from unittest.mock import MagicMock

from rag.query import guess_category, rewrite_query_for_search


# ─── guess_category ───────────────────────────────────────────────────────────

class TestGuessCategoryKeyword:
    """キーワード判定（LLM不使用）"""

    def test_service_解約(self):
        assert guess_category("解約したい") == "service"

    def test_service_返金(self):
        assert guess_category("返金してほしい") == "service"

    def test_service_請求(self):
        assert guess_category("今月の請求を確認したい") == "service"

    def test_service_料金(self):
        assert guess_category("料金プランを教えて") == "service"

    def test_service_アカウント(self):
        assert guess_category("アカウントにログインできない") == "service"

    def test_company_会社(self):
        assert guess_category("会社概要を教えてください") == "company"

    def test_company_所在地(self):
        assert guess_category("所在地はどこですか") == "company"

    def test_customer_顧客(self):
        assert guess_category("顧客プロフィールを確認したい") == "customer"

    def test_customer_カスタマー(self):
        assert guess_category("カスタマー情報を見たい") == "customer"

    def test_unknown_no_llm(self):
        assert guess_category("今日の天気は？") == "unknown"

    def test_keyword_wins_over_llm(self):
        """キーワードで判定できる場合はLLMを呼ばない"""
        mock_llm = MagicMock()
        result = guess_category("解約したい", llm=mock_llm)
        assert result == "service"
        mock_llm.invoke.assert_not_called()


class TestGuessCategoryLLMFallback:
    """LLMフォールバック（キーワードで判定できない場合）"""

    def test_llm_returns_service(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "service"
        assert guess_category("よくわからない質問", llm=mock_llm) == "service"

    def test_llm_returns_customer(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "customer"
        assert guess_category("よくわからない質問", llm=mock_llm) == "customer"

    def test_llm_returns_company(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "company"
        assert guess_category("よくわからない質問", llm=mock_llm) == "company"

    def test_llm_returns_invalid_category(self):
        """LLMが想定外の値を返した場合は unknown"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "無効なカテゴリ"
        assert guess_category("よくわからない質問", llm=mock_llm) == "unknown"

    def test_llm_raises_exception(self):
        """LLMがエラーを返した場合は unknown にフォールバック"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        assert guess_category("よくわからない質問", llm=mock_llm) == "unknown"


# ─── rewrite_query_for_search ─────────────────────────────────────────────────

class TestRewriteQueryForSearchRegex:
    """正規表現フォールバック（LLM不使用）"""

    def test_removes_したい(self):
        assert rewrite_query_for_search("解約したい") == "解約"

    def test_removes_したいです(self):
        assert rewrite_query_for_search("解約したいです") == "解約"

    def test_removes_教えて_ください(self):
        assert rewrite_query_for_search("返金条件を教えてください") == "返金条件"

    def test_removes_について_知りたい(self):
        assert rewrite_query_for_search("料金プランについて知りたい") == "料金プラン"

    def test_no_noise_words(self):
        """ノイズワードがない短い質問はそのまま返る"""
        assert rewrite_query_for_search("解約") == "解約"

    def test_fallback_to_original_when_empty(self):
        """変換後に空になる場合は元のクエリを返す"""
        assert rewrite_query_for_search("したい") == "したい"


class TestRewriteQueryForSearchLLM:
    """LLM使用"""

    def test_llm_returns_keyword(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "解約手続き"
        result = rewrite_query_for_search("解約したいのですが手続きを教えてください", llm=mock_llm)
        assert result == "解約手続き"

    def test_llm_returns_empty_falls_back_to_regex(self):
        """LLMが空文字を返した場合は正規表現にフォールバック"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = ""
        result = rewrite_query_for_search("解約したい", llm=mock_llm)
        assert result == "解約"

    def test_llm_raises_exception_falls_back_to_regex(self):
        """LLMがエラーの場合は正規表現にフォールバック"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        result = rewrite_query_for_search("解約したい", llm=mock_llm)
        assert result == "解約"
