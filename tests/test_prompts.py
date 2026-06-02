import pytest

from rag.prompts import get_mode_prompt, CALL_MODE_PROMPT, CHAT_MODE_PROMPT


class TestGetModePrompt:

    def test_call_mode_contains_answer(self):
        """コールモードの出力に回答が含まれる"""
        answer = "解約は月末までに申請が必要です。"
        result = get_mode_prompt("call", answer)
        assert answer in result

    def test_chat_mode_contains_answer(self):
        """チャットモードの出力に回答が含まれる"""
        answer = "返金は14日以内が条件です。"
        result = get_mode_prompt("chat", answer)
        assert answer in result

    def test_call_mode_matches_template(self):
        """コールモードが CALL_MODE_PROMPT テンプレートを使っている"""
        answer = "テスト回答"
        assert get_mode_prompt("call", answer) == CALL_MODE_PROMPT.format(answer=answer)

    def test_chat_mode_matches_template(self):
        """チャットモードが CHAT_MODE_PROMPT テンプレートを使っている"""
        answer = "テスト回答"
        assert get_mode_prompt("chat", answer) == CHAT_MODE_PROMPT.format(answer=answer)

    def test_call_and_chat_are_different(self):
        """コールモードとチャットモードは異なるプロンプトを生成する"""
        answer = "テスト回答"
        assert get_mode_prompt("call", answer) != get_mode_prompt("chat", answer)

    def test_invalid_mode_raises_value_error(self):
        """不正なモードは ValueError を発生させる"""
        with pytest.raises(ValueError):
            get_mode_prompt("invalid", "テスト回答")

    def test_invalid_mode_error_message(self):
        """ValueError のメッセージにモード名が含まれる"""
        with pytest.raises(ValueError, match="invalid_mode"):
            get_mode_prompt("invalid_mode", "テスト回答")
