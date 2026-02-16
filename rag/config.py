# ------------------------------------------------------------
# 設定値をここに集約（提出向け）
# ------------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0

# 検索設定
TOP_K = 5
WEAK_SCORE_THRESHOLD = 0.25  # スコアがこれ以下なら補助質問を出す

# スコア変換設定
# "similarity": スコアが0〜1で大きいほど良い場合（類似度）
# "distance": スコアが0に近いほど良い場合（距離）
SCORE_TYPE = "distance"  # Chromaのデフォルトは距離ベース
SHOW_RAW_SCORE = True    # raw値を併記するかどうか

# Agent設定
AGENT_ROUNDS = 0  # 速度優先: 改善ラウンドを無効化
