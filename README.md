# ミニRAG スターター

## セットアップ
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # GOOGLE_API_KEY を設定
python build_index.py  # docs/ を索引化
```

## CLI
```bash
python main.py "5S活動の目的を2つ挙げてください"
```

## Web UI
```bash
streamlit run app_streamlit.py
```

## 構成
- 生成: Geminiに根拠を渡して要約・引用付き回答
- プロンプト方針: 根拠外は「不明」と回答
