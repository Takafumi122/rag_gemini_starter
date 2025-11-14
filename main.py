import argparse
import os
from pathlib import Path

# .envファイルから環境変数をロード (APIキー利用のため必須)
from dotenv import load_dotenv
# LangChain, Gemini, ChromaDBに必要なライブラリをインポート
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 設定 ---
INDEX_DIR = Path("index")
# RAGに使用するLLMモデルとEmbeddingモデル
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
# -----------------

# 環境変数をロード
load_dotenv()

# RAGチェーンをグローバルに保持するための変数
rag_chain = None

def setup_rag_chain():
    """
    ベクトルDBとLLMを読み込み、RAGチェーンを構築する
    """
    global rag_chain

    print("--- 1. RAGコンポーネントの初期化 ---")
    
    # 埋め込みモデルの初期化
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    # ベクトルDBのロード
    if not INDEX_DIR.exists():
        print(f"🚨 エラー: ベクトルデータベースが見つかりません。'{INDEX_DIR}' を確認し、事前に 'python build_index.py' を実行してください。")
        return None

    try:
        # データベースのロード。構築時と同じ埋め込みモデルを使う
        vector_db = Chroma(
            persist_directory=str(INDEX_DIR),
            embedding_function=embedding_model
        )
    except Exception as e:
        print(f"🚨 エラー: ChromaDBのロードに失敗しました。{e}")
        return None
        
    # Gemini LLMの初期化
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
    
    # --- 2. RAGチェーンの構築 ---
    
    # 検索コンポーネント（リトリーバー）: 上位3つの関連チャンクを取得
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # プロンプトテンプレートの定義 (課題の要件とベースコードの骨組みを統合)
    system_template = (
        "あなたは製造業の新人教育向けRAGアシスタントです。\n"
        "与えられた「根拠（日本語文）」の範囲でのみ回答してください。根拠に無い内容は推測せず**「不明」**と述べてください。\n"
        "回答は日本語で、2〜4文で簡潔にまとめ、**最後に根拠となったドキュメントの出典を列挙**してください。\n"
        "\n# 根拠\n{context}\n"
        "\n# 出力フォーマット例\n"
        "（本文）\n"
        "出典:\n"
        "- [ファイル名/パス]\n"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "質問: {input}"),
        ]
    )

    # 3. ドキュメントをプロンプトに組み込むチェーン
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 4. 検索（Retriever）と生成（Document Chain）を統合するRAGチェーン
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain


def answer(question: str) -> str:
    """
    RAGチェーンを実行し、質問に対する回答を生成する
    """
    global rag_chain

    # RAGチェーンがまだ初期化されていない場合は初期化する
    if rag_chain is None:
        rag_chain = setup_rag_chain()
        if rag_chain is None:
            return "RAGシステムの設定に失敗しました。エラーメッセージを確認してください。"

    print("🤖 回答生成中...")
    
    # RAGチェーンの実行
    try:
        # invokeで実行。context（根拠）とanswer（回答）が返される
        response = rag_chain.invoke({"input": question})
        
        # 回答部分を返す
        return response["answer"].strip()
        
    except Exception as e:
        return f"🚨 回答生成中にエラーが発生しました: {e}"


if __name__ == "__main__":
    # 環境変数チェック（dotenvが読み込まれていても、念のため）
    if 'GOOGLE_API_KEY' not in os.environ and 'GEMINI_API_KEY' not in os.environ:
        print("🚨 エラー: 環境変数 'GOOGLE_API_KEY' または 'GEMINI_API_KEY' が設定されていません。")
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("question", type=str, nargs="+")
        args = ap.parse_args()
        q = " ".join(args.question)
        
        print("Q:", q)
        print("-" * 30)
        
        # answer関数を呼び出して回答を取得
        result = answer(q)
        print("A:\n" + result)