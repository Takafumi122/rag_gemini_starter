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

# ★ 追加インポート: ハイブリッド検索のためにEnsembleRetrieverとBM25Retrieverを使用
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever # キーワード検索リトリーバー

# --- 設定 ---
INDEX_DIR = Path("index")
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
# ★ ハイブリッド検索の重み: ベクトル検索を重視 (例: 0.5)
VECTOR_SEARCH_WEIGHT = 0.5
# -----------------

# 環境変数をロード
load_dotenv()

# RAGチェーンをグローバルに保持するための変数
rag_chain = None

def setup_rag_chain():
    """
    ベクトルDBとLLMを読み込み、ハイブリッド検索RAGチェーンを構築する
    """
    global rag_chain

    print("--- 1. RAGコンポーネントの初期化 ---")
    
    # 埋め込みモデルの初期化
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    # ベクトルDBのロード (エラー処理は省略せずmain.pyに記載)
    if not INDEX_DIR.exists():
        print(f"🚨 エラー: ベクトルデータベースが見つかりません。'{INDEX_DIR}' を確認し、事前に 'python build_index.py' を実行してください。")
        return None

    try:
        vector_db = Chroma(
            persist_directory=str(INDEX_DIR),
            embedding_function=embedding_model
        )
    except Exception as e:
        print(f"🚨 エラー: ChromaDBのロードに失敗しました。{e}")
        return None
        
    # --- 2. ハイブリッド検索の設定 ---

    # 2-1. ベクトル検索リトリーバー (意味の類似性検索)
    # k=10で多めに候補を取得し、ハイブリッド検索内で絞り込む
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    # 2-2. キーワード検索リトリーバー (単語の完全一致検索)
    # ChromaDBに保存されているドキュメント全体を取得
    all_documents = vector_db.get(include=['metadatas', 'documents'])['documents']
    # BM25Retrieverをドキュメント全体で初期化
    keyword_retriever = BM25Retriever.from_texts(
        all_documents,
        metadatas=[{"source": f"docs/bm25_source_{i}"} for i in range(len(all_documents))] # メタデータの付与
    )
    # k=10で多めに候補を取得
    keyword_retriever.k = 10 

    # 2-3. EnsembleRetrieverで統合
    # 2つのリトリーバーの結果を統合。VECTOR_SEARCH_WEIGHTに基づき、結果の重み付けを行う。
    retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[VECTOR_SEARCH_WEIGHT, 1.0 - VECTOR_SEARCH_WEIGHT],
        search_type="similarity",
        k=5 # 最終的にLLMに渡すチャンク数を5に設定 (ハイブリッド検索結果の上位5つ)
    )

    # --- 3. RAGチェーンの構築（プロンプトとLLM） ---
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
    
    # プロンプトテンプレートの定義 (CoT強化版を使用)
    system_template = (
        "あなたは製造業の新人教育向けRAGアシスタントです。回答は常に断定的な口調を使用してください。\n"
        "回答は、以下の【根拠】の範囲内でのみ行い、根拠に無い情報は**一切推測せず**、その場合は**「不明」**とだけ回答してください。\n"
        "\n# 回答生成のための思考ステップ (Chain-of-Thought)\n"
        "1. **【情報抽出】**: 質問に対する答えとなる**具体的なキーワードや文章**を【根拠】からすべて抜き出し、一時的にリスト化せよ。\n"
        "2. **【判断と統合】**: リスト化された情報のみを使い、質問への回答を2〜4文で簡潔にまとめよ。もしリストが空であれば、手順3に進め。\n"
        "3. **【検証と出力】**: 抽出情報が不十分または存在しない場合は、回答を「不明」とせよ。十分な場合は、必ず回答本文の後に、使用した根拠のファイル名（パス）を『出典』として列挙せよ。\n"
        "\n【根拠】:\n{context}\n"
        "\n# 最終的な出力フォーマット\n"
        "（回答本文）\n"
        "出典:\n"
        "- [ファイル名/パス 1]\n"
        "- [ファイル名/パス 2]\n"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "質問: {input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain

# ... (answer関数とif __name__ブロックは変更なし)
# main.pyの残りの部分は、前の回答で提示したコードと同じです。

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