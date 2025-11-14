import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# LangChain, Gemini, ChromaDBに必要なライブラリをインポート
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 設定 ---
INDEX_DIR = Path("index")
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
# -----------------

# 環境変数をロード (.envを読み込む)
load_dotenv()

# StreamlitのセッションステートにRAGチェーンをキャッシュ
# @st.cache_resource を使用し、アプリの再実行時もDBとLLMの初期化をスキップして高速化
@st.cache_resource
def setup_rag_chain():
    """
    ベクトルDBとLLMを読み込み、RAGチェーンを構築し、キャッシュする
    """
    
    # APIキーの存在チェック (キャッシュ前に実行)
    if 'GOOGLE_API_KEY' not in os.environ and 'GEMINI_API_KEY' not in os.environ:
        return None, "🚨 APIキーが設定されていません。.env ファイルに GOOGLE_API_KEY を設定してください。"

    # 埋め込みモデルの初期化
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    # ベクトルDBのロード
    if not INDEX_DIR.exists():
        return None, f"🚨 エラー: ベクトルデータベースが見つかりません。'{INDEX_DIR}' を確認し、事前に 'python build_index.py' を実行してください。"

    try:
        vector_db = Chroma(
            persist_directory=str(INDEX_DIR),
            embedding_function=embedding_model
        )
    except Exception as e:
        return None, f"🚨 エラー: ChromaDBのロードに失敗しました。{e}"
        
    # Gemini LLMの初期化
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
    
    # --- RAGチェーンの構築 ---
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # プロンプトテンプレートの定義
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

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain, None


def run():
    """
    Streamlitアプリケーションのメイン関数
    """
    st.set_page_config(page_title="ミニRAG スターター", layout="wide")
    st.title("🏭 ミニRAG スターター: 質問応答システム")
    st.caption("知識ベース: docs/*.md | LLM: Gemini")

    # RAGチェーンのロード/構築（キャッシュから取得）
    rag_chain, error_message = setup_rag_chain()
    
    if error_message:
        st.error(error_message)
        return

    # UIの定義
    question = st.text_area("質問を入力してください：", placeholder="例：工場で安全を守るために最も大切な考え方は何ですか？")
    
    if st.button("回答を生成") and question:
        with st.spinner("🤖 回答を生成中です..."):
            try:
                # RAGチェーンの実行
                # rag_chainはsetup_rag_chain()で正常に構築されている
                response = rag_chain.invoke({"input": question})
                
                # 回答の表示
                st.subheader("✅ 回答")
                st.markdown(response["answer"].strip())

                # 根拠となったドキュメントの出典をデバッグ情報として表示
                sources = set(doc.metadata.get('source', '不明なソース') for doc in response["context"])
                st.info(f"🔍 根拠（検索チャンク）: {len(response['context'])} 件, 出典ファイル: {', '.join(sources)}")
                
            except Exception as e:
                st.error(f"🚨 回答生成中にエラーが発生しました: {e}")
    elif st.button("回答を生成"):
        st.warning("質問を入力してください。")

if __name__ == "__main__":
    run()