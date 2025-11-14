import os
from pathlib import Path
# ★ 追記: .envファイルから環境変数をロード
from dotenv import load_dotenv
load_dotenv()

# RAGに必要なライブラリをインポート
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 設定値 ---
# ドキュメントが格納されているディレクトリ
DOC_DIR = Path("docs")
# ベクトルデータベースを保存するディレクトリ
INDEX_DIR = Path("index")
# チャンクサイズとオーバーラップ（分割したテキスト片のサイズと重複幅）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# エンベディングモデル名
EMBEDDING_MODEL = "models/text-embedding-004"
# -----------------

def load_markdowns():
    """
    docs/ディレクトリ内のMarkdownファイルを読み込み、チャンクに分割する。
    """
    print(f"--- 1. ドキュメントのロード開始: {DOC_DIR} ---")
    
    # DirectoryLoader: 指定ディレクトリ内の全ての .md ファイルをロード
    loader = DirectoryLoader(
        str(DOC_DIR), 
        glob="**/*.md",
        show_progress=True,
        # メタデータとして元のファイルパスを保持
    )
    documents = loader.load()

    if not documents:
        print("🚨 エラー: ドキュメントが見つかりません。docs/ ディレクトリ内に .md ファイルがあるか確認してください。")
        return []

    print(f"ロードされたドキュメント数: {len(documents)}")

    print("--- 2. テキストの分割（チャンキング）開始 ---")
    # RecursiveCharacterTextSplitter: 複数の区切り文字を使って賢く分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"生成されたチャンク数: {len(chunks)}")
    
    return chunks

def build():
    """
    ドキュメントからベクトルデータベース（ChromaDB）を構築する。
    """
    # チャンクの取得
    chunks = load_markdowns()
    if not chunks:
        return

    print("--- 3. エンベディングモデルの初期化 ---")
    try:
        # 環境変数 GOOGLE_API_KEY または GEMINI_API_KEY を使用
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"🚨 エラー: Gemini Embeddingモデルの初期化に失敗しました。APIキーを確認してください。{e}")
        return

    print(f"--- 4. ベクトルDB構築と保存: {INDEX_DIR} ---")
    
    # INDEX_DIRがなければ作成
    INDEX_DIR.mkdir(exist_ok=True)

    # ChromaDBにチャンクを格納し、ベクトル化（エンベディング）とディスクへの永続化を実行
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(INDEX_DIR)
    )
    
    # データベースの永続化を明示的に実行
    vector_db.persist()
    print(f"✅ Index saved at {INDEX_DIR} directory. データベースの構築が完了しました。")

if __name__ == "__main__":
    # 環境変数GOOGlE_API_KEYの確認
    if 'GOOGLE_API_KEY' not in os.environ and 'GEMINI_API_KEY' not in os.environ:
        print("🚨 エラー: 環境変数 'GOOGLE_API_KEY' または 'GEMINI_API_KEY' が設定されていません。")
    else:
        build()