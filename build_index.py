import os
from pathlib import Path

# RAGに必要なライブラリをインポート
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
# 汎用的なDirectoryLoaderは、ファイルパスの取得に使用
from langchain_community.document_loaders import DirectoryLoader 
# ★ 変更点: MarkdownHeaderTextSplitterをインポート
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema.document import Document 

# --- 設定値 ---
DOC_DIR = Path("docs")
INDEX_DIR = Path("index")
# 構造ベースのため、チャンクサイズとオーバーラップはMarkdownの構造が優先されます
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 0 
EMBEDDING_MODEL = "models/text-embedding-004"
# -----------------

# 環境変数をロード
from dotenv import load_dotenv
load_dotenv() 

def load_markdowns():
    """
    docs/ディレクトリ内のMarkdownファイルを読み込み、Markdownの見出しに基づいてチャンクに分割する。
    """
    print(f"--- 1. ドキュメントのロード開始: {DOC_DIR} ---")
    
    # globを使って読み込み対象のMarkdownファイルのパスを取得
    markdown_files = list(DOC_DIR.glob("**/*.md"))

    if not markdown_files:
        print("🚨 エラー: ドキュメントが見つかりません。docs/ ディレクトリ内に .md ファイルがあるか確認してください。")
        return []

    # MarkdownHeaderTextSplitterの設定: #, ##, ### レベルの見出しを区切りとして使用
    headers_to_split_on = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]

    # 見出しに基づき分割するスプリッターの初期化
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # 見出しをチャンクのテキストに含める
    )
    
    chunks = []
    
    print("--- 2. 構造ベースのテキスト分割（チャンキング）開始 ---")
    
    for file_path in markdown_files:
        # ファイルの内容を読み込む
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
        except Exception as e:
            print(f"警告: ファイル {file_path} の読み込みに失敗しました。スキップします。エラー: {e}")
            continue

        # スプリッターで分割
        splits = splitter.split_text(markdown_text)
        
        # 分割されたチャンクに元のファイル名とパスをメタデータとして付与
        for split in splits:
            # メタデータとして元のファイルパス（出典）を保持
            split.metadata['source'] = str(file_path)
            chunks.append(split)

    print(f"ロードされたファイル数: {len(markdown_files)}")
    print(f"生成された構造化チャンク数: {len(chunks)}")
    
    return chunks

def build():
    """
    ドキュメントからベクトルデータベース（ChromaDB）を構築する。
    """
    chunks = load_markdowns()
    if not chunks:
        return

    print("--- 3. エンベディングモデルの初期化 ---")
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"🚨 エラー: Gemini Embeddingモデルの初期化に失敗しました。APIキーを確認してください。{e}")
        return

    print(f"--- 4. ベクトルDB構築と保存: {INDEX_DIR} ---")
    
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
    if 'GOOGLE_API_KEY' not in os.environ and 'GEMINI_API_KEY' not in os.environ:
        print("🚨 エラー: 環境変数 'GOOGLE_API_KEY' または 'GEMINI_API_KEY' が設定されていません。")
    else:
        build()