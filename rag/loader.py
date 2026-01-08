from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf_documents(pdf_root: Path):
    """
    pdf_root配下のPDFを再帰的に読み込み、Documentのリストを返す
    metadata["source"] に相対パス（例: data/service/解約.pdf）を入れる
    """
    pdf_files = sorted(pdf_root.rglob("*.pdf"))
    if not pdf_files:
        raise RuntimeError("PDFが見つかりません（data配下にPDFを置いてください）")

    documents = []
    base_dir = pdf_root.parent  # 例: portfolio/

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        rel = pdf.relative_to(base_dir)
        for page in pages:
            page.metadata["source"] = str(rel).replace("\\", "/")
        documents.extend(pages)

    return documents, pdf_files
