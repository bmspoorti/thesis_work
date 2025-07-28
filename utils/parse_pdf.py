import os
import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts and cleans text from a given PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def load_all_pdfs(folder_path: str) -> list:
    """Loads and returns text from all PDFs in the folder."""
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            raw_text = extract_text_from_pdf(full_path)
            texts.append({
                "filename": filename,
                "content": raw_text
            })
    return texts

# 🔍 Test block
if __name__ == "__main__":
    pdf_folder = "data/curriculum_pdfs"
    pdfs = load_all_pdfs(pdf_folder)
    for pdf in pdfs:
        print(f"\n📘 {pdf['filename']}")
        print(f"Preview (first 500 chars):\n{pdf['content'][:500]}")
