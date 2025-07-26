import os
import re
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        self._configure_api_key()
        self.embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
        self.llm = genai.GenerativeModel("gemini-2.5-flash-lite")
        self.chroma_client = chromadb.Client()
        # This will create and process the PDF only once if the collection doesn't exist
        self.collection = self._create_or_get_collection()
        print("Pipeline Initialized Successfully.")

    def _configure_api_key(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        genai.configure(api_key=api_key)

    def _extract_story_with_ocr(self, pdf_path: str) -> str:
        # This is my text extraction function, now part of the class
        if not pdf_path:
            return ""
        print("\n Starting OCR-based text extraction...")
        doc = fitz.open(pdf_path)
        full_ocr_text = ""
        for page_num in range(5, 49):
            if page_num < len(doc):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                image = Image.open(io.BytesIO(pix.tobytes()))
                try:
                    text = pytesseract.image_to_string(image, lang="ben")
                    full_ocr_text += text + "\n"
                except Exception as e:
                    print(f" OCR failed on page {page_num + 1}: {e}")
                    continue
        doc.close()
        start_marker = "আজ আমার বয়স সাতাশ"
        end_marker = "জায়গা পাইয়াছি"
        start_index = full_ocr_text.find(start_marker)
        end_index = full_ocr_text.rfind(end_marker)
        if start_index == -1 or end_index == -1:
            return full_ocr_text
        end_marker_full_line_end = full_ocr_text.find("\n", end_index)
        story_text = full_ocr_text[start_index:end_marker_full_line_end]
        return re.sub(r"\s*\n\s*", "\n", story_text).strip()

    def _create_or_get_collection(self):
        collection_name = "oporichita_e5_final_pass"
        # Check if the collection already exists
        if collection_name in [c.name for c in self.chroma_client.list_collections()]:
            print(f"Collection '{collection_name}' already exists. Reusing it.")
            return self.chroma_client.get_collection(name=collection_name)

        # If it doesn't exist, create it by processing the PDF
        print(
            "Creating new collection and processing PDF. This will take a few minutes..."
        )
        story_text = self._extract_story_with_ocr(
            "app/assets/HSC26-Bangla1st-Paper.pdf"
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750, chunk_overlap=100, separators=["\n\n", "\n", "।"]
        )
        chunks = text_splitter.split_text(story_text)

        collection = self.chroma_client.create_collection(name=collection_name)
        prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]
        collection.add(
            embeddings=self.embed_model.encode(prefixed_chunks).tolist(),
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )
        print("PDF processed and collection created successfully.")
        return collection

    def get_answer(self, query: str, history: list = []):
        # This is my memory-enabled answer generation function
        prefixed_query = f"query: {query}"
        results = self.collection.query(
            query_embeddings=[self.embed_model.encode(prefixed_query).tolist()],
            n_results=4,
        )
        context = "\n\n---\n\n".join(results["documents"][0])

        formatted_history = "\n".join([f"Human: {q}\nAI: {a}" for q, a in history])

        prompt = f"""You are a helpful assistant for the story 'Oporichita'.
        Answer the user's 'Human' question based on the 'Chat History' and the 'Retrieved Context'.
        Be concise and answer in Bengali. If the context doesn't support the answer, say "গল্পের তথ্য অনুযায়ী উত্তরটি আমার জানা নেই।"

        Chat History:
        {formatted_history}

        Retrieved Context:
        {context}

        Human: {query}
        AI:"""

        response = self.llm.generate_content(prompt)
        return response.text.strip()
