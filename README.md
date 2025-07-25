# Multilingual RAG System for "Oporichita"

> A lightweight, multilingual Retrieval-Augmented Generation (RAG) API that allows users to ask questions about Rabindranath Tagore's short story "Oporichita" in both English and Bengali.

It extracts text from a scanned PDF, builds a semantic knowledge base, and uses a generative AI model to provide context-grounded answers.

---

## Features

* **Multilingual Queries**: Accepts and understands questions in both Bengali and English.
* **Conversational Memory**: Maintains short-term and long-term chat history for follow-up questions via the `/chat` endpoint.
* **Accurate Data Extraction**: Uses Tesseract OCR to extract text from scanned PDF documents.
* **Semantic Search**: Employs a powerful multilingual sentence transformer model (`intfloat/multilingual-e5-large`) to find the most relevant document chunks.
* **Grounded Generation**: Uses Google's Gemini 2.5-flash-lite model to generate answers based solely on the retrieved context, minimizing hallucinations.
* **REST API**: A fully functional backend with interactive documentation provided by FastAPI.

---

## Tech Stack & Tools Used

* **Backend**: FastAPI, Uvicorn
* **LLM**: Google Gemini (`gemini-2.5-flash-lite`)
* **Embedding Model**: `intfloat/multilingual-e5-large`
* **Vector Database**: ChromaDB
* **Data Processing**: PyMuPDF, Tesseract OCR, LangChain (for text splitting)
* **Language**: Python 3.11+

---

## Setup and Installation

Follow these steps to set up and run the project locally.

#### 1. Prerequisites

You must have **Tesseract OCR** installed on your system.

* **Windows (GUI Installer)**: Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). **Ensure you select the "Bengali" language pack during installation.**

* **Windows (Command Line - Chocolatey): Open PowerShell and use Run as Adminstration**:

  ```powershell
  choco install tesseract --params="'/AddLang:ben'"
  ```
  ```cmd
  tesseract --list-langs
  ```

* **macOS (Homebrew)**:

  ```bash
  brew install tesseract tesseract-lang
  ```

* **Linux (APT)**:

  ```bash
  sudo apt-get install tesseract-ocr tesseract-ocr-ben
  ```

#### 2. Clone the Repository

```bash
git clone https://github.com/Sajidcodecrack/Multilingual-RAG-System.git
cd Backend
```

#### 3. Create and Activate Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Set Up Environment Variables

Create a file named `.env` in the root of the project directory and add your Google API key:

```
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
```

#### 6. Run the API Server

```bash
uvicorn app.main:app --reload
```

The API will now be running at `http://127.0.0.1:8000`.

---

## API Documentation

Interactive API documentation (Swagger UI) is automatically generated by FastAPI. Once the server is running, you can access it at:

**`http://127.0.0.1:8000/docs`**

### Endpoints

#### `POST /ask`

For single-turn questions without memory. The Swagger UI includes pre-filled examples to test the core functionality.

* **Example `curl` request:**

  ```bash
  curl -X 'POST' \
    'http://127.0.0.1:8000/ask' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
  }'
  ```

#### `POST /chat`

For conversational questions. This endpoint accepts a `history` of the conversation to provide context for follow-up questions.

* **Example `curl` request:**

  ```bash
  curl -X 'POST' \
    'http://127.0.0.1:8000/chat' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "question": "সে কোথায় কাজ করে?",
    "history": [
      ["হরিশ কে?", "হরিশ অনুপমের বন্ধু এবং কানপুরে কাজ করে।"]
    ]
  }'
  ```

---

## Evaluation

The system was evaluated using a set of human-labeled examples based on the story's content. The model successfully passed all test cases.

| Question                                        | Expected Answer | RAG Output   | Result |
| ----------------------------------------------- | --------------- | ------------ | ------ |
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?         | শস্তুনাথবাবু    | শস্তুনাথবাবু | Pass   |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামা            | মামা         | Pass   |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?        | পনেরো           | পনেরো        | Pass   |
| হরিশ কোথায় কাজ করে?                            | কানপুরে         | কানপুর       | Pass   |

---

## Here is your **Assessment Questions & Answers**

---

##  Assessment Questions & Answers

<details>
<summary><strong>1. What method or library did you use to extract the text, and why? Did you face any formatting challenges?</strong></summary>

I used a combination of **PyMuPDF (`fitz`)** to render PDF pages into high-resolution images and **Tesseract OCR (`pytesseract`)** with the Bengali language pack.
This method was essential because the provided PDF was a scanned, image-based document where direct text extraction would fail.

> **Formatting Challenges:**
>
> * Minor OCR inaccuracies
> * Extra newlines and layout noise
>
>  Solution: I isolated the story content between specific start/end markers and used regex to clean up whitespace.

</details>

---

<details>
<summary><strong>2. What chunking strategy did you choose and why do you think it works well for semantic retrieval?</strong></summary>

I used **`RecursiveCharacterTextSplitter`** from LangChain with:

* **Chunk size:** 750 characters
* **Overlap:** 100 characters

This strategy works well for semantic retrieval because it:

* Prioritizes splitting on natural language boundaries: `\n\n`, `\n`, and Bengali full-stop `।`
* Keeps full thoughts or sentences intact within each chunk

This structure preserves contextual meaning, making it ideal for embedding-based semantic search.

</details>

---

<details>
<summary><strong>3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?</strong></summary>

I used **`intfloat/multilingual-e5-large`**, a multilingual transformer-based embedding model.

>  Why this model?
>
> * Strong performance in both **English and Bengali**
> * Designed for **semantic similarity tasks**
> * Easily integrates with sentence-transformers and ChromaDB

It works by encoding text into dense vector representations, where semantically similar inputs are geometrically close in vector space — allowing for accurate conceptual matching.

</details>

---

<details>
<summary><strong>4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?</strong></summary>

I use **ChromaDB**, a purpose-built vector database for semantic retrieval.

* Each document chunk is embedded and stored in ChromaDB
* At query time, the question is embedded and compared to the chunks using **Cosine Similarity**

>  Why ChromaDB?
>
> * Fast and lightweight
> * Designed for retrieval-augmented generation (RAG)
> * Simple integration with LangChain and sentence-transformers

</details>

---

<details>
<summary><strong>5. How do you ensure meaningful comparison? What would happen if the query is vague or missing context?</strong></summary>

To ensure meaningful comparison:

* I use **high-quality embeddings** from the multilingual-e5-large model
* I follow best practices by **prefixing inputs** as `query:` and `passage:` before encoding

>  What if the query is vague?
>
> * A vague question produces a vague vector → results in irrelevant chunk retrieval
> * The system is instructed to respond with:
>   *“গল্পের তথ্য অনুযায়ী উত্তরটি আমার জানা নেই।”*
>   if the context is insufficient to generate an accurate answer

</details>

---

<details>
<summary><strong>6. Do the results seem relevant? If not, what might improve them?</strong></summary>

 Yes, the results are highly relevant and accurately reflect the content of the story. Test cases confirm this.

>  Potential Improvements:
>
> * **Tune Chunking:** Try different chunk sizes (e.g., 512, 1024) and overlap values
> * **Enhance OCR Cleaning:** Post-process common Bengali OCR misreads
> * **Add Re-Ranking:** Use a secondary relevance model to re-rank top-k retrieved chunks before passing to LLM

</details>

---






