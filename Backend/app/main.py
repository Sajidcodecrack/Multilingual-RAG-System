
from fastapi import FastAPI
from .pipeline import RAGPipeline
from .schemas import Query, ChatQuery, Answer

# Create the FastAPI app instance
app = FastAPI(
    title="Anupoma the  RAG API",
    description="Welcome to the  RAG API to chat with the 'Oporichita' story.",
    version="1.0.0"
)

# Load the RAG pipeline once when the application starts
rag_pipeline = RAGPipeline()


@app.post("/ask", response_model=Answer)
async def ask_question(query: Query):
    """
    Endpoint for single-turn questions (no chat memory).
    """
    response_text = rag_pipeline.get_answer(query.question)
    return {"response": response_text}


@app.post("/chat", response_model=Answer)
async def chat_with_story(query: ChatQuery):
    """
    Conversational endpoint that uses short-term memory (chat history).
    """
    response_text = rag_pipeline.get_answer(query.question, query.history)
    return {"response": response_text}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Oporichita RAG API. Go to /docs for API documentation."}
