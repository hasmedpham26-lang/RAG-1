
import os
import uuid
import shutil
from typing import Optional
from datetime import datetime
from pathlib import Path

# fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# langchain
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

persistent_directory = "db/chroma"
os.makedirs("db", exist_ok=True)

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
parser = StrOutputParser()
chain = model | parser

chat_history = []
MAX_HISTORY = 8 


def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


class QueryRequest(BaseModel):
    question: str
    image_data: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(BASE_DIR/"static"/"index.html")


@app.post("/api/upload")
def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".txt"):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files supported")

    temp_path = f"temp_{uuid.uuid4().hex}{ext}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if ext == ".pdf":
            loader = PyMuPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path, encoding="utf-8")

        docs = loader.load()
        for d in docs:
            d.metadata["source"] = file.filename

        chunks = split_documents(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in the file")

        db.add_documents(chunks)
        print(f"[{timestamp}] Successfully indexed {len(chunks)} chunks from {file.filename}")
        return {
            "status": "success",
            "message": f"Successfully indexed {len(chunks)} chunks from {file.filename}!",
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[{timestamp}] Error indexing document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/chat")
def handle_chat(payload: QueryRequest):
    user_question = payload.question
    image_data = payload.image_data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n---[{timestamp}] You asked: {user_question} ---")
    if image_data:
        print(f"[{timestamp}] --- [Image attached] ---")

    try:
        if chat_history:
            rewrite_prompt = [
                SystemMessage(
                    content="Given the conversation history, rewrite the newest question "
                    "into a standalone search query. Return only the rewritten string."
                ),
                *chat_history,
                HumanMessage(content=f"New question: {user_question}"),
            ]
            search_query = chain.invoke(rewrite_prompt).strip()
            print(f"[{timestamp}] Searching for: {search_query}")
        else:
            search_query = user_question

        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(search_query)
        print(f"[{timestamp}] Found {len(docs)} relevant documents")
        context_text = "\n\n".join(f"- {d.page_content}" for d in docs)

        system_instruction = f"""You are a helpful and intelligent assistant.

REFERENCE DOCUMENTS:
{context_text}

INSTRUCTIONS:
1. First, answer the user's question using ONLY the information in the REFERENCE DOCUMENTS above. If you find the answer there, synthesize a clear, accurate and helpful response based strictly on that text.
2. If the user attaches an image, analyze it using your vision capabilities in the context of the user's question and the documents.
3. FALLBACK TO GENERAL KNOWLEDGE: If the information needed is completely missing from the REFERENCE DOCUMENTS, still try to answer using your own general knowledge.
4. DISCLAIMER REQUIREMENT: Whenever you use general knowledge because the documents lacked the answer, you MUST start your response with this exact disclaimer: "⚠️ *Note: This information was not found in your uploaded documents and is based on general knowledge, which may not be completely accurate for your specific context.*"
"""

        human_content = [{"type": "text", "text": user_question}]
        if image_data:
            url = (
                image_data
                if image_data.startswith("data:")
                else f"data:image/jpeg;base64,{image_data}"
            )
            human_content.append({"type": "image_url", "image_url": {"url": url}})

        prompt_message = [
            SystemMessage(content=system_instruction),
            *chat_history,
            HumanMessage(content=human_content),
        ]

        answer = chain.invoke(prompt_message)
    except Exception as e:
        print(f"[{timestamp}] Error answering: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")


    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    del chat_history[:-MAX_HISTORY]

    print(f"[{timestamp}] Answer: {answer}")
    return {"answer": answer}
