from fastapi import FastAPI
from typing import List
import uvicorn
from model import ChatModel
import rag_util

app = FastAPI()

def load_model():
    model = ChatModel(model_id="google/gemma-2b-it", device="cuda")
    return model

def load_encoder():
    encoder = rag_util.Encoder(
        model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu"
    )
    return encoder

model = load_model()  # load our models once and then cache it
encoder = load_encoder()

file_paths = [r'D:\LLM\llm-chatbot-rag\files\nationalism-in-india.pdf']
docs = rag_util.load_and_split_pdfs(file_paths)
DB = rag_util.FaissDb(docs=docs, embedding_function=encoder.embedding_function)

@app.get("/")
def read_root():
    return {"message": "DoubtChatbot API is running!"}

@app.post("/chat")
async def query(question: List[str]):
    answer = model.generate(question=question)
    return {"answer": answer}

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='127.0.0.1')