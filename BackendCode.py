import os
import tempfile
import numpy as np
import faiss
import warnings

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore import InMemoryDocstore

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Suppress warnings
warnings.filterwarnings("ignore")

# === FastAPI App ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Models ===
embedding_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
language_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# === Global Vector Store ===
vector_store = None

class QueryRequest(BaseModel):
    query: str

# === Upload PDF ===
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])
            embeddings = np.array(embeddings).astype('float32')

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            docstore = InMemoryDocstore({i: docs[i] for i in range(len(docs))})
            index_to_docstore_id = {i: i for i in range(len(docs))}

            vector_store = FAISS(index=index, docstore=docstore,
                                 index_to_docstore_id=index_to_docstore_id,
                                 embedding_function=embedding_model.embed_documents)

            return JSONResponse(content={"answer": "PDF uploaded and processed successfully!"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": f"Upload failed: {str(e)}"}, status_code=500)

# === Ask Question ===
@app.post("/ask")
async def ask_question(payload: QueryRequest):
    global vector_store
    query = payload.query.strip()

    if not vector_store:
        return {"answer": "No PDF uploaded yet."}

    # Step 1: Embed and search top document chunks
    query_embedding = embedding_model.embed_query(query)
    query_embedding = np.array(query_embedding).astype('float32')

    similar_docs = vector_store.similarity_search_with_score_by_vector(query_embedding, k=3)

    if not similar_docs:
        return {"answer": "No relevant information found in the document."}

    # Step 2: Combine top-3 context chunks
    context = "\n".join([doc[0].page_content for doc in similar_docs])

    # Step 3: Create prompt
    prompt = f"Answer the question clearly using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer in clear bullet points or sentences:"

    # Step 4: Generate answer with FLAN-T5
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    outputs = language_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer.strip()}

# === Run the App ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
