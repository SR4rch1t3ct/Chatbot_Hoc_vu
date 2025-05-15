import os
import time
import torch
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO)
load_dotenv()

# ========== MODEL CONFIG ==========
# LLAMA_MODEL_PATH = "models/model"
LLAMA_MODEL_PATH = "models/T-LLAMA/models"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_PATH = "vectorstore/faiss_index"

# ========== LOAD LLAMA ==========
try:
    logging.info(f"🔁 Loading LLAMA model from {LLAMA_MODEL_PATH}...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        low_cpu_mem_usage=True,
        torch_dtype="auto"
    )
    model.eval()
    logging.info(f"✅ LLAMA model loaded in {time.time() - start:.2f}s")
except Exception as e:
    logging.error(f"❌ Failed to load LLAMA model: {e}")
    raise

# ========== LOAD VECTORSTORE + EMBEDDING ==========
try:
    logging.info("🔁 Loading embedding model and FAISS vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3, "lambda_mult": 0.7})
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    logging.info("✅ Vectorstore loaded and retriever initialized")
except Exception as e:
    logging.error(f"❌ Failed to load vectorstore: {e}")
    raise

# ========== BOT RESPONSE WITH RAG ==========

import re

def clean_context(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)

    lines = text.split(". ")
    lines = [line for line in lines if len(line.strip()) > 30 and not line.isupper()]
    text = ". ".join(lines)

    text = re.sub(r"(TÀI LIỆU.*?CHƯƠNG [0-9]+)", "", text, flags=re.IGNORECASE)

    return text[:1000].strip()



def get_bot_response(user_input: str) -> str:
    try:
        # 1. Retrieve related documents
        # docs = retriever.invoke(user_input)
        # raw_context = "\n".join(doc.page_content for doc in docs)
        # context = clean_context(raw_context)


        docs = retriever.invoke(user_input)
        context = docs[0].page_content.strip()
        context = context[:1000]
        #context = "\n\n".join([doc.page_content[:500] for doc in docs])

        # 2. Compose prompt
        prompt = f"""Dựa vào ngữ cảnh dưới đây, hãy trả lời câu hỏi một cách chính xác, rõ ràng và ngắn gọn. 

        Nếu ngữ cảnh không chứa thông tin để trả lời, hãy trả lời: "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu."

        [Ngữ cảnh]
        {context}

        [Câu hỏi]
        {user_input}

        

        [Trả lời]"""
        # 3. Generate response using LLAMA
        print("Generating response from model file llama_chatbot")
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.1
            )
        print("Generated outputs with model")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        print("Tokenized response")

        return response[len(prompt):].strip()

    except Exception as e:
        logging.error(f"❌ Error during RAG response: {e}")
        return "Xin lỗi, tôi không thể xử lý câu hỏi này ngay lúc này."
