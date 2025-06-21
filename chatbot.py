import os
import streamlit as st
import torch
import PyPDF2
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="MindCare Chatbot", layout="centered")


@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "local_llama_model",
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("local_llama_model")
    return model, tokenizer

model, tokenizer = load_model()


@st.cache_resource
def load_vectorstore(index_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return None

vectorstore = load_vectorstore()


def extract_text_from_pdfs(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            with open(os.path.join(pdf_folder, filename), "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                documents.append(text)
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text("\n".join(documents))


def generate_response(prompt, model, tokenizer, max_new_tokens=128):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.title(" MindCare ")


with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", placeholder="Ask anything about mental health...")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
  
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(user_input, k=2)
        context = " ".join([doc.page_content for doc in docs])
    
    prompt = f"Context: {context}\n\nQuestion: {user_input}\nAnswer:"
    with st.spinner("🤖 MindCare is thinking..."):
        response = generate_response(prompt, model, tokenizer)


  
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))


for speaker, message in st.session_state.chat_history:
    if speaker == "user":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 MindCare:** {message}")
