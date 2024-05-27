from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline
import streamlit as st

import torch
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import os
import psutil

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / (1024 * 1024)} MB")
    print(f"VMS: {mem_info.vms / (1024 * 1024)} MB")

print_memory_usage()

device = torch.device('cpu')
checkpoint = "LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float32
)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# Function to load documents (assuming documents are in a list of dictionaries format)
def load_documents():
    # Example documents, replace with actual document loading logic
    return [
        {"text": "Document 1 text", "metadata": {"source": "doc1"}},
        {"text": "Document 2 text", "metadata": {"source": "doc2"}}
    ]

@st.cache_resource
def build_vector_index():
    documents = load_documents()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create FAISS index
    vectorIndex = FAISS.from_documents(documents, embedding=embeddings)
    
    # Save the vector index to a file for future use
    with open("db/vector_index.pkl", "wb") as f:
        pickle.dump(vectorIndex, f)
    
    return vectorIndex

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    
    # Build or load the vector index
    if os.path.exists("db/vector_index.pkl"):
        with open("db/vector_index.pkl", "rb") as f:
            vectorIndex = pickle.load(f)
    else:
        vectorIndex = build_vector_index()

    retriever = vectorIndex.as_retriever()
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result'] if 'result' in generated_text else 'No result found'
    source_documents = generated_text['source_documents'] if 'source_documents' in generated_text else []
    return answer, source_documents

def main():
    st.title('Chat with your PDF 🦜📄')
    with st.expander('About app'):
        st.markdown(
            """ 
            generative ai

            """
        )

    question = st.text_area("Enter question")
    if st.button("Search"):
        st.info("your ques: " + question)
        st.info("your answer ")
        answer, source_documents = process_answer(question)
        st.write(answer)
        st.write(source_documents)

if __name__ == "__main__":
    main()
