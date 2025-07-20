!pip install langchain langchain-groq chromadb sentence-transformers pypdf python-docx Pillow pytesseract

!pip install -U langchain-community

import os
import tempfile
from typing import Tuple, Dict, Any
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract

# Set up Groq API Key
def set_groq_api_key(api_key: str):
    os.environ["GROQ_API_KEY"] = api_key # Use the provided API key

def extract_text_from_pdf(pdf_path: str) -> str:
    raw_text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        raw_text += page.extract_text() + "\n"
    return raw_text

def extract_text_from_docx(docx_path: str) -> str:
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image(image_path: str) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def split_text(raw_text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(raw_text)

def create_vectordb(chunks):
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    temp_dir = tempfile.mkdtemp()
    vectordb = Chroma.from_texts(chunks, embedding=embedding_model, persist_directory=temp_dir)
    return vectordb, temp_dir 


def setup_qa_chain(vectordb):
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return qa_chain

def classify_claim(text: str) -> Tuple[str, str]:
    # Dummy classifier for demonstration
    # Ensure input is treated as string and is lowercase for reliable matching
    text_lower = str(text).lower()
    if "accident" in text_lower:
        return "Auto", "High"
    elif "health" in text_lower or "medical" in text_lower:
        return "Health", "Medium"
    elif "property" in text_lower:
        return "Property", "Low"
    else:
        return "General", "Low"

def check_policy_compliance(text: str) -> str:
    # Dummy compliance check
    # Ensure input is treated as string and is lowercase
    text_lower = str(text).lower()
    if "fraud" in text_lower:
        return "Non-compliant: Possible fraud detected."
    if "excluded" in text_lower:
        return "Non-compliant: Exclusion clause triggered."
    return "Compliant"

# Modified function to accept raw_text directly
def process_claim_document(raw_text: str, user_query: str, api_key: str) -> Dict[str, Any]:
    set_groq_api_key(api_key)

    # Use the raw_text provided as input
    # raw_text = extract_text(file_path, file_type) # Removed file extraction

    claim_type, priority = classify_claim(raw_text)
    compliance = check_policy_compliance(raw_text)
    chunks = split_text(raw_text)
    vectordb, temp_dir = create_vectordb(chunks) # Get temp_dir from create_vectordb
    qa_chain = setup_qa_chain(vectordb)
    answer = qa_chain.run(user_query)

    # Clean up the temporary Chroma directory
    try:
        vectordb.delete_collection()
    except Exception as e:
        print(f"Error during Chroma cleanup: {e}")


    routing = f"Routed to {claim_type} claims team. Priority: {priority}."
    return {
        "answer": answer,
        "claim_type": claim_type,
        "priority": priority,
        "routing": routing,
        "compliance": compliance
    }
import pandas as pd

data = {
    "ClaimID": ["CLM001", "CLM002", "CLM003", "CLM004", "CLM005", "CLM006"],
    "CustomerName": ["John Doe", "Jane Smith", "Amit Kumar", "Sara Lee", "Michael Chen", "Raj Patel"],
    "ClaimType": ["Vehicle Damage", "Medical", "Home Theft", "Travel Delay", "Property Fire", "Health"],
    "ClaimAmount": [12000, 5400, 25000, 800, 40000, 3000],
    "Priority": ["High", "Medium", "High", "Low", "High", "Medium"],
    "Status": ["Submitted", "In Review", "Approved", "Submitted", "Escalated", "Submitted"],
    "SubmissionDate": ["2024-05-12", "2024-06-01", "2024-04-23", "2024-07-10", "2024-03-16", "2024-07-01"],
    "Description": [
        "The customer's car was rear-ended at a traffic light and sustained heavy damage to the bumper and trunk.",
        "Claim submitted for hospitalization expenses due to a fractured leg from a biking accident.",
        "Theft reported at the claimant's residence with missing electronics and jewelry.",
        "Flight to Dubai was delayed by 12 hours due to technical issues, resulting in additional hotel costs.",
        "Fire damage to kitchen and living room caused by electrical short circuit.",
        "Outpatient surgery reimbursement request with supporting hospital documents attached."
    ]
}

df = pd.DataFrame(data)
df.to_csv("sample_claims.csv", index=False)
import pandas as pd
# Removed tempfile and os imports as we are not using a temporary file anymore

df = pd.read_csv("/content/sample_claims.csv")

row = df.iloc[1] 

claim_text = f"""
Claim ID: {row.ClaimID}
Customer: {row.CustomerName}
Type: {row.ClaimType}
Amount: {row.ClaimAmount}
Priority: {row.Priority}
Status: {row.Status}
Submitted On: {row.SubmissionDate}
Description: {row.Description}
"""

query = "What is the claim about?"

# Call process_claim_document with the raw text directly
response = process_claim_document(claim_text, query, api_key)
print(response)


#For PDF : 

import os
from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract

file_path = "/content/sample_claim.pdf" 
file_type = "pdf" 
user_query = "What is the claim about?"
api_key = "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"


try:
    raw_text = extract_text_from_file(file_path, file_type)
    result = process_claim_document(raw_text, user_query, api_key)
    print(result)

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except ValueError as ve:
    print(f"Error processing file: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
