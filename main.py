import os
import langchain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval import create_retrieval_chain
from vertexai import generative_models
from vertexai.generative_models import GenerationConfig, GenerativeModel
import vertexai
from dotenv import load_dotenv
import json
from pdfminer.high_level import extract_text
import streamlit as st
from fpdf import FPDF
import google.generativeai as genai
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import tempfile
from streamlit_pdf_viewer import pdf_viewer
import traceback
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PyPDF2 import PdfReader
import logging
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pymupdf
logging.basicConfig(filename='main.log', level=logging.DEBUG)

# Create a directory for user files
USER_FILES_DIR = "user_files"
if not os.path.exists(USER_FILES_DIR):
    os.makedirs(USER_FILES_DIR)

st.set_page_config(page_title="Personal Learning Assistant", page_icon="📚", layout="wide")

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"Path\to\your\credentials.json"
project_id = "personalised-learning-system-project"
vertexai.init(project=project_id, location="us-central1")

quiz_response_schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "question-number": {"type": "NUMBER"},
            "question": {"type": "STRING"},
            "options": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "answer": {"type": "STRING"},
            "difficulty": {"type": "STRING"}
        },
        "required": ["question-number", "question", "options", "answer", "difficulty"]
    }
}

evaluation_theory_response_schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "question": {"type": "STRING"},
            "user_answer": {"type": "STRING"},
            "evaluation": {"type": "STRING"},
            "correct_answer": {"type": "STRING"},
            "content": {"type": "STRING"}
        },
        "required": ["question", "user_answer", "evaluation"]
    }
}

theory_response_schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "question": {"type": "STRING"},
            "answer": {"type": "STRING"}
        },
        "required": ["question", "answer"]
    }
}

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

init_db()

def signup(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        hashed_password = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return check_password_hash(result[0], password)
    return False

def format_text(text):
    response = model.generate_content(
        "In the Following text remove all the numbers and special characters, make it more readable and give the response in paragraphs, don't give it in points only in paragraphs. Here is text: \n" + text)
    return response.text

def extract_text_from_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        temp_file_path = temp_file.name

    text = ""
    with pdfplumber.open(temp_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    os.unlink(temp_file_path)
    return text

def interact_with_gemini(model_id, prompt_text):
    model_instance = GenerativeModel(model_id)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
        ),
    )
    print("Raw response from Gemini:", response.text)  # Debug print

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return response.text

def interact_with_gemini_quiz(model_id, prompt_text):
    model_instance = GenerativeModel(model_id)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
            response_mime_type="application/json",
            response_schema = {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "question-number": {"type": "NUMBER"},
                                        "question": {"type": "STRING"},
                                        "options": {
                                            "type": "ARRAY",
                                            "items": {"type": "STRING"}
                                        },
                                        "answer": {"type": "STRING"},
                                        "difficulty": {"type": "STRING"}
                                    },
                                    "required": ["question-number", "question", "options", "answer", "difficulty"]
                                }
                            }
        )
    )
    print("Raw response from Gemini:", response.text)  # Debug print

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return response.text

def interact_with_gemini_evaluation_theory(model_id, prompt_text):
    model_instance =GenerativeModel(model_id)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=GenerationConfig(
            temperature=0.6,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
            response_mime_type="application/json",
            response_schema = {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "question": {"type": "STRING"},
                                        "user_answer": {"type": "STRING"},
                                        "evaluation": {"type": "STRING"},
                                        "correct_answer": {"type": "STRING"},
                                        "content": {"type": "STRING"}
                                    },
                                    "required": ["question", "user_answer", "evaluation"]
                                }
}
        )
    )
    print("Raw response from Gemini:", response.text)  # Debug print

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return response.text

def interact_with_gemini_theory(model_id, prompt_text):
    model_instance = GenerativeModel(model_id)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
            response_mime_type="application/json",
            response_schema = {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "question": {"type": "STRING"},
                                        "answer": {"type": "STRING"}
                                    },
                                    "required": ["question", "answer"]
                                }
                            }
        )
    )
    print("Raw response from Gemini:", response.text)  # Debug print

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return response.text

def interact_with_gemini_summariser(model_id, prompt_text):
    model_instance = GenerativeModel(model_id)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=GenerationConfig(
            temperature=0.1,
            max_output_tokens=4096,
            top_p=0.8,
            top_k=40,
        ),
    )
    print("Raw response from Gemini:", response.text)  # Debug print

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return response.text

def generate_quiz(content, num_questions=5):
    prompt = f"""Generate a quiz with {num_questions} questions based on the following content:

    {content}

    For each question, provide:
    1. The question text
    2. Four multiple-choice options (A, B, C, D)
    3. The correct answer (A, B, C, or D)

    Format the output with the given response schema.
    """
    model_id = 'gemini-1.5-pro-001'
    response = interact_with_gemini_quiz(model_id, prompt)

    if isinstance(response, list):
        return response
    else:
        st.error("Failed to generate quiz. Please try again.")
        return []

def generate_simplified_content(content, incorrect_questions):
    prompt = f"""Based on the following content and the questions the user answered incorrectly,
    provide a simplified explanation of the key concepts related to these questions:

    Content: {content}

    Incorrect questions:
    {incorrect_questions}

    Please provide a concise, easy-to-understand explanation of the relevant concepts.
    """
    model_id = 'gemini-1.5-pro-001'
    response = interact_with_gemini(model_id, prompt)
    return response

def generate_theory_questions(content, num_questions=5):
    prompt = f"""Generate {num_questions} theoretical questions based on the following content:

    {content}

    Follow the response schema and make sure to generate questions and answers based on the theory response schema.

    """
    model_id = 'gemini-1.5-pro-001'
    response = interact_with_gemini_theory(model_id, prompt)
    if isinstance(response, list):
        return response
    else:
        st.error("Failed to generate theoretical questions. Please try again.")
        return []

def evaluate_theory_answers(content, questions_and_answers):
    prompt = f"""Based on the following content and the user's answers, evaluate the answers and provide feedback:

    Content: {content}

    Questions and user's answers:
    {questions_and_answers}

    Follow the response schema for evaluation. If the evaluation is incorrect, provide the correct answer by reading the content.
    """
    model_id = 'gemini-1.5-pro-001'
    response = interact_with_gemini_evaluation_theory(model_id, prompt)
    return response

def summarize_text_pdf(content):
    prompt = f"""
    You are a highly skilled summarizer. Please summarize the following content in a clear, precise, and concise manner. The summary should be at least 7 pages long and cover all key topics and important information. Ensure the summary captures the essence of the content, highlighting major points and critical details while maintaining readability, End the summary with a breif conclusion.

    Content:
    {content}

    Please provide the summary below:
    """
    model_id = 'gemini-1.5-pro-001'
    response = interact_with_gemini_summariser(model_id, prompt)
    return response

def interact_with_gemini_chatbot(query, context):
    # This is a mock function. Replace it with the actual call to the Gemini model.
    model_id = 'gemini-1.5-pro-001'
    prompt = f"""
    You are a highly skilled assistant. Please provide a detailed and accurate response to the user's query based on the following context.

    Context:
    {context}

    User Query:
    {query}

    Please provide the response below:
    """
    response = interact_with_gemini(model_id, prompt)
    return response

def save_summary_to_pdf(summary, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in summary.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)

def load_user_files(username):
    user_dir = os.path.join(USER_FILES_DIR, username)
    if not os.path.exists(user_dir):
        return []
    return [f for f in os.listdir(user_dir) if f.endswith('.pdf')]

# Step 1: Extract text from PDF
def extract_text_from_pdf(file):
    try:
        doc = pymupdf.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Step 2: Preprocess and segment text
def preprocess_text(text):
    paragraphs = text.split('\n\n')
    return [para.strip() for para in paragraphs if para.strip()]

# Step 3: Index the text
def vectorize_text(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # or use TF-IDF, etc.
    vectors = model.encode(text_chunks)
    return vectors

# Step 4: Process user query
def vectorize_query(query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode([query])
    return query_vector

# Step 5: Retrieve relevant context
def retrieve_relevant_context(query_vector, text_vectors, text_chunks, top_n=5):
    similarities = cosine_similarity(query_vector, text_vectors)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    relevant_context = [text_chunks[i] for i in top_indices]
    return relevant_context

# Main function to handle the process
def answer_query_from_pdf(pdf_path, query):
    text = extract_text_from_pdf(pdf_path)
    text_chunks = preprocess_text(text)
    text_vectors = vectorize_text(text_chunks)
    query_vector = vectorize_query(query)
    context = retrieve_relevant_context(query_vector, text_vectors, text_chunks)
    return context

# Streamlit app
st.title("Personal Learning Assistant")

# Initialize session state
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'pdf_text' not in st.session_state:
    st.session_state['pdf_text'] = None
if 'quiz_data' not in st.session_state:
    st.session_state['quiz_data'] = None
if 'theory_questions' not in st.session_state:
    st.session_state['theory_questions'] = None
if 'text_documents' not in st.session_state:
    st.session_state['text_documents'] = None
if 'user_files' not in st.session_state:
    st.session_state['user_files'] = []

# Login and Signup form
if not st.session_state['authentication_status']:
    st.header("Login or Signup")

    login_tab, signup_tab = st.tabs(["Login", "Signup"])

    with login_tab:
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login(login_username, login_password):
                st.session_state['authentication_status'] = True
                st.session_state['username'] = login_username
                st.session_state['user_files'] = load_user_files(login_username)
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with signup_tab:
        new_username = st.text_input("New Username", key="new_username")
        new_password = st.text_input("New Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        if st.button("Signup"):
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif signup(new_username, new_password):
                st.success("Signup successful! You can now login.")
            else:
                st.error("Username already exists. Please choose a different username.")

if st.session_state['authentication_status']:
    # Initialize the page in session state if it doesn't exist
    if 'page' not in st.session_state:
        st.session_state['page'] = "PDF Upload"

    # Sidebar for navigation
    page = st.sidebar.radio(
        "Choose a feature",
        ["PDF Upload", "Generate Quiz", "Take Quiz", "Theory Q&A", "PDF Summary", "Chat and View", "Notes", "Logout"],
        key="sidebar",
        index=["PDF Upload", "Generate Quiz", "Take Quiz", "Theory Q&A", "PDF Summary", "Chat and View", "Notes", "Logout"].index(st.session_state['page'])
    )

    # Update the page in session state
    st.session_state['page'] = page

    if page == "PDF Upload":
        st.header("PDF Upload")
        pdf_file = st.file_uploader("Upload a PDF", type="pdf", key="pdf_upload")
        if pdf_file:
            user_dir = os.path.join(USER_FILES_DIR, st.session_state['username'])
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
            
            file_path = os.path.join(user_dir, pdf_file.name)
            with open(file_path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            reader = PdfReader(file_path)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text()
            st.session_state['pdf_text'] = pdf_text
            st.session_state['user_files'] = load_user_files(st.session_state['username'])
            st.success(f"PDF '{pdf_file.name}' uploaded and processed successfully!")

    elif page == "Notes":
        st.header("Your Notes")
        if st.session_state['user_files']:
            for file in st.session_state['user_files']:
                st.subheader(file)
                file_path = os.path.join(USER_FILES_DIR, st.session_state['username'], file)
                with open(file_path, "rb") as f:
                    pdf_content = f.read()
                pdf_viewer(input=pdf_content, width=700, height=600)
                st.markdown("---")  # Add a separator between files
        else:
            st.info("You haven't uploaded any files yet.")

    elif page == "Generate Quiz":
        st.header("Generate Quiz from PDF Content")

        if st.session_state['pdf_text'] is None:
            st.warning("No PDF uploaded yet. Please upload a PDF in the PDF Upload section first.")
        else:
            st.success("PDF content loaded. You can now generate a quiz.")

            num_questions = st.number_input("Number of questions", min_value=1, max_value=10)
            if st.button("Generate Quiz", key="generate_quiz_button"):
                quiz_data = generate_quiz(st.session_state['pdf_text'], num_questions)
                st.session_state['quiz_data'] = quiz_data
                if quiz_data:
                    st.success("Quiz generated successfully! Go to 'Take Quiz' to start.")

    elif page == "Take Quiz":
        st.header("Take Quiz")

        if st.session_state['quiz_data'] is None:
            st.warning("No quiz generated yet. Please generate a quiz first.")
        else:
            quiz_data = st.session_state['quiz_data']
            user_answers = []

            for idx, q in enumerate(quiz_data):
                st.subheader(f"Question {idx + 1}: {q['question']}")
                options = q['options']
                user_answer = st.radio("Choose an answer:", options, key=f"q_{idx}")
                user_answers.append({
                    "question": q['question'],
                    "user_answer": user_answer,
                    "answer": q['answer']
                })

            if st.button('Submit Answers', key="submit_answers"):
                score = sum(int(ua['user_answer'] == ua['answer']) for ua in user_answers)
                total_questions = len(user_answers)
                st.success(f'Your score: {score} out of {total_questions}')

                incorrect_questions = [ua['question'] for ua in user_answers if ua['user_answer'] != ua['answer']]
                if incorrect_questions:
                    st.warning("You got some questions wrong. Let's review the key concepts.")
                    simplified_content = generate_simplified_content(st.session_state['pdf_text'], incorrect_questions)
                    st.markdown("### Key Concepts")
                    st.write(simplified_content)
                else:
                    st.success("Great job! You answered all questions correctly.")

    elif page == "Theory Q&A":
        st.header("Theory Q&A")

        if st.session_state['pdf_text'] is None:
            st.warning("No PDF uploaded yet. Please upload a PDF in the PDF Upload section first.")
        else:
            st.success("PDF content loaded. You can now generate and answer theoretical questions.")

            if st.session_state['theory_questions'] is None:
                num_theory_questions = st.number_input("Number of theoretical questions", min_value=1, max_value=10, value=5)
                if st.button("Generate Theory Questions", key="generate_theory_questions"):
                    theory_questions = generate_theory_questions(st.session_state['pdf_text'], num_theory_questions)
                    st.session_state['theory_questions'] = theory_questions
                    if theory_questions:
                        st.success("Theory questions generated successfully!")

            if st.session_state['theory_questions'] is not None:
                theory_questions = st.session_state['theory_questions']
                with st.form(key='theory_form'):
                    user_answers = []
                    for idx, tq in enumerate(theory_questions):
                        st.subheader(f"Theory Question {idx + 1}")
                        st.markdown(tq['question'])
                        user_answer = st.text_area(f"Your Answer to Question {idx + 1}", key=f"theory_q_{idx}")
                        user_answers.append({
                            "question": tq['question'],
                            "user_answer": user_answer
                        })
                    submit_button = st.form_submit_button('Submit All Answers')

                if submit_button:
                    evaluation_result = evaluate_theory_answers(st.session_state['pdf_text'], user_answers)
                    st.markdown("### Evaluation")
                    for idx, result in enumerate(evaluation_result):
                        st.subheader(f"Question {idx + 1}")
                        st.write(f"Question: {result['question']}")
                        st.write(f"Your Answer: {result['user_answer']}")
                        st.write(f"Correct Answer: {result['correct_answer']}")
                        st.write(f"Evaluation: {result['evaluation']}")
                        st.write("---")

    elif page == "PDF Summary":
        st.title("PDF Summarizer")

        if 'summary' not in st.session_state:
            st.session_state['summary'] = ""

        pdf_file = st.file_uploader("Upload a PDF to summarize", type="pdf", key="summary_pdf_upload")

        if pdf_file:
            pdf_text = extract_text_from_pdf(pdf_file)
            st.success("PDF uploaded and text extracted successfully!")
            st.session_state['summary'] = None
            if st.button("Generate Summary"):
                summary = summarize_text_pdf(pdf_text)
                st.session_state['summary'] = summary
                st.success("Summary generated successfully!")
                st.write(summary)

    elif page == "Chat and View":
        st.header("Chat and View")
        uploaded_pdf = st.file_uploader("Upload a PDF for Chat and View", type="pdf")
        user_query = st.text_input("Enter your query here:")
        
        if uploaded_pdf and user_query:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_pdf.read())
                temp_pdf_path = temp_pdf.name

            # Extract text from the uploaded PDF and process the query
            context = answer_query_from_pdf(temp_pdf_path, user_query)
            
            # Assume gemini_model is an instance of Gemini 1.5 Pro
            response = interact_with_gemini_chatbot(user_query, context)
            
            st.write("### Query Response:")
            st.write(response)
   
    elif page == "Logout":
        if st.button("Logout"):
            st.session_state['authentication_status'] = False
            st.session_state['username'] = ""
            st.session_state['pdf_text'] = None
            st.session_state['quiz_data'] = None
            st.session_state['theory_questions'] = None
            st.session_state['text_documents'] = None
            st.session_state['user_files'] = []
            st.success("You have been logged out successfully.")
            st.rerun()

else:
    st.warning("Please enter your credentials to access the app.")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.title("Navigation")
