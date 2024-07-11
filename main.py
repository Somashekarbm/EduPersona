import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
import vertexai.generative_models
import vertexai
from dotenv import load_dotenv
import json
from pdfminer.high_level import extract_text
import streamlit as st
from fpdf import FPDF
# from google.cloud import texttospeech
import google.generativeai as genai
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import tempfile
from streamlit_pdf_viewer import pdf_viewer

# Create a directory for user files
USER_FILES_DIR = "user_files"
if not os.path.exists(USER_FILES_DIR):
    os.makedirs(USER_FILES_DIR)

st.set_page_config(page_title="Personal Learning Assistant", page_icon="📚", layout="wide")

load_dotenv()

API_KEY = os.getenv("API_KEY")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"credentials.json"
project_id = "gemini-practice-sai"
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

# def generate_speech(text):
#     client = texttospeech.TextToSpeechClient()
#     synthesis_input = texttospeech.SynthesisInput(text=text)
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="en-US",
#         ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
#     )
#     audio_config = texttospeech.AudioConfig(
#         audio_encoding=texttospeech.AudioEncoding.MP3
#     )
#     response = client.synthesize_speech(
#         input=synthesis_input, voice=voice, audio_config=audio_config
#     )
#     return response.audio_content

def extract_text_from_pdf(pdf_file):
    if isinstance(pdf_file, str):
        # If pdf_file is a string (file path), open the file and extract text
        return extract_text(pdf_file)
    else:
        # If pdf_file is a file object, save it temporarily and extract text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_file_path = temp_file.name
        
        text = extract_text(temp_file_path)
        os.remove(temp_file_path)
        return text

def interact_with_gemini(model_id, prompt_text):
    model_instance = vertexai.generative_models.GenerativeModel(model_id)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=vertexai.generative_models.GenerationConfig(
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
    model_instance = vertexai.generative_models.GenerativeModel(model_id)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=vertexai.generative_models.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
            response_mime_type="application/json",
            response_schema=quiz_response_schema,
        )
    )
    print("Raw response from Gemini:", response.text)  # Debug print

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return response.text

def interact_with_gemini_evaluation_theory(model_id, prompt_text):
    model_instance = vertexai.generative_models.GenerativeModel(model_id)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=vertexai.generative_models.GenerationConfig(
            temperature=0.6,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
            response_mime_type="application/json",
            response_schema=evaluation_theory_response_schema,
        )
    )
    print("Raw response from Gemini:", response.text)  # Debug print

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return response.text

def interact_with_gemini_theory(model_id, prompt_text):
    model_instance = vertexai.generative_models.GenerativeModel(model_id)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=vertexai.generative_models.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
            response_mime_type="application/json",
            response_schema=theory_response_schema,
        )
    )
    print("Raw response from Gemini:", response.text)  # Debug print

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return response.text

def interact_with_gemini_summariser(model_id, prompt_text):
    model_instance = vertexai.generative_models.GenerativeModel(model_id)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=vertexai.generative_models.GenerationConfig(
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

def save_summary_to_pdf(summary, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in summary.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)


# New function to load user files
def load_user_files(username):
    user_dir = os.path.join(USER_FILES_DIR, username)
    if not os.path.exists(user_dir):
        return []
    return [f for f in os.listdir(user_dir) if f.endswith('.pdf')]

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
                st.experimental_rerun()
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
            
            pdf_text = extract_text_from_pdf(pdf_file)
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
        st.title("Chat with PDF Content")
        
        pdf = st.file_uploader("Upload a PDF to chat", type="pdf", key="chat_pdf_upload")

        if pdf:
            st.subheader("PDF Viewer")
            binary_data = pdf.getvalue()
            pdf_viewer(input=binary_data, width=1400, height=800)  # Adjust height as needed

        st.subheader("Chat Interface")

        if pdf:
            st.write("Processing PDF...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf.read())
                temp_file_path = temp_file.name
            
            try:
                loader = PyPDFLoader(temp_file_path)
                text_documents = loader.load()
                os.remove(temp_file_path)
                st.session_state['text_documents'] = text_documents
                st.success(f"PDF processed successfully! {len(text_documents)} pages loaded.")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.session_state['text_documents'] = None

        if 'text_documents' in st.session_state and st.session_state['text_documents']:
            text_documents = st.session_state['text_documents']
            
            # Splitting the text documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            try:
                split_text_documents = text_splitter.split_documents(text_documents)
                
                # Add page numbers to metadata
                for i, doc in enumerate(split_text_documents):
                    doc.metadata['page'] = i + 1
                
                db = Chroma.from_documents(split_text_documents, OpenAIEmbeddings())
                
                user_question = st.text_input("Enter your question about the PDF:")
                if user_question:
                    st.write("Processing your question...")
                    prompt = ChatPromptTemplate.from_template(
                        '''
                        Answer the following question based on the provided context.
                        Think step by step and explain the process in detail.
                        You get a bonus point if you say it correctly.
                        Include the page numbers where you found the information in your answer.
                        <context>
                            {context}
                        </context>

                        question: {input}
                        '''
                    )
                    document_chain = create_stuff_documents_chain(
                        ChatOpenAI(model="gpt-3.5-turbo"),
                        prompt,
                    )
                    retriever = db.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    response = retrieval_chain.invoke({'input': user_question})

                    st.write("Answer:")
                    st.write(response['answer'])

                    st.write("Source Pages:")
                    source_pages = set(doc.metadata['page'] for doc in response['context'])
                    st.write(f"Information retrieved from page(s): {', '.join(map(str, sorted(source_pages)))}")

            except Exception as e:
                st.error(f"Error processing text documents: {str(e)}")
        else:
            st.warning("Please upload a PDF to start chatting.")

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
            st.experimental_rerun()

else:
    st.warning("Please enter your credentials to access the app.")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.title("Navigation")