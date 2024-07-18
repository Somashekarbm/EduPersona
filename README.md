# Personal Learning Assistant

Welcome to the **Personal Learning Assistant** project! This project, created for a national-level hackathon, aims to provide users with an AI-powered tool to assist with personalized learning experiences. The app features PDF content extraction, quiz generation, theory Q&A, PDF summarization, and interactive chat with PDF content.

## Features

- **User Authentication:** Signup and login functionality with password hashing.
- **PDF Upload and Viewing:** Upload PDF files, view them, and extract text content.
- **Quiz Generation:** Generate multiple-choice quizzes based on uploaded PDF content.
- **Theory Q&A:** Generate and answer theoretical questions based on PDF content.
- **PDF Summarization:** Summarize PDF content into concise, readable formats.
- **Interactive Chat:** Chat interface to interact with PDF content.

## Getting Started

If you get any error related to Response Schema after running the project , you might want to run the below code :

```bash
pip install --upgrade google-cloud-aiplatform
```

### Prerequisites

- Python 3.7+
- Install required packages:

```bash
pip install langchain_openai langchain_community streamlit fpdf pdfminer.six streamlit_pdf_viewer sqlite3 werkzeug google-cloud texttospeech google-auth google-auth-oauthlib google-auth-httplib2
```

-install the packages present in the requirements.txt as well using the command:

```bash
pip install -r requirements.txt
```

### Setting Up

1. **Clone the Repository**

```bash
git clone https://github.com/Somashekarbm/EduPersona.git
cd personal-learning-assistant
```

2. **Set Up Environment Variables**

Create a `.env` file with your API key:

```env
API_KEY=your_openai_api_key
```

3. **Initialize Database**

```python
import sqlite3

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

init_db()
```

### Running the App

```bash
streamlit run app.py
```

## Project Structure

```
personal-learning-assistant/
│
├── main.py               # Main Streamlit app
├── requirements.txt     # Required Python packages
├── .env                 # Environment variables
├── user_files/          # Directory to store user files
└── users.db             # SQLite database for user authentication
└── credentials.json     #credentials for your GCP project
```

## Usage

### User Authentication

- **Signup:** Create a new user account.
- **Login:** Login with existing credentials.

### PDF Upload and Viewing

1. Navigate to "PDF Upload" section.
2. Upload a PDF file.
3. The PDF content will be extracted and displayed.

### Generate Quiz

1. After uploading a PDF, go to "Generate Quiz".
2. Specify the number of questions.
3. Click "Generate Quiz" to create a quiz based on the PDF content.

### Take Quiz

1. Navigate to "Take Quiz" after generating a quiz.
2. Answer the multiple-choice questions.
3. Submit answers to get a score and review key concepts for incorrect answers.

### Theory Q&A

1. Go to "Theory Q&A" after uploading a PDF.
2. Generate theoretical questions.
3. Answer the questions and get feedback on your responses.

### PDF Summary

1. Navigate to "PDF Summary".
2. Upload a PDF to summarize.
3. Generate and view the summary.

### Chat and View

1. Upload a PDF in the "Chat and View" section.
2. Interact with the PDF content through the chat interface.

## Contributions

Feel free to contribute to this project by submitting issues or pull requests. 

