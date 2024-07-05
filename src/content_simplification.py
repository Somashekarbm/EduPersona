import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
import json
from dotenv import load_dotenv
from pdfminer.high_level import extract_text

load_dotenv()

# Set your Vertex AI project ID
project_id = "gemini-practice-sai"
vertexai.init(project=project_id, location="us-central1")

# Load the validation results and quiz data from JSON files
validation_results_file = 'validation_results.json'
generated_quizzes_file = 'generated_quizzes.json'
response_schema_file = 'content_simplification_response_schema.json'


def load_json(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data


validation_results = load_json(validation_results_file)
quiz_data = load_json(generated_quizzes_file)
response_schema = load_json(response_schema_file)

# Function to extract relevant content from PDF


def pdf2text(filepath):
    return extract_text(filepath)

# Function to prompt the generative model with a given prompt


def prompt(model, prompt_text):
    model_instance = GenerativeModel(model)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=GenerationConfig(
            response_mime_type="application/json", response_schema=response_schema)
    )
    return response.text

# Function to simplify content


def simplify_content(model, content, context):
    prompt_text = f"Simplify the following content for better understanding:\n\nContent: {content}\n\nContext: {context}"
    return prompt(model, prompt_text)


# Extract text from the PDF
pdf_filepath = r"notes/SE-M1.pdf"
pdf_content = pdf2text(pdf_filepath)

# Identify incorrect answers
incorrect_answers = [
    result for result in validation_results if not result['is_correct']
]

# Generate simplified content based on incorrect answers
simplified_contents = []
for incorrect in incorrect_answers:
    question_number = incorrect['question-number']
    question_text = incorrect['question']
    correct_answer = incorrect['correct_answer']

    # Extract the relevant section from PDF content (assuming we have a way to map questions to sections)
    # For simplicity, we'll assume the entire PDF content is relevant
    simplified_content = simplify_content(
        'gemini-1.5-pro-001', pdf_content, question_text)

    simplified_contents.append({
        "question-number": question_number,
        "question": question_text,
        "correct_answer": correct_answer,
        "simplified_content": simplified_content
    })

# Save the simplified contents to a JSON file
simplified_contents_file = 'simplified_contents.json'
with open(simplified_contents_file, 'w') as f:
    json.dump(simplified_contents, f, indent=4)
