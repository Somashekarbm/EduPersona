import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
import json
from dotenv import load_dotenv
from pdfminer.high_level import extract_text


load_dotenv()
# Set your Vertex AI project ID
project_id = "gemini-practice-sai"

# Initialize Vertex AI with your project ID and location
vertexai.init(project=project_id, location="us-central1")

# Load the response schema from JSON file
response_schema = 'quiz_generator_response-schema.json'


def pdf2text(filepath):
    return extract_text(filepath)


def load_json(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data


response_schema_file = load_json(response_schema)

# Function to prompt the generative model with a given prompt and schema


def quiz_generator(model, prompt, response_schema):
    model_instance = GenerativeModel(model)

    response = model_instance.generate_content(
        prompt,
        generation_config=GenerationConfig(
            response_mime_type="application/json", response_schema=response_schema_file
        ),
    )

    return response


pdf_filepath = r"notes/SE-M1.pdf"
text = pdf2text(pdf_filepath)
number_of_quiz_questions = 5


# Example prompt text (modify as needed)

# the prompt is generating in such a fashion that the first option is always the correct answer. We need to change it. We need to randomise the correct answers position.

prompt_text = f"Generate {number_of_quiz_questions} quiz questions for the given context: {text} with options and correct answers from the options generated. The question should start from being easy then medium and later hard difficulty."

# Example usage:
# Replace 'gemini-1.5-pro-001' with your actual model identifier
quizes = quiz_generator(
    'gemini-1.5-pro-001', prompt_text, response_schema
)

quizes_output_file = 'generated_quizzes.json'
with open(quizes_output_file, 'w') as f:
    json.dump(json.loads(quizes.text), f, indent=4)
