import json
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from dotenv import load_dotenv

load_dotenv()

# Set your Vertex AI project ID
project_id = "gemini-practice-sai"
vertexai.init(project=project_id, location="us-central1")

# Load the generated quiz and user answers from JSON files
quiz_file = 'generated_quizzes.json'
user_answers_file = 'user_answers.json'


def load_json(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data


quiz_data = load_json(quiz_file)
user_answers = load_json(user_answers_file)

# Function to prompt the generative model with a given prompt


def prompt(model, prompt_text):
    model_instance = GenerativeModel(model)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=GenerationConfig(
            response_mime_type="application/json")
    )
    return response.text


# Validate user answers
results = []
for question_data in quiz_data:
    question_number = question_data['question-number']
    question = question_data['question']
    correct_answer = question_data['answer']
    user_answer_data = next(
        (ua for ua in user_answers if ua['question-number'] == question_number), None)
    if user_answer_data:
        user_answer = user_answer_data['answer']
        result = {
            "question-number": question_number,
            "question": question,
            "correct_answer": correct_answer,
            "user_answer": user_answer,
            "is_correct": user_answer == correct_answer
        }
        results.append(result)

# Generate a validation prompt for the LLM
validation_prompt = f"Validate the following user answers based on the correct answers:\n{json.dumps(results, indent=2)}"

# Example usage with Vertex AI's generative model
model_name = 'gemini-1.5-pro-001'
validation_response = prompt(model_name, validation_prompt)

# Print the validation response
print(validation_response)

# Optionally, save the validation results
with open('validation_results.json', 'w') as f:
    json.dump(results, f, indent=4)
