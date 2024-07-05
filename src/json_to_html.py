import vertexai
from vertexai.generative_models import GenerativeModel
import json
from dotenv import load_dotenv

load_dotenv()

# Set your Vertex AI project ID
project_id = "gemini-practice-sai"
vertexai.init(project=project_id, location="us-central1")

simplified_content_file = "simplified_contents.json"

# Load the simplified content from JSON file


def load_json(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data


simplified_content = load_json(simplified_content_file)

# Function to prompt the generative model with a given prompt


def prompt(model, prompt_text):
    model_instance = GenerativeModel(model)
    response = model_instance.generate_content(
        prompt_text,
    )
    return response.text

# Function to convert JSON content to HTML


def json_to_html(json_file, model):
    prompt_text = f"Here is the json content: {json_file} convert this into beautiful and easily understandable, ready html with appropriate styles"
    return prompt(model, prompt_text)

# Function to write HTML content to a file


def write_html_file(html_content, output_file):
    with open(output_file, 'w') as f:
        f.write(html_content)


# Example usage:
if __name__ == "__main__":
    html_result = json_to_html(simplified_content, "gemini-1.5-pro-001",)
    output_file = "generated_content.html"
    write_html_file(html_result, output_file)
    print(f"HTML content written to {output_file}")
