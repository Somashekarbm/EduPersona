import json
import streamlit as st

# Load the generated quiz from JSON file
json_file = 'generated_quizzes.json'


def load_json(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data


quiz_data = load_json(json_file)

st.title("Quiz App")

user_answers = []
for idx, q in enumerate(quiz_data):
    st.subheader(f"Question {q['question-number']}: {q['question']}")
    options = q['options']
    user_answer = st.radio("Choose an answer:", options, key=idx)
    user_answers.append({
        "question-number": q['question-number'],
        "question": q['question'],
        "answer": user_answer
    })

if st.button('Submit Answers'):
    st.write("Answers submitted successfully!")
    with open('user_answers.json', 'w') as f:
        json.dump(user_answers, f, indent=4)
