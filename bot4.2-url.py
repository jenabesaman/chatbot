from flask import Flask, request, jsonify
import os
import requests
from gpt4all import GPT4All
import torch

app = Flask(__name__)

# Verify CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

# Specify the path to your .gguf model
model_path = "E:/Workarea/ai-chatbots/models/Dorna-Llama3-8B-Instruct.Q8_0.gguf"  # Replace with your actual model path

# Initialize the GPT4All model with CUDA device
model = GPT4All(model_path, device="cuda")

# Specify the path to the folder containing the text file
folder_path = "./data"  # Replace with your actual folder path
file_name = "test1.txt"  # Replace with your actual file name
file_path = os.path.join(folder_path, file_name)


# Function to fetch data from a URL with a token
def fetch_data_from_url(url, token, timeout=5):
    headers = {'Authorization': f'Bearer {token}'}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.exceptions.ConnectTimeout:
        print("Connection timed out. Proceeding without data from the URL.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}. Proceeding without data from the URL.")
        return None


# Read basic data from the text file
def read_basic_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    return None


basic_data = read_basic_data(file_path)

# Dictionary to store session contexts
session_contexts = {}


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    if not data or 'prompt' not in data or 'session_id' not in data:
        return jsonify({'error': 'Invalid input, please provide prompt and session_id in JSON format.'}), 400

    prompt = data['prompt']
    session_id = data['session_id']

    # Initialize session context if not present
    if session_id not in session_contexts:
        session_contexts[session_id] = []
        # Add basic data to the context for the first request
        if basic_data:
            session_contexts[session_id].append(f"keep this basic data, I will ask questions: {basic_data}")

        # Fetch data from the URL only once per session
        url = "https://172.16.40.170/data"  # Replace with your actual URL
        token = "your_token_here"  # Replace with your actual token
        url_data = fetch_data_from_url(url, token)
        if url_data:
            session_contexts[session_id].append(f"data from URL: {url_data}")

    # Add user input to the session context
    session_contexts[session_id].append(prompt)

    # Generate output considering the entire session context
    with model.chat_session() as session:
        context = " ".join(session_contexts[session_id])
        output = session.generate(context, max_tokens=500, temp=0.01, top_k=10)

        # Filter out unwanted sections
        filtered_response = output.split("#### Explanation:")[0].strip()
        filtered_response = filtered_response.split("#### Next question?")[0].strip()
        filtered_response = filtered_response.split("#### Answer:")[0].strip()
        filtered_response = filtered_response.split("#### End of explanation.")[0].strip()
        filtered_response = filtered_response.split("#### Final Answer:")[0].strip()
        filtered_response = filtered_response.split("### Session:Generate;")[0].strip()

        # Add the model's response to the session context
        session_contexts[session_id].append(filtered_response)

        return jsonify({'response': filtered_response})


if __name__ == '__main__':
    app.run(debug=True)