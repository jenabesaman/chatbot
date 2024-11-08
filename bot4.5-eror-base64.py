from flask import Flask, request, jsonify
import os
import requests
from gpt4all import GPT4All
import torch
from PyPDF2 import PdfReader
from docx import Document
import base64
import filetype

app = Flask(__name__)

# Verify CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

# Specify the path to your .gguf model
model_path = "E:/Workarea/ai-chatbots/models/Dorna-Llama3-8B-Instruct.Q8_0.gguf"  # Replace with your actual model path

# Initialize the GPT4All model with CUDA device
model = GPT4All(model_path, device="cuda")


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


# Function to process base64 data
def process_base64_data(base64_data):
    try:
        binary_data = base64.b64decode(base64_data)
        kind = filetype.guess(binary_data)
        if kind:
            if kind.extension == 'txt':
                return binary_data.decode('utf-8').strip()
            elif kind.extension == 'pdf':
                return read_pdf_from_binary(binary_data)
            elif kind.extension == 'docx':
                return read_word_from_binary(binary_data)
        else:
            print("Unsupported file type or invalid base64 data.")
            return None
    except Exception as e:
        print(f"Failed to process base64 data: {e}")
        return None


# Function to read text from a PDF binary
def read_pdf_from_binary(binary_data):
    try:
        from io import BytesIO
        reader = PdfReader(BytesIO(binary_data))
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        print(f"Failed to read PDF from binary: {e}")
        return None


# Function to read text from a Word binary
def read_word_from_binary(binary_data):
    try:
        from io import BytesIO
        doc = Document(BytesIO(binary_data))
        text = ''
        for para in doc.paragraphs:
            text += para.text + '\n'
        return text.strip()
    except Exception as e:
        print(f"Failed to read Word from binary: {e}")
        return None


# Path to the folder containing the basic data files
folder_path = "./data"  # Replace with your actual folder path
file_name = "test1.txt"  # Replace with your actual file name
file_path = os.path.join(folder_path, file_name)

# Read basic data from the specified file
basic_data = process_base64_data(file_path) if os.path.exists(file_path) else None

# Dictionary to store session contexts
session_contexts = {}


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    if not data or 'prompt' not in data or 'session_id' not in data:
        return jsonify({'error': 'Invalid input, please provide prompt and session_id in JSON format.'}), 400

    prompt = data['prompt']
    session_id = data['session_id']

    # Check for base64 data from user input
    if 'base_64' in data and data['base_64']:
        user_base64_data = process_base64_data(data['base_64'])
        if user_base64_data is None:
            return jsonify({'error': 'The provided base64 data is not correct or unsupported.'}), 400
        else:
            prompt = f"{user_base64_data} {prompt}"

    # Initialize session context if not present
    if session_id not in session_contexts:
        session_contexts[session_id] = []

        # Only add basic data if base_64 is not provided
        if 'base_64' not in data or not data['base_64']:
            # Add basic data to the context for the first request
            if basic_data:
                session_contexts[session_id].append(f"keep this basic data, I will ask questions: {basic_data}")

            # Fetch data from the URL only once per session
            url = "https://172.16.40.170/data"  # Replace with your actual URL
            token = "your_token_here"  # Replace with your actual token
            url_data = fetch_data_from_url(url, token)
            if url_data:
                # Check if URL data is base64 and process it
                url_base64_data = process_base64_data(url_data)
                if url_base64_data:
                    session_contexts[session_id].append(f"data from URL: {url_base64_data}")
                else:
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