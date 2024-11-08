import os

from gpt4all import GPT4All
import torch

# Verify CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your CUDA installation.")


# Specify the path to your .gguf model
model_path = "E:/Workarea/ai-chatbots/models/Dorna-Llama3-8B-Instruct.Q8_0.gguf"  # Replace with your actual model path

# Initialize the GPT4All model with CUDA device
model = GPT4All(model_path, device="cuda")  # Use "cuda" instead of "gpu"

def read_basic_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    return None

# Specify the path to the folder containing the text file
folder_path = "./data"  # Replace with your actual folder path
file_name = "test1.txt"  # Replace with your actual file name
file_path = os.path.join(folder_path, file_name)

# Read basic data from the text file
basic_data = read_basic_data(file_path)
if basic_data:
    print(f"Loaded basic data from {file_name}.")
else:
    print(f"No basic data file found at {file_path}.")



print("Welcome to the chat! Type 'exit' to end the session.")

# Use the chat session to handle context efficiently
with model.chat_session() as session:
    # If basic data exists, send it to the model first
    if basic_data:
        special_prompt = f"Based on this basic data, I will provide information: {basic_data}"
        output = session.generate(special_prompt, max_tokens=500, temp=0.01, top_k=10)
        print("AI has processed the basic data.")

    while True:
        # Get user input
        user_input = input("You: ")

        # Check for exit condition
        if user_input.lower() == "exit":
            print("Ending chat session. Goodbye!")
            break



        # Generate output within the chat session
        output = session.generate(user_input, max_tokens=500, temp=0.01, top_k=10)
        # Filter out unwanted sections
        filtered_response = output.split("#### Explanation:")[0].strip()
        filtered_response = filtered_response.split("#### Next question?")[0].strip()
        filtered_response = filtered_response.split("#### Answer:")[0].strip()
        filtered_response = filtered_response.split("#### End of explanation.")[0].strip()
        filtered_response = filtered_response.split("#### Final Answer:")[0].strip()
        filtered_response = filtered_response.split("### Session:Generate;")[0].strip()

        # Print the model's response
        print(f"AI: {filtered_response}")