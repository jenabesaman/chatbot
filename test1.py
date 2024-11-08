from gpt4all import GPT4All
import torch

# Verify CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

# Specify the path to your .gguf model
model_path = "E:/Workarea/ai-chatbots/models/Dorna-Llama3-8B-Instruct.Q8_0.gguf"  # Replace with your actual model path

# Initialize the GPT4All model with CUDA device
model = GPT4All(model_path, device="cuda")  # Use "cuda" instead of "gpu"

# Generate output
output = model.generate("hi i born in 1996. now is 2024. how old am i? just give me a single number", max_tokens=20)
print(output)


# Example of generating text
# input_text = "What is the capital of France?"
#
# # Generate output
# with model.chat_session() as chat:
#     response = chat.generate(input_text)
#     print(response)