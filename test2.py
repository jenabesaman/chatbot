from gpt4all import GPT4All
import torch

# Verify CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

# Specify the path to your .gguf model
model_path = "E:/Workarea/ai-chatbots/models/Dorna-Llama3-8B-Instruct.Q8_0.gguf"  # Replace with your actual model path

# Initialize the GPT4All model with CUDA device
model = GPT4All(model_path, device="cuda")  # Use "cuda" instead of "gpu"
prompt="hi i born in 1996. now is 2024. how old am i? just give me a single number"
#prompt = "I was born in 1996. The current year is 2024. How old am I? Answer with a single number only."
# Generate output
output = model.generate(prompt,max_tokens=50000,temp=0.01,top_k=10)
print(output)
