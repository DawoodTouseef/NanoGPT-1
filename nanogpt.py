import torch
from gpt import GPTLanguageModel,encode,decode
# Define your PyTorch model
model = GPTLanguageModel()

# Load the saved model state dictionary
model.load_state_dict(torch.load('nanogpt.pth'))
# Put model in evaluation mode
model.eval()

# Prepare input data for prediction
inp=input("Enter the message:")
input_data = encode(inp)
input_tensor = torch.tensor(input_data, dtype=torch.long)

# Make a prediction
with torch.no_grad():
    output_tensor = model(input_tensor)
    output = output_tensor.numpy()

# Use the output for further processing or display
print(decode(m.generate(output, max_new_tokens=500)[0].tolist()))
