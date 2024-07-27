from transformers import GPT2LMHeadModel
import torch
import pandas as pd

# Load the model
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)

# Print model structure
print(model)

# Get the list of all parameter names
param_names = list(model.state_dict().keys())

# Print the first few parameter names
print("\nFirst few parameter names:")
for name in param_names[:10]:
    print(name)

# Choose a specific layer to examine (e.g., first layer of the transformer)
layer_name = "transformer.h.0.attn.c_attn.weight"  # This should be correct for GPT-2 Medium
weights = model.state_dict()[layer_name].cpu().numpy()

# Convert to DataFrame
df = pd.DataFrame(weights)

# Save to CSV
csv_file = "gpt2_medium_weights.csv"
df.to_csv(csv_file, index=False)

print(f"\nSaved weights to {csv_file}")

# Print shape of the weights
print(f"Shape of the weights: {weights.shape}")