import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# Load the model and tokenizer
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="tf")
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "summarize Barack Obama's presidency"
generated_text = generate_text(prompt)
print(f"Prompt: {prompt}")
print(f"Generated text: {generated_text}")