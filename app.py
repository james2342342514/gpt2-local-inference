import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, render_template, request

app = Flask(__name__)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and tokenizer
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def generate_text(prompt, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        generated_text = generate_text(prompt)
        return render_template('index.html', prompt=prompt, generated_text=generated_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)