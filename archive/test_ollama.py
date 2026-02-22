import dspy

# Configure DSPy to use Ollama (using OpenAI-compatible endpoint)
ollama_model = dspy.LM(
    model='ollama/llama3.2',
    api_base='http://localhost:11434',
    api_key='ollama',  # Dummy key, Ollama doesn't need it
    max_tokens=500
)
dspy.configure(lm=ollama_model)

# Simple test
print("Testing DSPy with Ollama...")
response = ollama_model("What is the capital of France?")
print(f"Response: {response}")
