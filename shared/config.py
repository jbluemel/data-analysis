import os
import dspy

def configure_llm(provider='claude', temperature=0.3):
    """
    Configure DSPy with specified LLM provider.
    
    Args:
        provider: 'claude' or 'llama'
        temperature: 0.0-1.0 (lower = more factual)
    """
    
    if provider == 'claude':
        model = dspy.LM(
            model='anthropic/claude-sonnet-4-20250514',
            api_key=os.environ.get('ANTHROPIC_API_KEY'),
            max_tokens=1000,
            temperature=temperature
        )
    elif provider == 'llama':
        model = dspy.LM(
            model='ollama/llama3.1:8b',
            api_base='http://localhost:11434',
            api_key='ollama',
            max_tokens=1000,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'claude' or 'llama'")
    
    dspy.configure(lm=model)
    print(f"Configured DSPy with {provider} (temp={temperature})")
    return model
