import os
import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from tavily import TavilyClient
from tavily import TavilyClient


# Load environment variables from .env file
load_dotenv('.env')

# Retrieve API keys from environment variables
hf_api_key = os.getenv("HF_API_KEY")
tvly_api_key = os.getenv("TVLY_API_KEY")

# Check if the keys are loaded properly (optional, but useful for debugging)
if not hf_api_key:
    raise ValueError("Hugging Face API key is missing")
if not tvly_api_key:
    raise ValueError("Tavily API key is missing")

# Now you can use these API keys in your client initialization


# Load environment variables
load_dotenv()

# Initialize clients
client = InferenceClient(
    provider="novita",
    api_key=os.getenv("HF_TOKEN"),
    bill_to="huggingface"
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def get_web_context(query):
    """
    Get relevant web search results using Tavily
    """
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=3
        )
        
        # Format the search results
        context = "Web Search Results:\n\n"
        for result in response['results']:
            context += f"Title: {result['title']}\n"
            context += f"URL: {result['url']}\n"
            context += f"Content: {result['content']}\n\n"
        
        return context
    except Exception as e:
        return f"Error getting web context: {str(e)}"

def chat(message, history):
    """
    Process chat messages using Hugging Face's Inference Provider with web context
    """
    try:
        # Get web context
        web_context = get_web_context(message)
        
        # Format the conversation history
        messages = []
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        
        # Add system message with web context
        messages.append({
            "role": "system",
            "content": f"You are a helpful AI assistant. Use the following web search results to inform your response:\n\n{web_context}"
        })
        
        # Add user message
        messages.append({"role": "user", "content": message})

        # Get streaming response from the model
        stream = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            stream=True
        )

        # Stream the response
        partial_message = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                partial_message += chunk.choices[0].delta.content
                yield partial_message

    except Exception as e:
        yield f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="DeepSearch - AI Search Assistant") as demo:
    
    chatbot = gr.ChatInterface(
        fn=chat,
        examples=[
            "What is the capital of France?",
            "Explain quantum computing in simple terms",
            "Write a short poem about artificial intelligence"
        ],
        title="DeepSearch",
        description="Ask me anything, powered by Hugging Face Inference Providers",
        theme=gr.themes.Soft()
    )

if __name__ == "__main__":
    demo.launch(share=True) 
