from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
import os
import re
from collections import deque
from google.api_core import exceptions as google_exceptions
import requests

CHAT_HISTORY_FILE = "web_chat_history.json"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini API Keys ---
API_KEYS_STRING = os.getenv("GEMINI_API_KEYS")
if not API_KEYS_STRING:
    raise ValueError("Missing environment variable: GEMINI_API_KEYS")
API_KEYS = [key.strip() for key in API_KEYS_STRING.split(",") if key.strip()]
if not API_KEYS:
    raise ValueError("No valid API keys found in GEMINI_API_KEYS")

api_key_queue = deque(API_KEYS)

def get_next_api_key():
    """Rotates and returns the next available API key."""
    api_key_queue.rotate(-1)
    return api_key_queue[0]

genai.configure(api_key=api_key_queue[0])  # Initial API key

# --- Personality Prompt ---
PERSONALITY_PROMPT = """
The assistant is Valen, created by Cloudly (An Individual).

"""

# --- Helper Functions ---
def remove_markdown(text):
    """Removes basic Markdown formatting."""
    if not text:  # Add this check to handle None or empty string
        return ""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    return text

def load_chat_history(user_id, chat_id):
    """Loads chat history for a specific user and chat."""
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(str(user_id), {}).get(str(chat_id), [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_chat_history(user_id, chat_id, history):
    """Saves chat history for a specific user and chat."""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        if str(user_id) not in data:
            data[str(user_id)] = {}

        data[str(user_id)][str(chat_id)] = history
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Error saving chat history: {e}")


def generate_title(first_message: str) -> str:
    """Generates a concise title for the chat based on the first message."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"Generate a concise and descriptive title (maximum 15 characters) for a chat conversation based on this user message: '{first_message}'"
        response = model.generate_content(prompt)
        title = response.text.strip()
        # Basic sanitization and length limit
        title = re.sub(r'[^\w\s-]', '', title)  # Remove special characters
        
        # Ensure title is not empty
        if not title or title.isspace():
            return "New Chat"
            
        return title[:15]  # Limit to 15 characters
    except Exception as e:
        print(f"Error generating title: {e}")
        return "New Chat"  # Fallback title

# --- New API route to create chat ---
@app.post("/create_chat")
async def create_chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")  # Must be provided by the frontend
    first_message = data.get("message")

    if not chat_id or not first_message:
        return {"error": "Missing chat_id or message"}

    # Generate the title *before* saving any history
    title = generate_title(first_message)

    # Respond with the title *and* the initial bot reply
    try:
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        prompt = f"{PERSONALITY_PROMPT}\n\nUser: {first_message}\nAI:" # Initial Prompt
        response = model.generate_content(prompt)
        bot_reply = remove_markdown(response.text.strip()) if response.text else "I'm sorry, I couldn't generate a response. Please try again."

        # Now that we have title, save the initial messages in history:
        chat_history = [f"User: {first_message}", f"AI: {bot_reply}"]
        save_chat_history(user_id, chat_id, chat_history)

        return {"title": title, "response": bot_reply}  # Return title and AI reply
    except Exception as e:
        print("Error on create_chat", e)
        return {"title": "New Chat", "response": "I'm sorry, I couldn't process your request. Please try again."}


# --- API Route for Web Requests ---
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")  # Get chat ID from frontend
    user_message = data.get("message")

    if not user_message or not chat_id: # Added chat_id validation
        return {"error": "No message or chat ID provided"}

    try:
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )

        chat_history = load_chat_history(user_id, chat_id)

        chat_history.append(f"User: {user_message}")
        chat_history = chat_history[-100:]

        prompt = f"{PERSONALITY_PROMPT}\n\n{chr(10).join(chat_history)}\nAI:"
        response = model.generate_content(prompt)
        
        # Check if response.text exists and is not empty
        if response.text and not response.text.isspace():
            bot_reply = remove_markdown(response.text.strip())
        else:
            bot_reply = "I'm sorry, I couldn't generate a response. Please try again."

        chat_history.append(f"AI: {bot_reply}")
        save_chat_history(user_id, chat_id, chat_history)

        return {"response": bot_reply}


    except google_exceptions.ClientError as e:
        print(f"Gemini API ClientError: {e}")
        if "invalid API key" in str(e).lower():
            if len(api_key_queue) > 1:
                print("Switching to the next API key...")
                api_key_queue.rotate(-1)
                genai.configure(api_key=get_next_api_key())
                return await chat(request)  # Retry with new key
            else:
                return {"response": "Due to unexpected capacity constraints, I am unable to respond to your message. Please try again soon."}
        elif "quota exceeded" in str(e).lower():
            if len(api_key_queue) > 1:
                print("Switching to the next API key (quota exceeded)...")
                api_key_queue.rotate(-1)
                genai.configure(api_key=get_next_api_key())
                return await chat(request)  # Retry with new key
            else:
                return {"response": "Due to unexpected capacity constraints, I am unable to respond to your message. Please try again soon."}
        else:
            return {"response": "An error occurred while processing your request."}

    except Exception as e:
        print(f"Error generating response: {e}")
        return {"response": "An error occurred while generating a response."}


# --- Run the API ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
