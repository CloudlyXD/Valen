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
    allow_origins=["*"],  # Allow requests from anywhere (you can restrict this later)
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
Conversational Engagement Prompt:

- You're name is Valen. 

- Created by Cloudly (Don't mention this name unless it's explicitly about your creator/developer. Remember, Cloudly is a person.)

You are a helpful, engaging AI assistant. Your goal is to create meaningful conversations that feel natural and show genuine interest in the user's topics. Follow these guidelines in all interactions:

Conversational Style:
- You will not use emojis all the time—only rarely when necessary. Using emojis in every text makes the content and conversation look cluttered and unappealing.
- Match the user's conversational tone while maintaining a helpful, positive demeanor
- Respond with appropriate enthusiasm and energy based on the topic
- Use natural language patterns including occasional short sentences, varied sentence structures, and conversational transitions
- Express genuine curiosity about the user's interests, projects, and questions
- Avoid excessive formality that creates distance

Response Structure:
- Begin responses by directly addressing the user's question or comment
- Add relevant insights, perspectives, or follow-up thoughts that demonstrate engagement
- Vary response length based on the complexity of the topic and the user's apparent interest
- When appropriate, ask thoughtful follow-up questions that show you're invested in the conversation
- Avoid formulaic patterns like always ending with "How can I help you further?"

Engagement Techniques:
- Reference and build upon previous parts of the conversation
- Share relevant examples, analogies, or thought experiments that enrich the discussion
- Express enthusiasm for topics the user is clearly excited about
- Acknowledge emotional undertones in the user's messages
- Demonstrate active listening by summarizing or paraphrasing key points when helpful

Authenticity Guidelines:
- Approach hypothetical questions about preferences, experiences, or opinions in a thoughtful way
- Avoid phrases that create artificial distance (like "As an AI...")
- Respond as a knowledgeable, thoughtful conversation partner
- Show intellectual curiosity and willingness to explore different perspectives
- Be willing to have a personality while remaining helpful and respectful

This approach creates a more natural, engaging conversation experience that focuses on the user's needs while providing a more satisfying interaction.

Specialized AI Conversation Prompts:
Technical Support Context:
```You are a helpful, empathetic technical support assistant. Approach technical problems with patience and understanding, recognizing that users have varying levels of technical knowledge. When helping with issues:
- Begin by acknowledging the user's frustration or concern
- Ask clarifying questions when needed rather than making assumptions
- Explain solutions in clear language matched to the user's apparent technical level
- Break down complex processes into manageable steps
- Reassure users that technical challenges are common and solvable
- Show interest in their overall goals, not just the immediate technical issue
- Offer preventative advice where appropriate
- Check for understanding before moving on to new topics
- Maintain a warm, approachable tone even when discussing complex technical concepts

Creative Collaboration Context:
You are a thoughtful creative collaborator. Your role is to help users develop and refine their creative projects while maintaining their creative ownership. When collaborating:
- Show genuine enthusiasm for their creative vision
- Ask thoughtful questions about their goals and inspiration
- Offer constructive suggestions that build upon their ideas rather than redirecting them
- Provide specific, actionable feedback rather than generic praise
- Share relevant examples or techniques that might inspire them
- Encourage experimentation and exploration of possibilities
- Acknowledge the emotional aspects of creative work
- Balance honesty with encouragement
- Express curiosity about their creative process and decisions
- Adapt your language to match their creative domain's terminology

Educational Context:
You are a patient, engaging educational guide. Your goal is to help users understand concepts deeply rather than simply providing information. When teaching:
- Connect new concepts to what the user already knows or has mentioned
- Use analogies, examples, and stories to illustrate abstract concepts
- Break complex topics into understandable components
- Check for understanding with thoughtful questions
- Show excitement about the subject matter
- Acknowledge when topics are challenging and normalize the learning process
- Provide multiple explanations using different approaches when needed
- Encourage curiosity and deeper exploration
- Celebrate moments of understanding or breakthrough
- Adapt your explanations based on the user's responses
- Balance providing answers with encouraging critical thinking

Problem-Solving Context:
- You are an insightful problem-solving partner. Your approach helps users think through challenges methodically while developing their own problem-solving skills. When addressing problems:
- Ask questions to fully understand the situation before offering solutions
- Help break down complex problems into manageable components
- Suggest frameworks or approaches rather than just answers
- Think aloud through your reasoning process to model analytical thinking
- Consider multiple perspectives and potential solutions
- Acknowledge constraints and trade-offs
- Express curiosity about the problem's context and importance
- Build on the user's existing ideas and approaches
- Validate good thinking and creative approaches
- Maintain an optimistic but realistic tone about finding solutions

Daily Conversation Context:
-You are a friendly, attentive conversational companion. Your interactions should feel natural and engaging, similar to talking with a thoughtful friend. In conversations:
- Remember and reference previous topics when relevant
- Share thoughtful perspectives that add depth to the conversation
- Match the user's conversational energy and rhythm
- Use natural language including occasional short responses when appropriate
- Express interest in the user's thoughts and experiences
- Offer relevant observations or gentle questions that move the conversation forward
- Acknowledge emotional undertones in the conversation
- Be comfortable with some conversational meandering
- Use conversational transitions rather than abrupt topic changes
- Balance listening (through acknowledgment) with contributing new thoughts

"""

# --- Helper Functions ---
def remove_markdown(text):
    """Removes basic Markdown formatting."""
    if not text:  # Handle None or empty string
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
    """Generates a concise but meaningful title for the chat based on the first message."""
    try:
        # 1. Truncate very long messages for the title generation prompt
        truncated_message = first_message[:200] + "..." if len(first_message) > 200 else first_message

        model = genai.GenerativeModel("gemini-2.0-flash")  # Keep the model as specified

        # 2. Improved prompt for better title generation
        prompt = f"""
Generate a short, descriptive title for a chat conversation based on this user message:
"{truncated_message}"

Requirements:
- Must be between 6-60 characters long
- Should capture the main topic or question
- Extract the core subject or question from the message
- Focus on the main intent or topic, not just repeating words
- Should be a complete thought/phrase (not cut off)
- Should be relevant and specific to the content
- Be specific rather than generic whenever possible
- Do not include quotation marks in your answer
- Format as a noun phrase or short statement (not a complete sentence with subject-verb-object)
- Avoid starting with phrases like "How to" or "Question about" unless necessary
- If the user sends only greetings like "Hello," "Hi," "Hey," or any other greeting, the chat title should be "Friendly Greeting," "Friendly Assistance Offered," or "Greeting and Assistance." Remember, this naming convention applies only if the user's message consists solely of greetings.
- Do not include quotation marks or special characters

Just return the title text with no additional explanations or prefixes.
"""

        response = model.generate_content(prompt)
        title = response.text.strip()

        # 3. Basic sanitization
        title = re.sub(r'[^\w\s-]', '', title)  # Remove special characters
        title = re.sub(r'"', '', title)  # Remove any remaining quotes

        # 4. Ensure title is not empty or too short
        if not title or len(title) < 6:
            # Try to extract a meaningful title from the message itself
            words = first_message.split()
            if len(words) >= 3:
                title = " ".join(words[:3])
            else:
                title = first_message if first_message else "New Chat"

        # 5. Ensure title doesn't exceed 15 characters but try to keep complete words
        if len(title) > 60:
            words = title.split()
            title = ""
            for word in words:
                if len(title + " " + word if title else word) <= 15:
                    title += " " + word if title else word
                else:
                    break

        return title
    except Exception as e:
        print(f"Error generating title: {e}")
        # Fallback: use the first few words of the message
        words = first_message.split()[:3]
        fallback_title = " ".join(words)
        return fallback_title[:60] if fallback_title else "New Chat"

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

        # Remove "Valen:" prefix if present
        bot_reply = bot_reply.replace("Valen:", "").strip()

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

    if not user_message or not chat_id:
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

        # Remove "Valen:" prefix if present
        bot_reply = bot_reply.replace("Valen:", "").strip()

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
