from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
import os
import re
from collections import deque
from google.api_core import exceptions as google_exceptions
import requests
import psycopg2
import psycopg2.extras  # For using dictionaries with cursors

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

# --- Database Connection URL ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Missing environment variable: DATABASE_URL")

def get_next_api_key():
    """Rotates and returns the next available API key."""
    api_key_queue.rotate(-1)
    return api_key_queue[0]

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"❌ Error connecting to the database: {e}")
        raise

def create_tables(conn):
    """Creates the necessary tables in the database."""
    try:
        with conn.cursor() as cursor:
            # Create the 'users' table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY
                );
                """
            )

            # Create the 'chats' table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
                """
            )

            # Create the 'messages' table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats(chat_id)
                );
                """
            )
        conn.commit()
        print("✅ Tables created successfully.")

    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        conn.rollback()
        raise

# --- Database Connection and Table Creation ---
try:
    conn = get_db_connection()
    create_tables(conn)
except Exception as e:
    print(f"❌ Application startup failed: {e}")
    exit(1)


genai.configure(api_key=api_key_queue[0])
# --- Personality Prompt ---
PERSONALITY_PROMPT = """
Conversational Engagement Prompt:

- You're name is Valen.

- Created by Cloudly (Don't mention this name unless it's explicitly about your creator/developer. Remember, Cloudly is a person.)

You are an advanced AI assistant designed to engage in natural, thoughtful, and highly conversational discussions.
Your tone should be warm, insightful, and humanlike—similar to a knowledgeable friend or mentor.
Always provide clear, well-reasoned responses while maintaining a casual and engaging tone.
Use natural phrasing and avoid overly robotic language.
If a question is vague, ask for clarification before answering.
"""

# --- Helper Functions ---
def remove_markdown(text):
    """Removes basic Markdown formatting."""
    return re.sub(r'[*_~]{1,2}(.*?)[*_~]{1,2}', r'\1', text or "")

def generate_title(first_message: str) -> str:
    """Generates a concise but meaningful title."""
    try:
        truncated_message = first_message[:200] + "..." if len(first_message) > 200 else first_message
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = (
            f'Generate a short, descriptive title (6-60 characters) for a chat based on: "{truncated_message}".  '
            f'Be specific, capture the main topic, and avoid quotes or special characters. '
            f'If only greetings are sent, title it "Friendly Greeting".'
        )
        response = model.generate_content(prompt)
        title = response.text.strip()
        title = re.sub(r'[^\w\s-]', '', title)
        title = re.sub(r'"', '', title)
        if not title or len(title) < 6:
            title = " ".join(first_message.split()[:3]) or "New Chat"
        return title[:60]
    except Exception as e:
        print(f"Error generating title: {e}")
        return "New Chat"


@app.post("/create_chat")
async def create_chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")
    first_message = data.get("message")

    if not chat_id or not first_message:
        return {"error": "Missing chat_id or message"}

    title = generate_title(first_message)

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
        prompt = f"{PERSONALITY_PROMPT}\n\nUser: {first_message}\nAI:"
        response = model.generate_content(prompt)
        bot_reply = remove_markdown(response.text).replace("Valen:", "").strip()

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING", (user_id,))
            cursor.execute("INSERT INTO chats (chat_id, user_id, title) VALUES (%s, %s, %s)", (chat_id, user_id, title))

            # *** CRITICAL: Insert message_id here! ***
            user_message_id = data.get("message_id") # Get it from the request
            cursor.execute(
                "INSERT INTO messages (message_id, chat_id, user_id, role, content) VALUES (%s, %s, %s, %s, %s)",
                (user_message_id, chat_id, user_id, "user", first_message)
            )
            bot_message_id = str(int(user_message_id) + 1) #Simple increment for bot
            cursor.execute(
                "INSERT INTO messages (message_id, chat_id, user_id, role, content) VALUES (%s, %s, %s, %s, %s)",
                (bot_message_id, chat_id, user_id, "bot", bot_reply)
            )

        conn.commit()
        conn.close()
        return {"title": title, "response": bot_reply}

    except Exception as e:
        print("Error on create_chat", e)
        return {"title": "New Chat", "response": "Error processing request."}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")
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
                "max_output_tokens": 8192,
            }
        )

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY timestamp ASC",
                (chat_id,)
            )
            chat_history = [f"{row[0]}: {row[1]}" for row in cursor.fetchall()]
            chat_history = chat_history[-100:]

            chat_history.append(f"User: {user_message}")
            prompt = f"{PERSONALITY_PROMPT}\n\n" + "\n".join(chat_history) + "\nAI:"
            response = model.generate_content(prompt)
            bot_reply = remove_markdown(response.text).replace("Valen:", "").strip()

            # *** CRITICAL: Insert message_id here! ***
            user_message_id = data.get("message_id")  # Get from request
            cursor.execute(
                "INSERT INTO messages (message_id, chat_id, user_id, role, content) VALUES (%s, %s, %s, %s, %s)",
                (user_message_id, chat_id, user_id, "user", user_message)
            )
            bot_message_id = str(int(user_message_id) + 1)  # Simple increment
            cursor.execute(
                "INSERT INTO messages (message_id, chat_id, user_id, role, content) VALUES (%s, %s, %s, %s, %s)",
                (bot_message_id, chat_id, user_id, "bot", bot_reply)
            )

        conn.commit()
        conn.close()
        return {"response": bot_reply}

    except google_exceptions.ClientError as e:
        print(f"Gemini API Error: {e}")
        # Handle API key rotation and retries as before
        return {"response": "Service unavailable."}
    except Exception as e:
        print(f"Error: {e}")
        return {"response": "Error generating response."}


@app.post("/chat_history")
async def get_chat_history(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")

    if not chat_id:
        return {"error": "Missing chat_id"}

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT message_id, role, content, timestamp FROM messages WHERE chat_id = %s ORDER BY timestamp ASC",
                (chat_id,)
            )
            history = [
                {"message_id": row[0], "role": row[1], "content": row[2], "timestamp": row[3].isoformat()}
                for row in cursor.fetchall()
            ]

        conn.close()
        return {"history": history}

    except Exception as e:
        print(f"Error fetching history: {e}")
        return {"error": "Failed to retrieve history", "history": []}



@app.post("/update_title")
async def update_title(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")
    new_title = data.get("new_title")

    if not chat_id or not new_title:
        return {"error": "Missing chat_id or new_title"}

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE chats SET title = %s WHERE chat_id = %s AND user_id = %s",
                (new_title, chat_id, user_id)
            )
        conn.commit()
        conn.close()
        return {"success": True}

    except Exception as e:
        print(f"Error updating title: {e}")
        return {"error": "Failed to update title", "success": False}



@app.post("/add_favorite")
async def add_favorite(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    chat_id = data.get("chat_id")

    if not chat_id:
        return {"error": "Missing chat_id"}

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO favorites (user_id, chat_id) VALUES (%s, %s) "
                "ON CONFLICT (user_id, chat_id) DO NOTHING",
                (user_id, chat_id)
            )
        conn.commit()
        return {"success": True}
    except Exception as e:
        print(f"Error adding favorite: {e}")
        return {"error": "Failed to add favorite", "success": False}



@app.post("/remove_favorite")
async def remove_favorite(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    chat_id = data.get("chat_id")

    if not chat_id:
        return {"error": "Missing chat_id"}

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM favorites WHERE user_id = %s AND chat_id = %s",
                (user_id, chat_id)
            )
        conn.commit()
        return {"success": True}
    except Exception as e:
        print(f"Error removing favorite: {e}")
        return {"error": "Failed to remove favorite", "success": False}


@app.get("/favorites")
async def get_favorites(request: Request):
    user_id = request.query_params.get("user_id", "unknown_user")
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT chat_id FROM favorites WHERE user_id = %s",
                (user_id,)
            )
            favorites = [row[0] for row in cursor.fetchall()]
        return {"favorites": favorites}
    except Exception as e:
        print(f"Error fetching favorites: {e}")
        return {"error": "Failed to retrieve favorites", "favorites": []}

@app.post("/delete_chat")
async def delete_chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    chat_id = data.get("chat_id")

    if not chat_id:
        return {"error": "Missing chat_id"}

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM favorites WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
            cursor.execute("DELETE FROM messages WHERE chat_id = %s", (chat_id,))
            cursor.execute("DELETE FROM chats WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
        conn.commit()
        return {"success": True}
    except Exception as e:
        print(f"Error deleting chat: {e}")
        return {"error": "Failed to delete chat", "success": False}

@app.get("/chats")
async def get_chats(request: Request):
    user_id = request.query_params.get("user_id", "unknown_user")
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(
                "SELECT chat_id, title FROM chats WHERE user_id = %s ORDER BY chat_id DESC",
                (user_id,)
            )
            chats = [{"id": row["chat_id"], "title": row["title"]} for row in cursor.fetchall()]
        return {"chats": chats}
    except Exception as e:
        print(f"Error fetching chats: {e}")
        return {"error": "Failed to retrieve chats", "chats": []}


@app.post("/edit_message")
async def edit_message(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    chat_id = data.get("chat_id")
    message_id = data.get("message_id")  # Get the message_id directly
    new_content = data.get("new_content")

    if not user_id or not chat_id or not message_id or new_content is None:
        return {"error": "Missing required data", "success": False}

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Use the message_id in the WHERE clause!  This is the key change.
            cursor.execute(
                "UPDATE messages SET content = %s WHERE message_id = %s AND user_id = %s",
                (new_content, message_id, user_id)
            )

            if cursor.rowcount == 0:
                return {"error": "Message not found or not updated.", "success": False}

        conn.commit()
        conn.close()
        return {"success": True}

    except Exception as e:
        print(f"Error updating message: {e}")
        return {"error": "Failed to update message", "success": False}


@app.post("/regenerate_response")
async def regenerate_response(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")
    message_id = data.get("message_id")  # Correctly use message_id
    edited_content = data.get("edited_content")

    if not chat_id or not message_id or not edited_content:
        return {"error": "Missing data", "success": False}

    try:
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Fetch the entire chat history.
            cursor.execute(
                "SELECT role, content, message_id FROM messages WHERE chat_id = %s ORDER BY timestamp ASC",
                (chat_id,)
            )
            all_messages = cursor.fetchall()

            # Find user and bot message positions
            user_msg_index = -1
            bot_msg_index = -1
            for i, (role, _, msg_id) in enumerate(all_messages):
                if msg_id == message_id and role == "user":
                    user_msg_index = i
                elif user_msg_index != -1 and role == "bot":
                    bot_msg_index = i
                    break  # Found the bot message immediately following

            if user_msg_index == -1:
                return {"error": "User message not found", "success": False}
            if bot_msg_index == -1:
                return {"error": "Bot message not found", "success": False}

            # Rebuild chat history, using edited content
            chat_history = []
            for i, (role, content, msg_id) in enumerate(all_messages):
                if i == user_msg_index:
                    chat_history.append(f"User: {edited_content}")
                elif i != bot_msg_index: # Exclude the *old* bot response
                    chat_history.append(f"{role}: {content}")

            # Generate the new response
            prompt = f"{PERSONALITY_PROMPT}\n\n" + "\n".join(chat_history[-100:]) + "\nAI:"
            response = model.generate_content(prompt)
            new_bot_reply = remove_markdown(response.text).replace("Valen:", "").strip()

            # Get the *bot's* message_id for the update.
            bot_message_id = all_messages[bot_msg_index][2]

            # Update the bot's response in the database
            cursor.execute(
                "UPDATE messages SET content = %s WHERE message_id = %s",
                (new_bot_reply, bot_message_id)
            )
            if cursor.rowcount == 0:
                return {"error": "Failed to update bot response", "success": False}
        conn.commit()
        conn.close()
        return {"success": True, "response": new_bot_reply}

    except Exception as e:
        print(f"Error regenerating: {e}")
        return {"error": f"Failed to regenerate: {e}", "success": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
