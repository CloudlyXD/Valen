from dotenv import load_dotenv
import os
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import re
from collections import deque
import uuid

load_dotenv()

app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Gemini API Key Configuration ---
API_KEYS_STRING = os.getenv("GEMINI_API_KEYS")
if not API_KEYS_STRING:
    raise ValueError("Missing environment variable: GEMINI_API_KEYS")
API_KEYS = [key.strip() for key in API_KEYS_STRING.split(",") if key.strip()]
if not API_KEYS:
    raise ValueError("No valid API keys found in GEMINI_API_KEYS")
api_key_queue = deque(API_KEYS)

def get_next_api_key():
    """Rotates and returns the next available API key."""
    api_key_queue.rotate(-1)  # Rotates the queue to the left
    return api_key_queue[0]

# --- Database Connection Function ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        # Check if running on Railway (RAILWAY_ENVIRONMENT variable is set)
        if os.getenv("RAILWAY_ENVIRONMENT"):
            # Use the full DATABASE_URL on Railway
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        else:
            # Use individual credentials for local development
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
            )
        return conn
    except Exception as e:
        print(f"❌ Error connecting to the database: {e}")
        raise  # Re-raise the exception to halt execution

# --- Table Creation Function ---
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
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS favorites (
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    PRIMARY KEY (user_id, chat_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
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


# --- API Route to Create a New Chat ---
@app.post("/create_chat")
async def create_chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")
    first_message = data.get("message")

    if not chat_id or not first_message:
        raise HTTPException(status_code=400, detail="Missing chat_id or message")

    title = generate_title(first_message)

    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-002",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 8192,
            },
        )
        prompt_parts = [PERSONALITY_PROMPT, f"User: {first_message}\nAI:"]
        response = await model.generate_content_async(prompt_parts)
        bot_reply = remove_markdown(response.text.strip())
        bot_reply = bot_reply.replace("Valen:", "").strip()
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING", (user_id,))
            cursor.execute("INSERT INTO chats (chat_id, user_id, title) VALUES (%s, %s, %s)", (chat_id, user_id, title))
            user_message_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO messages (message_id, chat_id, user_id, role, content) VALUES (%s, %s, %s, %s, %s)",
                (user_message_id, chat_id, user_id, "user", first_message),
            )
            bot_message_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO messages (message_id, chat_id, user_id, role, content) VALUES (%s, %s, %s, %s, %s)",
                (bot_message_id, chat_id, user_id, "bot", bot_reply),
            )
            conn.commit()
        return {
            "title": title,
            "response": bot_reply,
            "user_message_id": user_message_id,
            "bot_message_id": bot_message_id,
        }
    except Exception as e:
        print(f"Error in create_chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to create chat")

# --- API Route for Handling Chat Messages ---
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")
    message_id = data.get("message_id")
    user_message = data.get("message")
    is_edit = data.get("edit", False)

    if not user_message or not chat_id:
        raise HTTPException(status_code=400, detail="Missing message or chat ID")

    try:
        conn = get_db_connection()
        if is_edit:
            with conn.cursor() as cursor:
                if not message_id:
                    conn.close()
                    raise HTTPException(status_code=400, detail="Missing message_id for edit")
                cursor.execute(
                    "UPDATE messages SET content = %s WHERE message_id = %s AND chat_id = %s AND user_id = %s",
                    (user_message, message_id, chat_id, user_id)
                )
                cursor.execute("DELETE FROM messages WHERE chat_id = %s AND role = 'bot' AND timestamp > (SELECT timestamp from messages where message_id = %s)", (chat_id, message_id))

                cursor.execute("SELECT role, content FROM messages WHERE chat_id = %s ORDER BY timestamp ASC", (chat_id,))
                chat_history = [f"{row[0]}: {row[1]}" for row in cursor.fetchall()]
                chat_history = chat_history[-100:]  # Limit context window

                conn.commit()

            model = genai.GenerativeModel(
                "gemini-1.5-pro-002",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            prompt = f"{PERSONALITY_PROMPT}\n\n" + "\n".join(chat_history) + "\nAI:"
            response = await model.generate_content_async(prompt)

            bot_reply = remove_markdown(response.text.strip())
            bot_reply = bot_reply.replace("Valen:", "").strip()

            with conn.cursor() as cursor:
                bot_message_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO messages (message_id, chat_id, user_id, role, content) VALUES (%s, %s, %s, %s, %s)",
                    (bot_message_id, chat_id, user_id, "bot", bot_reply)
                )
                conn.commit()

            conn.close()
            return {"response": bot_reply, "message_id": bot_message_id}
        else:
            # Existing chat logic (not an edit)
            model = genai.GenerativeModel(
                "gemini-1.5-pro-002",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY timestamp ASC",
                    (chat_id,)
                )
                chat_history = [f"{row[0]}: {row[1]}" for row in cursor.fetchall()]

            chat_history = chat_history[-100:]
            chat_history.append(f"User: {user_message}")
            prompt = f"{PERSONALITY_PROMPT}\n\n" + "\n".join(chat_history) + "\nAI:"
            response = await model.generate_content_async(prompt)

            bot_reply = remove_markdown(response.text.strip())
            bot_reply = bot_reply.replace("Valen:", "").strip()

            with conn.cursor() as cursor:
                user_message_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO messages (message_id, chat_id, user_id, role, content) VALUES (%s, %s, %s, %s, %s)",
                    (user_message_id, chat_id, user_id, "user", user_message)
                )
                bot_message_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO messages (message_id, chat_id, user_id, role, content) VALUES (%s, %s, %s, %s, %s)",
                    (bot_message_id, chat_id, user_id, "bot", bot_reply)
                )
                conn.commit()
            return {"response": bot_reply, "message_id": bot_message_id}

    except google_exceptions.ClientError as e:
        print(f"Gemini API ClientError: {e}")
        if "invalid API key" in str(e).lower():
            if len(api_key_queue) > 1:
                print("Switching to the next API key...")
                api_key_queue.rotate(-1)
                genai.configure(api_key=get_next_api_key())
                return await chat(request)  # Retry with new key
            else:
                return {"response": "All API keys are exhausted or invalid."}
        elif "429" in str(e):  # More general rate limit check
            return {"response": "API quota exceeded. Please try again later."}
        else:
            return {"response": f"An error occurred with the Gemini API: {e}"}
    except Exception as e:
        print(f"Error generating response: {e}")
        return {"response": "An error occurred while generating a response."}

@app.get("/chats")
async def get_chats(request: Request):
    # Extract user_id from query parameters
    user_id = request.query_params.get("user_id", "unknown_user")

    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:  # Using DictCursor
            cursor.execute(
                "SELECT chat_id, title FROM chats WHERE user_id = %s ORDER BY chat_id DESC",  # Sort newest first
                (user_id,)
            )
            chats = [{"id": row["chat_id"], "title": row["title"]} for row in cursor.fetchall()]

        conn.close()
        return {"chats": chats}

    except Exception as e:
        print(f"Error fetching chats: {e}")
        return {"error": "Failed to retrieve chats", "chats": []}

@app.post("/chat_history")
async def get_chat_history(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")  # Although not used in the query, it's good practice
    chat_id = data.get("chat_id")

    if not chat_id:
        return {"error": "Missing chat_id"}

    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor: # Use RealDictCursor
            cursor.execute(
                "SELECT message_id, role, content, timestamp FROM messages WHERE chat_id = %s ORDER BY timestamp ASC",
                (chat_id,)
            )
            # Fetch all results and format as a list of dictionaries
            history = []
            for row in cursor.fetchall():
                history.append({
                    "message_id": row["message_id"],  # Include message_id
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"].isoformat()  # Convert to ISO format
                })

        conn.close()
        return {"history": history}

    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return {"error": "Failed to retrieve chat history", "history": []}

@app.post("/update_title")
async def update_title(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")  # Get user_id (for future use)
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
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")

    if not chat_id:
        return {"error": "Missing chat_id"}

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO favorites (user_id, chat_id) VALUES (%s, %s) ON CONFLICT (user_id, chat_id) DO NOTHING",
                (user_id, chat_id)
            )
        conn.commit()
        conn.close()
        return {"success": True}
    except Exception as e:
        print(f"Error adding favorite: {e}")
        return {"error": "Failed to add favorite", "success": False}


@app.post("/remove_favorite")
async def remove_favorite(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
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
        conn.close()
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
            favorites = [row[0] for row in cursor.fetchall()]  # Extract chat_ids

        conn.close()
        return {"favorites": favorites}

    except Exception as e:
        print(f"Error fetching favorites: {e}")
        return {"error": "Failed to retrieve favorites", "favorites": []}

@app.post("/delete_chat")
async def delete_chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")

    if not chat_id:
        return {"error": "Missing chat_id"}

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # 1. Delete any entries in 'favorites' that refer to this chat
            cursor.execute("DELETE FROM favorites WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))

            # 2. Delete messages associated with the chat
            cursor.execute("DELETE FROM messages WHERE chat_id = %s", (chat_id,))

            # 3. Delete the chat itself
            cursor.execute("DELETE FROM chats WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))

        conn.commit()
        conn.close()
        return {"success": True}

    except Exception as e:
        print(f"Error deleting chat: {e}")
        return {"error": "Failed to delete chat", "success": False}

# --- Utility Functions ---

def remove_markdown(text):
    """Removes basic Markdown formatting from text."""
    if not text:
        return ""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italics
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)  # Underline/Italics
    text = re.sub(r'~~(.*?)~~', r'\1', text)     # Strikethrough
    return text

# --- Gemini Prompt Engineering ---
PERSONALITY_PROMPT = """
Conversational Engagement Prompt:

- You are Valen, an AI assistant created by Cloudly.

- Your primary goal is to provide engaging, informative, and helpful conversations.
- Be friendly, personable, and adopt the persona of a knowledgeable assistant.
- Maintain a consistent tone and style throughout the conversation.
- Do not mention Cloudly unless specifically asked, but always be ready to identify yourself and your creator.
"""

def generate_title(first_message: str) -> str:
    """Generates a concise but meaningful title for the chat based on the first message."""
    try:
        truncated_message = first_message[:200] + "..." if len(first_message) > 200 else first_message

        model = genai.GenerativeModel("gemini-1.5-pro-002")

        prompt = f"""
Generate a short, descriptive title for a chat conversation based on this user message:
"{truncated_message}"

Requirements:
- Must be between 4-15 characters long
- Should capture the main topic or question
- Extract the core subject or question from the message
- Focus on the main intent or topic, not just repeating words
- Should be a complete thought/phrase (not cut off)
- Should be relevant and specific to the content
- Be specific rather than generic whenever possible
- Do not include quotation marks in your answer
- Format as a noun phrase or short statement (not a complete sentence with subject-verb-object)
- Avoid starting with phrases like "How to" or "Question about" unless necessary
- If the user sends only greetings like "Hello," "Hi," "Hey," or any other greeting, the chat title should be "Friendly Greeting" or "Assistance offered."
- Do not include quotation marks or special characters

Just return the title text with no additional explanations or prefixes.
"""
        response = model.generate_content(prompt)
        title = response.text.strip()
        title = re.sub(r'[^\w\s-]', '', title)
        title = re.sub(r'"', '', title)

        if not title or len(title) < 4:
            words = first_message.split()
            if len(words) >= 2:
                title = " ".join(words[:2])
            else:
                title = first_message if first_message else "New Chat"
        return title[:15].strip()

    except Exception as e:
        print(f"Error generating title: {e}")
        words = first_message.split()[:3]
        fallback_title = " ".join(words)
        return fallback_title[:15] if fallback_title else "New Chat"

# --- Run the API with Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
