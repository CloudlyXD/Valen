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
        raise  # Re-raise the exception to halt execution

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
                    message_id SERIAL PRIMARY KEY,
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
        conn.rollback()  # Rollback changes if an error occurs
        raise

# --- Database Connection and Table Creation --- 
try:
    conn = get_db_connection()  # Establish the connection
    create_tables(conn)  # Create tables (if they don't exist)
except Exception as e:
    print(f"❌ Application startup failed: {e}")
    exit(1) # Exit the application if database setup fails


genai.configure(api_key=api_key_queue[0])  # Initial API key
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
    if not text:  # Handle None or empty string
        return ""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    return text

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


        # --- Database Operations ---
        conn = get_db_connection()  # Get a database connection
        with conn.cursor() as cursor:
            # 1. Insert the user (if they don't exist)
            cursor.execute("INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING", (user_id,))

            # 2. Insert the chat
            cursor.execute("INSERT INTO chats (chat_id, user_id, title) VALUES (%s, %s, %s)", (chat_id, user_id, title))

            # 3. Insert the user's message
            cursor.execute(
                "INSERT INTO messages (chat_id, user_id, role, content) VALUES (%s, %s, %s, %s)",
                (chat_id, user_id, "user", first_message)
            )
            # 4. Insert the bot's reply
            cursor.execute(
                "INSERT INTO messages (chat_id, user_id, role, content) VALUES (%s, %s, %s, %s)",
                (chat_id, user_id, "bot", bot_reply)
            )

        conn.commit()  # Commit the changes
        conn.close()

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
           "gemini-2.0-flash",  # Keep your preferred model
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 8192, # Use your preferred max tokens
            }
        )

        # --- Database Operations (LOAD HISTORY) ---
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY timestamp ASC",
                (chat_id,)
            )
            chat_history = [f"{row[0]}: {row[1]}" for row in cursor.fetchall()]

                    # --- CONTEXT WINDOW LIMIT ---
        chat_history = chat_history[-100:]  # Keep only the last 100 entries

        # Append user message to history *before* generating prompt
        chat_history.append(f"User: {user_message}")
        prompt = f"{PERSONALITY_PROMPT}\n\n" + "\n".join(chat_history) + "\nAI:"

        response = model.generate_content(prompt)

        # Check if response.text exists and is not empty
        if response.text and not response.text.isspace():
            bot_reply = remove_markdown(response.text.strip())
        else:
            bot_reply = "I'm sorry, I couldn't generate a response. Please try again."

        # Remove "Valen:" prefix if present
        bot_reply = bot_reply.replace("Valen:", "").strip()


        # --- Database Operations (SAVE NEW MESSAGES) ---
        with conn.cursor() as cursor: #Reusing the connection
            # Insert the user's message
            cursor.execute(
                "INSERT INTO messages (chat_id, user_id, role, content) VALUES (%s, %s, %s, %s)",
                (chat_id, user_id, "user", user_message)
            )
            # Insert the bot's reply
            cursor.execute(
                "INSERT INTO messages (chat_id, user_id, role, content) VALUES (%s, %s, %s, %s)",
                (chat_id, user_id, "bot", bot_reply)
            )

        conn.commit()
        conn.close()
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
            history = []
            for row in cursor.fetchall():
                message_id, role, content, timestamp = row  # Unpack all values
                history.append({
                    "message_id": message_id,  # Include message_id
                    "role": role,
                    "content": content,
                    "timestamp": timestamp.isoformat()
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

@app.get("/chats")
async def get_chats(request: Request):
    # Extract user_id from query parameters
    user_id = request.query_params.get("user_id", "unknown_user")
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:  # Using DictCursor for cleaner code
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

# --- New API route to update message content ---
@app.post("/update_message")
async def update_message(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")
    message_id = data.get("message_id")
    new_content = data.get("new_content")

    if not chat_id or not message_id or not new_content:
        return {"error": "Missing chat_id, message_id, or new_content"}

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Update the message content
            cursor.execute(
                "UPDATE messages SET content = %s WHERE chat_id = %s AND message_id = %s",
                (new_content, chat_id, message_id)
            )
            rows_updated = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if rows_updated > 0:
            return {"success": True}
        else:
            return {"success": False, "error": "Message not found or no changes were made"}

    except Exception as e:
        print(f"Error updating message: {e}")
        return {"error": "Failed to update message", "success": False}

# --- New API route to regenerate response after message edit ---
@app.post("/regenerate_response")
async def regenerate_response(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "unknown_user")
    chat_id = data.get("chat_id")
    message_id = data.get("message_id")
    edited_content = data.get("edited_content")

    if not chat_id or not message_id:
        return {"error": "Missing chat_id or message_id"}

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

        # --- Get chat history up to the edited message ---
        conn = get_db_connection()
        
        # First, find the position of the edited message
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY timestamp ASC",
                (chat_id,)
            )
            all_messages = cursor.fetchall()
            
            # Get all messages and create chat history
            chat_history = []
            for i, (role, content) in enumerate(all_messages):
                # If we've found the edited message, use the new content
                if role == "user" and i < len(all_messages) - 1 and all_messages[i+1][0] == "bot":
                    if str(message_id) == str(i):  # This is simplified - you'd need to match actual message_id
                        chat_history.append(f"User: {edited_content}")
                    else:
                        chat_history.append(f"{role}: {content}")
                else:
                    chat_history.append(f"{role}: {content}")
            
            # Find the position after the edited message (where bot response should be)
            edited_pos = -1
            for i, (role, content) in enumerate(all_messages):
                if str(message_id) == str(i) and role == "user":
                    edited_pos = i
                    break
            
            if edited_pos == -1:
                return {"error": "Edited message not found"}
            
            # Get just the history up to the edited message to regenerate the response
            prompt_history = chat_history[:edited_pos+1]
            prompt_history = prompt_history[-100:]  # Context window limit
            
            # Generate new response
            prompt = f"{PERSONALITY_PROMPT}\n\n" + "\n".join(prompt_history) + "\nAI:"
            response = model.generate_content(prompt)
            
            if response.text and not response.text.isspace():
                new_bot_reply = remove_markdown(response.text.strip())
            else:
                new_bot_reply = "I'm sorry, I couldn't generate a response. Please try again."
            
            # Remove "Valen:" prefix if present
            new_bot_reply = new_bot_reply.replace("Valen:", "").strip()
            
            # Update the bot's response in the database
            # This assumes the bot response is right after the edited message
            cursor.execute(
                "UPDATE messages SET content = %s WHERE chat_id = %s AND role = 'bot' AND message_id > %s ORDER BY message_id ASC LIMIT 1",
                (new_bot_reply, chat_id, message_id)
            )
            
            rows_updated = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if rows_updated > 0:
            return {"success": True, "response": new_bot_reply}
        else:
            return {"success": False, "error": "Bot response not found or couldn't be updated"}

    except Exception as e:
        print(f"Error regenerating response: {e}")
        return {"error": f"Failed to regenerate response: {str(e)}", "success": False}

# --- New API route to edit message ---
@app.post("/edit_message")
async def edit_message(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    chat_id = data.get("chat_id")
    message_id = data.get("message_id")  # This should be the message ID
    new_content = data.get("new_content")

    if not user_id or not chat_id or not message_id or new_content is None:
        return {"error": "Missing user_id, chat_id, message_id, or new_content", "success": False}

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # First get the timestamp of message, that user wants to edit.
            cursor.execute(
                "SELECT timestamp FROM messages WHERE chat_id = %s and role = 'user' and content = %s ORDER BY timestamp",
                (chat_id, user_message)
            )
            
            result = cursor.fetchone()
            
            # If there is no such message
            if result is None:
              return {"error": "Message not found or not updated.", "success": False}

            timestamp = result[0]

            # Get the count of messages from that timestamp
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_id = %s AND timestamp <= %s",
                (chat_id, timestamp)
            )
            count = cursor.fetchone()[0]

            # Now we know the message id of database.
            cursor.execute(
                "UPDATE messages SET content = %s WHERE message_id = %s",
                (new_content, count)
            )

            if cursor.rowcount == 0: # Check if updated
              return {"error": "Message not found or not updated.", "success": False}

        conn.commit()
        conn.close()
        return {"success": True}

    except Exception as e:
        print(f"Error updating message: {e}")
        return {"error": "Failed to update message", "success": False}

# --- Run the API ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
