"""
app.py - Complete Flask application with Dia, Polly, ElevenLabs, and OpenAI
integration
Provides a Flask web application with Supabase credentials, multiple TTS options
(ElevenLabs, Dia, Polly),
OpenAI for response generation, and improved error handling.
"""
from flask import Flask, render_template, request, jsonify, session
import os
import uuid
import json
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

# Import our custom modules
define_mods = ["dia_tts", "openai_therapist"]
from dia_tts import generate_and_save, generate_silent_audio
from openai_therapist import SarcasticTherapistAI

# Import Amazon Polly integration (if available)
try:
    from amazon_polly import generate_with_polly

    POLLY_AVAILABLE = True
except ImportError:
    POLLY_AVAILABLE = False
    logging.warning("Amazon Polly integration not available")

# Import ElevenLabs integration (if available)
try:
    from elevenlabs_tts import generate_elevenlabs_audio

    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logging.warning("ElevenLabs TTS integration not available")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app")

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
app.static_folder = 'static'

# Supabase initialization
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

# Log environment variables status
debug_vars = [("SUPABASE_URL", supabase_url), ("SUPABASE_KEY", supabase_key),
              ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
              ("DIA_API_KEY", os.getenv("DIA_API_KEY")),
              ("ELEVENLABS_API_KEY", os.getenv("ELEVENLABS_API_KEY"))]
for var, val in debug_vars:
    logger.info(f"{var} present: {bool(val)}")

# Default user preferences - DIA IS NOW PERMANENT DEFAULT
DEFAULT_PREFERENCES = {
    "tone_preferences": '{"sarcasm_level": 5}',  # JSON string format
    "voice": "Rishi (en-IN)",
    "speech_rate": 1.0,
    "use_dia": True  # Always True - Dia is permanent default
}


@app.before_request
def initialize_session():
    """Initialize session variables if they don't exist."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    if 'preferences' not in session:
        session['preferences'] = DEFAULT_PREFERENCES.copy()
    if 'conversation_history' not in session:
        session['conversation_history'] = []


@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')


@app.route('/api/preferences', methods=['GET', 'POST'])
def handle_preferences():
    """Get or update user preferences."""
    if request.method == 'POST':
        try:
            data = request.get_json()
            logger.info(f"Updating preferences: {data}")
            # Update only provided preferences
            for key, value in data.items():
                if key in session['preferences']:
                    # Force use_dia to always be True
                    if key == 'use_dia':
                        session['preferences'][key] = True
                    else:
                        session['preferences'][key] = value
            return jsonify({"success": True, "preferences": session['preferences']})
        except Exception as e:
            logger.error(f"Error updating preferences: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 400
    else:
        return jsonify({"preferences": session.get('preferences', {})})


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history."""
    return jsonify({"history": session.get('conversation_history', [])})


@app.route('/api/message', methods=['POST'])
def handle_message():
    """Process a new message from the user with enhanced context handling."""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        if not user_message.strip():
            return jsonify({"error": "Message cannot be empty"}), 400

        logger.info(f"Received message: '{user_message}'")

        # Add user message to history with additional metadata
        message_id = str(uuid.uuid4())
        timestamp = data.get('timestamp', None)  # Client can provide timestamp

        # Create a rich user message object
        user_message_obj = {
            "role": "user",
            "content": user_message,
            "id": message_id,
            "timestamp": timestamp
        }

        # Get current conversation history or initialize if none
        if 'conversation_history' not in session:
            session['conversation_history'] = []

        session['conversation_history'].append(user_message_obj)

        # Get current preferences
        preferences = session.get('preferences', DEFAULT_PREFERENCES)

        # Initialize therapist with improved error handling
        therapist = None
        try:
            therapist = SarcasticTherapistAI(
                supabase_url=supabase_url,
                supabase_key=supabase_key
            )

            # Set sarcasm level with multiple fallback options
            try:
                # First, try to parse tone_preferences if it's a JSON string
                if 'tone_preferences' in preferences and isinstance(preferences['tone_preferences'], str):
                    try:
                        tone_prefs = json.loads(preferences['tone_preferences'])
                        sarcasm_level = tone_prefs.get('sarcasm_level', 5)

                        # Try different methods to set sarcasm level
                        if hasattr(therapist, 'set_sarcasm_level'):
                            therapist.set_sarcasm_level(sarcasm_level)
                            logger.info(f"Set sarcasm level to {sarcasm_level} using set_sarcasm_level method")
                        elif hasattr(therapist, 'sarcasm_level'):
                            therapist.sarcasm_level = sarcasm_level
                            logger.info(f"Set sarcasm level to {sarcasm_level} using property assignment")
                        elif hasattr(therapist, 'configure'):
                            therapist.configure(sarcasm_level=sarcasm_level)
                            logger.info(f"Set sarcasm level to {sarcasm_level} using configure method")
                        else:
                            logger.warning("No method found to set sarcasm level")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse tone_preferences: {preferences['tone_preferences']}")

                # Direct access if tone_preferences is a dict
                elif 'tone_preferences' in preferences and isinstance(preferences['tone_preferences'], dict):
                    sarcasm_level = preferences['tone_preferences'].get('sarcasm_level', 5)
                    if hasattr(therapist, 'set_sarcasm_level'):
                        therapist.set_sarcasm_level(sarcasm_level)

                # Direct access if sarcasm_level is at the top level
                elif 'sarcasm_level' in preferences:
                    sarcasm_level = preferences['sarcasm_level']
                    if hasattr(therapist, 'set_sarcasm_level'):
                        therapist.set_sarcasm_level(sarcasm_level)
            except Exception as pref_error:
                logger.warning(f"Error setting sarcasm level: {pref_error}")
                # Continue without setting sarcasm level

            # Build enhanced context with more message history for richer responses
            history = session.get('conversation_history', [])

            # Enhanced context with more sophisticated filtering
            # Use last 10 messages as context instead of fewer
            start_idx = max(0, len(history) - 10)
            context = []

            for item in history[start_idx:-1]:  # Exclude the current user message
                # Add rich context by including metadata if available
                context_item = {
                    "role": item.get("role", "unknown"),
                    "content": item.get("content", ""),
                }
                # Optionally include additional metadata that might help with response quality
                if "timestamp" in item:
                    context_item["timestamp"] = item["timestamp"]
                if "id" in item:
                    context_item["id"] = item["id"]
                context.append(context_item)

            # Generate therapist response with enhanced context
            therapist_response = therapist.generate_response(
                user_message,
                context=context
            )

            logger.info(f"Generated response: '{therapist_response}'")

            # Create a rich therapist response object with metadata
            therapist_response_obj = {
                "role": "therapist",
                "content": therapist_response,
                "id": str(uuid.uuid4()),  # Generate a unique ID for the response
                "in_response_to": message_id,  # Track which message this responds to
                "timestamp": data.get('server_timestamp', None)  # Server can add timestamp
            }

            # Add therapist response to history
            session['conversation_history'].append(therapist_response_obj)

            # Generate audio for the response
            audio_url = generate_audio_for_response(therapist_response, preferences)

            # Return enhanced response with metadata
            return jsonify({
                "response": therapist_response,
                "audio_url": audio_url,
                "response_id": therapist_response_obj["id"],
                "in_response_to": message_id
            })

        except Exception as therapist_error:
            logger.error(f"Error with SarcasticTherapistAI: {therapist_error}", exc_info=True)
            # Provide more specific and human-like error message
            if "missing 2 required positional arguments" in str(therapist_error):
                therapist_response = "I'm having trouble connecting to my brain database. Please check your Supabase credentials or try again in a moment."
            else:
                therapist_response = "I'm having a moment of existential crisis. Could you try again with your question? Sometimes rephrasing helps me think more clearly."

            # Even for errors, maintain conversation history
            session['conversation_history'].append({
                "role": "therapist",
                "content": therapist_response,
                "id": str(uuid.uuid4()),
                "in_response_to": message_id,
                "error": True  # Mark this as an error response
            })

            # Still try to generate audio for error messages
            audio_url = generate_audio_for_response(therapist_response, preferences)

            return jsonify({
                "response": therapist_response,
                "audio_url": audio_url
            })

    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def generate_audio_for_response(text, preferences):
    """
    Generate audio with Dia as primary, but fallback to Polly if Dia fails.
    """
    try:
        filename = f"{uuid.uuid4().hex}.wav"
        output_path = os.path.join(app.static_folder, 'audio', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        voice = preferences.get('voice', 'Rishi (en-IN)')
        speech_rate = preferences.get('speech_rate', 1.0)

        # 1) Try Dia first (permanent primary choice)
        try:
            logger.info("Trying Dia for audio generation (primary choice)")
            generate_and_save(text, output_path)

            # Verify Dia worked
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                logger.info(f"Dia audio generated successfully: {os.path.getsize(output_path)} bytes")
                return f"/static/audio/{filename}"
            else:
                logger.warning("Dia generated empty/small file, trying fallback")

        except Exception as dia_error:
            logger.warning(f"Dia failed: {dia_error}")

        # 2) Fallback to Amazon Polly if available
        if POLLY_AVAILABLE:
            try:
                logger.info("Falling back to Amazon Polly")
                generate_with_polly(text, output_path, voice=voice, speech_rate=speech_rate)

                # Verify Polly worked
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    logger.info(f"Polly audio generated successfully: {os.path.getsize(output_path)} bytes")
                    return f"/static/audio/{filename}"

            except Exception as polly_error:
                logger.warning(f"Polly also failed: {polly_error}")

        # 3) Fallback to ElevenLabs if available
        if ELEVENLABS_AVAILABLE:
            try:
                logger.info("Falling back to ElevenLabs")
                elevenlabs_voice = os.getenv("ELEVENLABS_VOICE", voice)
                generate_elevenlabs_audio(text, output_path, voice_name=elevenlabs_voice)

                # Verify ElevenLabs worked
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    logger.info(f"ElevenLabs audio generated successfully: {os.path.getsize(output_path)} bytes")
                    return f"/static/audio/{filename}"

            except Exception as eleven_error:
                logger.warning(f"ElevenLabs also failed: {eleven_error}")

        # 4) Final fallback to silent audio (not buzzing sine wave)
        logger.warning("All TTS services failed, generating silent audio")
        generate_silent_audio(text, output_path, duration_ms=3000)
        return f"/static/audio/{filename}"

    except Exception as e:
        logger.error(f"Error in generate_audio_for_response: {e}", exc_info=True)
        return ""


# Additional debug routes for testing conversation quality
@app.route('/debug/conversation-quality')
def debug_conversation_quality():
    """Debug endpoint to test conversation quality."""
    history = session.get('conversation_history', [])
    metrics = {
        "total_messages": len(history),
        "user_messages": sum(1 for msg in history if msg.get("role") == "user"),
        "therapist_messages": sum(1 for msg in history if msg.get("role") == "therapist"),
        "average_user_length": sum(len(msg.get("content", "")) for msg in history if msg.get("role") == "user") /
                               max(1, sum(1 for msg in history if msg.get("role") == "user")),
        "average_therapist_length": sum(
            len(msg.get("content", "")) for msg in history if msg.get("role") == "therapist") /
                                    max(1, sum(1 for msg in history if msg.get("role") == "therapist")),
        "context_depth": min(10, len(history))
    }
    return jsonify(metrics)


@app.route('/debug/test-context')
def debug_test_context():
    """Debug endpoint to test the context building mechanism."""
    history = session.get('conversation_history', [])
    start_idx = max(0, len(history) - 10)
    context = []

    for item in history[start_idx:]:
        context_item = {
            "role": item.get("role", "unknown"),
            "content": item.get("content", "")
        }
        context.append(context_item)

    return jsonify({
        "context_size": len(context),
        "context": context
    })


if __name__ == '__main__':
    os.makedirs(os.path.join(app.static_folder, 'audio'), exist_ok=True)
    app.run(debug=True)