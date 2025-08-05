# elevenlabs_tts.py - direct API version
import os
import requests


def generate_elevenlabs_audio(text: str, output_path: str, voice_name: str = None) -> str:
    """
    Synthesize `text` with ElevenLabs API directly and save to `output_path`.
    Returns the path on success, or raises on failure.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set in environment")

    # Default voice ID for Rachel
    voice_id = "21m00Tcm4TlvDq8ikWAM"

    # If a specific voice was requested, try to find it
    if voice_name and voice_name != "Rachel":
        try:
            voices_response = requests.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": api_key}
            )
            voices = voices_response.json().get("voices", [])
            for voice in voices:
                if voice_name.lower() in voice["name"].lower():
                    voice_id = voice["voice_id"]
                    break
        except Exception as e:
            print(f"Error getting voices: {e}")
            # Continue with default voice

    # Generate audio using the API directly
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            # Save the audio file
            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path
        else:
            # Handle errors
            error_message = f"ElevenLabs API error: {response.status_code} - {response.text}"
            raise RuntimeError(error_message)
    except Exception as e:
        raise RuntimeError(f"Error calling ElevenLabs API: {e}")