# test_eleven.py
from elevenlabs_tts import generate_elevenlabs_audio

# Test with a simple message
output = generate_elevenlabs_audio("This is a test of the ElevenLabs integration", "test_audio.wav")
print(f"Created audio file at: {output}")