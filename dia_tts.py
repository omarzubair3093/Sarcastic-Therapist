"""
dia_tts.py - Implementation for Dia text-to-speech model integration
This module provides an implementation for integrating the Dia text-to-speech model
"""
import os
import logging
import torch
import torchaudio
import numpy as np
import requests
from typing import List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dia_tts")

# Load .env variables
load_dotenv()

# Singleton pattern for model instance
_model_instance = None


@dataclass
class Segment:
    """Represents a segment of speech with speaker, text, and audio."""
    speaker: int
    text: str
    audio: Optional[torch.Tensor] = None


def get_model_instance(device="cpu"):
    """
    Get or create a singleton instance of the DiaGenerator.
    Args:
        device (str): Device to load the model on ('cpu' or 'cuda')
    Returns:
        DiaGenerator: Instance of the generator
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = DiaGenerator(device=device)
    return _model_instance


class DiaGenerator:
    """
    Generator class for Dia text-to-speech model.
    """

    def __init__(self, device="cpu"):
        """
        Initialize the DiaGenerator.
        Args:
            device (str): Device to load the model on ('cpu' or 'cuda')
        """
        self.device = device
        self.sample_rate = 24000  # Sample rate for Dia
        self.api_key = os.getenv("DIA_API_KEY")
        self.api_url = os.getenv("DIA_API_URL", "https://api.nari-labs.com/predict")

        logger.info(f"Initializing DiaGenerator on device: {device}")
        if not self.api_key:
            logger.error("DIA_API_KEY not found in environment variables")
        else:
            logger.info("Dia API key found in environment variables")

    def generate(self, text, speaker=0, context=None, max_audio_length_ms=10000):
        """
        Generate audio from text using Dia API.
        Args:
            text (str): Text to convert to speech
            speaker (int): Speaker ID
            context (List[Segment], optional): Previous conversation segments
            max_audio_length_ms (int): Maximum audio length in milliseconds
        Returns:
            torch.Tensor: Generated audio waveform
        """
        if not self.api_key:
            logger.error("API key not available. Raising exception for fallback handling.")
            raise Exception("Dia API key not available")

        try:
            logger.info(f"Generating audio for text: '{text}', speaker: {speaker}")

            # Prepare payload for Dia API
            payload = {
                "text": text,
                "speaker_id": speaker,
                "voice": "default",  # Can be customized based on available voices
                "output_format": "wav"
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Request audio generation from Dia API
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                stream=True
            )

            if response.status_code != 200:
                logger.error(f"Error from Dia API: {response.status_code}, {response.text}")
                raise Exception(f"Dia API error: {response.status_code}")

            # Process response and convert to tensor
            audio_bytes = response.content

            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Normalize to [-1,1]
            audio_array = audio_array.astype(np.float32) / 32767.0

            # Convert to torch tensor
            waveform = torch.from_numpy(audio_array).unsqueeze(0)

            logger.info(f"Generated audio for: '{text}', shape: {waveform.shape}")
            return waveform

        except Exception as e:
            logger.error(f"Error generating audio: {e}", exc_info=True)
            raise Exception(f"Dia TTS failed: {e}")


def generate_and_save(text, output_path, speaker=0):
    """
    Generate audio from text and save to file.
    Args:
        text (str): Text to convert to speech
        output_path (str): Path to save the audio file
        speaker (int): Speaker ID
    Returns:
        str: Path to the saved audio file
    """
    logger.info(f"generate_and_save called with text: '{text}', output_path: {output_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    generator = get_model_instance(device=device)
    audio = generator.generate(text=text, speaker=speaker)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure audio is in the correct format for browsers (16-bit PCM)
    audio_numpy = audio.squeeze(0).cpu().numpy()

    # Convert to 16-bit PCM if needed
    if audio_numpy.dtype != 'int16':
        # Normalize to [-1,1] if not already
        if audio_numpy.max() > 1.0 or audio_numpy.min() < -1.0:
            audio_numpy = audio_numpy / max(abs(audio_numpy.max()), abs(audio_numpy.min()))
        # Convert to 16-bit PCM
        audio_numpy = (audio_numpy * 32767).astype(np.int16)

    # Save using torchaudio with explicit format
    logger.info(f"Saving audio to {output_path}")
    torchaudio.save(
        output_path,
        torch.tensor(audio_numpy).unsqueeze(0),
        generator.sample_rate,
        format='wav'
    )

    # Verify the file exists and has content
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        logger.info(f"Audio file generated: {output_path}, size: {file_size} bytes")
        if file_size == 0:
            logger.warning("WARNING: Generated audio file is empty")
            raise Exception("Generated audio file is empty")
    else:
        logger.error(f"ERROR: Audio file was not created at {output_path}")
        raise Exception(f"Audio file was not created at {output_path}")

    return output_path


def generate_silent_audio(text, output_path, duration_ms=3000):
    """
    Generate silent audio as a fallback.
    Args:
        text (str): Text that would have been converted to speech
        output_path (str): Path to save the audio file
        duration_ms (int): Duration of silent audio in milliseconds
    Returns:
        str: Path to the saved audio file
    """
    logger.warning(f"Generating silent audio for: '{text}'")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate silent audio
    sample_rate = 24000
    duration_samples = int(duration_ms * sample_rate / 1000)
    silent_audio = np.zeros(duration_samples, dtype=np.int16)

    # Save silent audio
    torchaudio.save(
        output_path,
        torch.tensor(silent_audio).unsqueeze(0),
        sample_rate,
        format='wav'
    )

    logger.info(f"Silent audio saved to: {output_path}")
    return output_path