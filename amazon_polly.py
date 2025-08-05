# amazon_polly.py
import os
import boto3
import torchaudio
from io import BytesIO

def generate_with_polly(text, output_path, voice="Joanna", speech_rate=1.0):
    # Init Polly client
    polly = boto3.client(
        "polly",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    # Synthesize as PCM @24 kHz
    resp = polly.synthesize_speech(
        Text=text,
        OutputFormat="pcm",
        VoiceId=voice.split()[0],
        SampleRate="24000"
    )
    pcm = resp["AudioStream"].read()

    # Load raw PCM into tensor
    waveform, sr = torchaudio.load(
        BytesIO(pcm),
        format="raw",
        frame_rate=24000,
        channels_first=True,
        channels=1,
        dtype=torch.int16
    )
    # Save as WAV
    torchaudio.save(output_path, waveform, sample_rate=sr)
    return output_path
