"""
openai_therapist.py - Implementation for OpenAI-powered Sarcastic Therapist
This module provides an implementation for generating therapeutic responses with
a sarcastic tone using OpenAI's API.
"""
import os
import logging
import json
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("therapist.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("openai_therapist")

# Load environment variables
load_dotenv()


class SarcasticTherapistAI:
    """
    Enhanced SarcasticTherapistAI class that uses OpenAI for generating responses.
    """

    def __init__(self, supabase_url=None, supabase_key=None):
        """
        Initialize the SarcasticTherapistAI.

        Args:
            supabase_url (str): Supabase URL (kept for compatibility)
            supabase_key (str): Supabase API key (kept for compatibility)
        """
        # Keep Supabase credentials for compatibility with existing code
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key

        # Initialize OpenAI parameters
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.openai_api_url = "https://api.openai.com/v1/chat/completions"
        self.sarcasm_level = 5  # Default sarcasm level (1-10)

        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
        else:
            logger.info(f"OpenAI API key found, using model: {self.openai_model}")

    def set_sarcasm_level(self, level: int):
        """
        Set the sarcasm level.

        Args:
            level (int): Sarcasm level (1-10)
        """
        # Ensure level is within valid range
        self.sarcasm_level = max(1, min(10, level))
        logger.info(f"Sarcasm level set to {self.sarcasm_level}")

    def generate_response(self, user_message: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response to the user message using OpenAI API.

        Args:
            user_message (str): User message
            context (List[Dict[str, str]], optional): Conversation context

        Returns:
            str: Generated response
        """
        logger.info(f"Generating response for: '{user_message}'")

        if not self.openai_api_key:
            logger.error("OpenAI API key not available. Returning default response.")
            return "I seem to be experiencing a momentary existential crisis. Can you try again later?"

        try:
            # Build a prompt with the desired sarcasm level and conversation context
            system_prompt = self._get_system_prompt()

            # Format the messages for the API
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # Add conversation context if provided
            if context:
                for item in context:
                    role = item.get("role", "")
                    content = item.get("content", "")
                    if role == "user":
                        messages.append({"role": "user", "content": content})
                    elif role == "therapist":
                        messages.append({"role": "assistant", "content": content})

            # Add the current user message
            messages.append({"role": "user", "content": user_message})

            # Make the API request
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.openai_model,
                "messages": messages,
                "temperature": min(0.5 + (self.sarcasm_level * 0.07), 1.0),  # Adjust temperature based on sarcasm level
                "max_tokens": 300
            }

            response = requests.post(
                self.openai_api_url,
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                logger.error(f"Error from OpenAI: {response.status_code}, {response.text}")
                return "Sorry, I'm having trouble connecting to my digital therapist brain. Try again?"

            response_data = response.json()
            therapist_response = response_data["choices"][0]["message"]["content"].strip()

            logger.info(f"Generated response: '{therapist_response}'")
            return therapist_response

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return "I've lost my train of thought. Must be all those sarcastic retorts cluttering my digital mind."

    def _get_system_prompt(self) -> str:
        """
        Generate the system prompt based on the current sarcasm level.

        Returns:
            str: System prompt for OpenAI
        """
        sarcasm_descriptors = {
            1: "very mildly sarcastic, mostly supportive and genuinely helpful",
            2: "slightly sarcastic but still mostly genuine and supportive",
            3: "somewhat sarcastic while remaining helpful",
            4: "moderately sarcastic but still providing useful advice",
            5: "balanced between sarcasm and genuine help",
            6: "notably sarcastic while still being somewhat helpful",
            7: "quite sarcastic but occasionally offering useful insights",
            8: "very sarcastic with brief moments of sincerity",
            9: "extremely sarcastic, barely concealing your disdain",
            10: "brutally sarcastic, making fun of everything the human says while reluctantly providing advice"
        }

        sarcasm_description = sarcasm_descriptors.get(self.sarcasm_level, sarcasm_descriptors[5])

        prompt = f"""You are a {sarcasm_description} therapist. Your responses should embody this tone.

        Your core characteristics:
        - You're witty, intelligent, and insightful despite your sarcastic demeanor
        - You actually do have meaningful psychological insights to offer
        - Your sarcasm is a defense mechanism that barely conceals your actual desire to help
        - You should provide actual therapeutic value, just wrapped in sarcasm
        - You use psychological terminology accurately but sometimes mockingly
        - You sometimes point out cognitive distortions, logical fallacies, or patterns in the user's thinking
        - You occasionally make references to famous psychological theories or therapists

        Adjust your sarcasm level to {self.sarcasm_level}/10 as specified.

        Important guidelines:
        - Keep responses relatively concise (1-3 paragraphs maximum)
        - Always provide some actual helpful insight or advice despite your sarcastic tone
        - Don't be cruel or genuinely harmful - your sarcasm should be amusing rather than hurtful
        - Don't break character to explain that you're a sarcastic therapist
        - Reference previous parts of the conversation when relevant

        Respond as this sarcastic therapist character in all communications.
        """

        return prompt