import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "allam-2-7b",
]


def generate(prompt):
    """
    Generate an AI response using Groq with automatic model failover.
    """
    last_error = None
    for model_name in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            continue

    return f"""
## Groq Error

{str(last_error)}
"""