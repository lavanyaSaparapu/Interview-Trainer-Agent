"""
IBM watsonx.ai client — wraps the Granite model for interview Q&A generation.

Reads credentials from environment variables (set via .env):
  WATSONX_API_KEY   — IBM Cloud API key
  WATSONX_PROJECT_ID — watsonx.ai project ID
  WATSONX_URL        — watsonx.ai endpoint (default: Dallas)
"""

import os
from dotenv import load_dotenv

load_dotenv()

WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")

# IBM Granite model ID supported on watsonx.ai
MODEL_ID = "ibm/granite-8b-code-instruct"

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        raise EnvironmentError(
            "WATSONX_API_KEY and WATSONX_PROJECT_ID must be set in the .env file."
        )

    from ibm_watsonx_ai import APIClient, Credentials

    credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    _client = APIClient(credentials)
    return _client


def generate_response(prompt: str, max_new_tokens: int = 700) -> str:
    """
    Send a prompt to IBM Granite and return the generated text.
    Falls back to a mock response if credentials are not configured,
    so the app is still usable during development/demo.
    """
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return _mock_response(prompt)

    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params

        client = _get_client()
        model = ModelInference(
            model_id=MODEL_ID,
            api_client=client,
            project_id=WATSONX_PROJECT_ID,
            params={
                Params.MAX_NEW_TOKENS: max_new_tokens,
                Params.TEMPERATURE: 0.7,
                Params.TOP_P: 0.9,
                Params.REPETITION_PENALTY: 1.1,
            },
        )
        result = model.generate_text(prompt=prompt)
        return result.strip()

    except Exception as exc:
        return f"[watsonx.ai error: {exc}]\n\n" + _mock_response(prompt)


def _mock_response(prompt: str) -> str:
    """
    Returns a structured mock when IBM credentials are not set.
    Used for local development and demo without IBM Cloud access.
    """
    return (
        "**[Demo Mode — IBM Granite not connected]**\n\n"
        "Here is a sample response your agent would provide:\n\n"
        "**Suggested Questions:**\n"
        "1. Can you walk me through a challenging technical problem you solved recently?\n"
        "2. How do you stay updated with the latest trends in your field?\n"
        "3. Describe your approach to writing clean, maintainable code.\n\n"
        "**Improvement Tips:**\n"
        "- Use the STAR method for behavioral questions.\n"
        "- Quantify your achievements wherever possible.\n"
        "- Research the company's tech stack and mention it specifically.\n\n"
        "_To enable full IBM Granite responses, set WATSONX_API_KEY and "
        "WATSONX_PROJECT_ID in the .env file._"
    )
