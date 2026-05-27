"""
core/llm_generator.py
======================
STEP 11 — LLM Generator (Groq Llama 3)

Handles:
  - Groq API calls with streaming
  - Non-streaming generation
  - Token usage tracking
  - Error handling and retries
"""

from typing import Generator, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_groq_client():
    from groq import Groq
    from config.settings import GROQ_API_KEY
    return Groq(api_key=GROQ_API_KEY)


def generate(
    messages: list[dict],
    stream: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> str | Generator:
    """
    Generate a response from Groq Llama 3.

    Args:
        messages: List of {role, content} dicts
        stream: If True, returns a generator for streaming
        temperature: Override default temperature
        max_tokens: Override default max tokens
        model: Override default model

    Returns:
        Full response string (non-streaming) or generator (streaming)
    """
    from config.settings import GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS

    client = _get_groq_client()
    _model = model or GROQ_MODEL
    _temperature = temperature if temperature is not None else GROQ_TEMPERATURE
    _max_tokens = max_tokens or GROQ_MAX_TOKENS

    logger.info(f"Generating with {_model} (stream={stream}, temp={_temperature})")

    if stream:
        return _stream_generate(client, messages, _model, _temperature, _max_tokens)
    else:
        return _full_generate(client, messages, _model, _temperature, _max_tokens)


def _full_generate(client, messages, model, temperature, max_tokens) -> str:
    """Non-streaming generation."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content

        # Log token usage
        usage = response.usage
        if usage:
            logger.info(
                f"Tokens — prompt: {usage.prompt_tokens}, "
                f"completion: {usage.completion_tokens}, "
                f"total: {usage.total_tokens}"
            )

        return content

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise


def _stream_generate(client, messages, model, temperature, max_tokens) -> Generator:
    """Streaming generation — yields text chunks."""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        yield f"\n\n❌ Generation error: {str(e)}"


def generate_with_fallback(
    messages: list[dict],
    stream: bool = False,
) -> str | Generator:
    """
    Generate with automatic fallback to a smaller model on failure.
    """
    from config.settings import GROQ_MODEL

    try:
        return generate(messages, stream=stream)
    except Exception as e:
        logger.warning(f"Primary model failed: {e}. Trying fallback model.")
        try:
            # Fallback to Llama 3.1 8B
            return generate(messages, stream=stream, model="llama-3.1-8b-instant")
        except Exception as e2:
            logger.error(f"Fallback model also failed: {e2}")
            if stream:
                def error_gen():
                    yield "❌ Unable to generate response. Please check your Groq API key and try again."
                return error_gen()
            return "❌ Unable to generate response. Please check your Groq API key and try again."
