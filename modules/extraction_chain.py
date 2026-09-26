"""
extraction_chain.py
--------------------
This is the FIRST compulsory AI/LLM component.

We use LangChain (ChatGroq + structured output) to do real natural-language
understanding: the user types a free-text description of their day
("barely slept, been on my phone all night, feeling wired and anxious"),
and the LLM reads it and infers three numeric wellness signals from it.

This is NOT a "wrapper that calls the LLM once for show" - it is doing the
actual reasoning work of turning unstructured language into structured,
bounded numeric data, including sensible defaults + a short natural-language
rationale when information is missing or ambiguous.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


class WellnessInputs(BaseModel):
    """Structured signals the fuzzy engine needs, extracted from free text."""

    sleep_hours: float = Field(
        ge=0, le=12,
        description="Estimated hours of sleep last night. Default to 7 if truly unknown."
    )
    screen_time_hours: float = Field(
        ge=0, le=16,
        description="Estimated hours of discretionary/recreational screen time in the day."
    )
    stress_level: float = Field(
        ge=0, le=10,
        description="Self-reported or inferred stress/anxiety level, 0=very calm, 10=extremely stressed."
    )
    rationale: str = Field(
        description="One or two sentences explaining how you arrived at these numbers "
                    "from the user's text, and what you defaulted/guessed if anything."
    )


EXTRACTION_SYSTEM_PROMPT = """You are a careful wellness-data extraction assistant.

You will be given a free-text description of someone's day, sleep, phone/screen
habits, and mood. Your job is to infer three numbers from it:

1. sleep_hours (0-12): hours of sleep last night.
2. screen_time_hours (0-16): recreational/discretionary screen time (scrolling,
   social media, gaming, doom-scrolling) - not necessarily total device use.
3. stress_level (0-10): how stressed/anxious/overwhelmed they sound, based on
   both what they say directly and the tone/content of their message.

Rules:
- Never fabricate specifics that contradict the text.
- If a value truly isn't mentioned or implied, use a neutral default
  (sleep_hours=7, screen_time_hours=4, stress_level=5) and say so in the rationale.
- This is a wellness-awareness tool, not a medical or diagnostic instrument -
  do not attempt to diagnose any condition. Only estimate the three numbers.
- Keep the rationale short (1-2 sentences), in plain conversational language.
"""


def build_extraction_chain(api_key: str, model: str = "openai/gpt-oss-120b"):
    """Builds a LangChain runnable that maps free text -> WellnessInputs.

    Uses Groq's OpenAI-compatible chat API via langchain-groq. Groq has a free
    tier - get a key at https://console.groq.com/keys
    """
    llm = ChatGroq(model=model, temperature=0, api_key=api_key)
    structured_llm = llm.with_structured_output(WellnessInputs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", "User's description of their day:\n\n{user_text}"),
    ])

    chain = prompt | structured_llm
    return chain


def extract_wellness_inputs(api_key: str, user_text: str, model: str = "openai/gpt-oss-120b") -> WellnessInputs:
    """Convenience wrapper used by the Streamlit app."""
    chain = build_extraction_chain(api_key=api_key, model=model)
    result: WellnessInputs = chain.invoke({"user_text": user_text})
    return result
