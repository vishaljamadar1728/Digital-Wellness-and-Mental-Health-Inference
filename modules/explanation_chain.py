"""
explanation_chain.py
---------------------
This is the SECOND real use of the LLM (still part of the same compulsory
AI/LLM component): it takes the numeric output of the fuzzy inference system
and turns it back into natural language - a warm, conversational explanation
with concrete, grounded suggestions. This is genuine language generation
conditioned on structured data, not a static template.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

EXPLANATION_SYSTEM_PROMPT = """You are a supportive digital-wellness coach.

You will be given:
- The user's original free-text description of their day.
- Structured signals extracted from it (sleep hours, screen time, stress).
- The output of a fuzzy logic inference system: a burnout-risk score (0-100)
  and category (Low/Moderate/High), plus the degree to which each input
  belongs to "low/moderate/high" (membership values).

Write a short (4-6 sentence) conversational response that:
1. Reflects back what you understood about their day, in your own words.
2. Explains in plain language WHY the risk score came out the way it did,
   referencing which factors (sleep/screen time/stress) drove it, based on
   the membership values you were given.
3. Offers 2-3 concrete, specific, low-effort suggestions tailored to what
   they described (not generic "get more sleep" filler).

Important boundaries:
- This is a wellness-awareness tool, not a diagnosis. Do not name or imply
  any clinical mental health condition (e.g. do not say "depression" or
  "anxiety disorder"), even if the user's words suggest distress.
- Do not be alarmist. Be warm, concrete, and practical.
- If the risk category is High, gently suggest that talking to a real person
  they trust, or a professional, could help - without being preachy about it.
"""

EXPLANATION_HUMAN_TEMPLATE = """User's original message:
\"\"\"{user_text}\"\"\"

Extracted signals:
- sleep_hours: {sleep_hours}
- screen_time_hours: {screen_time_hours}
- stress_level: {stress_level}

Fuzzy inference result:
- risk_score: {risk_score}/100
- risk_category: {risk_category}
- membership degrees: {memberships}
"""


def build_explanation_chain(api_key: str, model: str = "openai/gpt-oss-120b"):
    llm = ChatGroq(model=model, temperature=0.6, api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXPLANATION_SYSTEM_PROMPT),
        ("human", EXPLANATION_HUMAN_TEMPLATE),
    ])
    return prompt | llm | StrOutputParser()


def explain_result(api_key: str, user_text: str, inputs, fuzzy_result: dict, model: str = "openai/gpt-oss-120b") -> str:
    """Convenience wrapper used by the Streamlit app."""
    chain = build_explanation_chain(api_key=api_key, model=model)
    return chain.invoke({
        "user_text": user_text,
        "sleep_hours": inputs.sleep_hours,
        "screen_time_hours": inputs.screen_time_hours,
        "stress_level": inputs.stress_level,
        "risk_score": fuzzy_result["score"],
        "risk_category": fuzzy_result["category"],
        "memberships": fuzzy_result["memberships"],
    })
