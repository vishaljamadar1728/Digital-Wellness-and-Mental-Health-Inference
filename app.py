import streamlit as st
import plotly.graph_objects as go

from modules.extraction_chain import extract_wellness_inputs, WellnessInputs
from modules.fuzzy_engine import compute_burnout_risk
from modules.explanation_chain import explain_result

st.set_page_config(page_title="Digital Wellness Risk Check", page_icon="🧠", layout="centered")

st.title("🧠 Digital Wellness & Burnout Risk Check")
st.caption(
    "Describe your day in your own words. An LLM (via LangChain) reads it and estimates "
    "your sleep, screen time, and stress. A fuzzy logic engine (scikit-fuzzy) turns those "
    "into a burnout-risk score, and the LLM explains the result conversationally."
)

with st.expander("⚠️ Please read: what this tool is and isn't"):
    st.write(
        "This is a student/demo project for a class assignment on AI + fuzzy logic. "
        "It is **not** a medical or diagnostic tool and cannot assess mental health "
        "conditions. It estimates a general 'digital wellness' signal from self-reported "
        "text for awareness purposes only. If you're struggling, please reach out to a "
        "real person you trust or a mental health professional."
    )

# ---------------- API key handling ----------------
# For a PUBLIC deployment: set GROQ_API_KEY in Streamlit secrets (Settings →
# Secrets in Streamlit Community Cloud) so visitors never see or type a key -
# the app just uses yours silently, server-side. The manual text box below
# only appears as a local-dev fallback when no secret is configured, so it
# never shows up on the hosted public link once the secret is set.
secret_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

if secret_key:
    api_key = secret_key
    st.sidebar.success("Groq API key loaded from app secrets ✅")
else:
    api_key = st.sidebar.text_input(
        "Groq API key",
        type="password",
        help="No app-level key is configured, so this is for local testing only. "
             "Get a free key at console.groq.com/keys.",
    )
    st.sidebar.caption(
        "⚠️ No GROQ_API_KEY secret is set, so this box is showing. "
        "Anyone with this tab open can reveal a typed key with the 👁 icon - "
        "don't paste a real key here on a publicly shared link. Set "
        "GROQ_API_KEY in Streamlit secrets instead."
    )

model_name = st.sidebar.selectbox(
    "Model",
    ["openai/gpt-oss-120b", "openai/gpt-oss-120b", "openai/gpt-oss-120b"],
    index=0,
    help="70b-versatile: most capable. 8b-instant: fastest/cheapest. "
         "gpt-oss-120b: OpenAI's open-weight model, also hosted on Groq."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works**\n"
    "1. LangChain extracts sleep/screen-time/stress from your text\n"
    "2. You can review/adjust the extracted numbers\n"
    "3. A fuzzy inference system (membership → rules → centroid "
    "defuzzification) computes a risk score\n"
    "4. LangChain explains the result conversationally"
)

if "inputs" not in st.session_state:
    st.session_state.inputs = None
if "fuzzy_result" not in st.session_state:
    st.session_state.fuzzy_result = None

# ---------------- Step 1: free text input + extraction ----------------
user_text = st.text_area(
    "How was your day? (sleep, phone/screen habits, mood, stress — anything relevant)",
    height=140,
    placeholder="e.g. Didn't sleep well, maybe 4 hours. Been scrolling my phone almost "
                "non-stop since I woke up, feel really wired and anxious about a deadline.",
)

col1, col2 = st.columns([1, 1])
with col1:
    extract_clicked = st.button("1️⃣ Extract signals from my text", use_container_width=True)

if extract_clicked:
    if not api_key:
        st.error("No Groq API key available. Set GROQ_API_KEY in Streamlit secrets, or enter one in the sidebar for local testing.")
    elif not user_text.strip():
        st.error("Please describe your day first.")
    else:
        with st.spinner("Reading your message and extracting signals..."):
            try:
                extracted: WellnessInputs = extract_wellness_inputs(api_key, user_text, model=model_name)
                st.session_state.inputs = extracted
                st.session_state.fuzzy_result = None
            except Exception as e:
                st.error(f"Extraction failed: {e}")

# ---------------- Step 2: review/adjust + run fuzzy inference ----------------
if st.session_state.inputs:
    st.subheader("Extracted signals (edit if they look off)")
    st.info(st.session_state.inputs.rationale)

    sleep_hours = st.slider("Sleep hours", 0.0, 12.0, float(st.session_state.inputs.sleep_hours), 0.5)
    screen_time_hours = st.slider("Screen time (hrs)", 0.0, 16.0, float(st.session_state.inputs.screen_time_hours), 0.5)
    stress_level = st.slider("Stress level (0-10)", 0.0, 10.0, float(st.session_state.inputs.stress_level), 0.5)

    if st.button("2️⃣ Run fuzzy inference", use_container_width=True):
        with st.spinner("Fuzzifying inputs, evaluating rules, defuzzifying..."):
            st.session_state.inputs.sleep_hours = sleep_hours
            st.session_state.inputs.screen_time_hours = screen_time_hours
            st.session_state.inputs.stress_level = stress_level
            st.session_state.fuzzy_result = compute_burnout_risk(sleep_hours, screen_time_hours, stress_level)

# ---------------- Step 3: show fuzzy result + gauge ----------------
if st.session_state.fuzzy_result:
    result = st.session_state.fuzzy_result
    st.subheader("Fuzzy inference result")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result["score"],
        title={"text": f"Burnout risk: {result['category']}"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "color": "#c8f7c5"},
                {"range": [40, 65], "color": "#fff3b0"},
                {"range": [65, 100], "color": "#ffb3b3"},
            ],
            "bar": {"color": "#333333"},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("See membership degrees (fuzzification detail)"):
        st.json(result["memberships"])

    if st.button("3️⃣ Explain this result to me", use_container_width=True):
        with st.spinner("Writing a conversational explanation..."):
            try:
                explanation = explain_result(
                    api_key, user_text, st.session_state.inputs, result, model=model_name
                )
                st.subheader("What this means")
                st.write(explanation)
            except Exception as e:
                st.error(f"Explanation failed: {e}")

st.markdown("---")
st.caption(
    "Built with LangChain + Groq (LLM reasoning) + scikit-fuzzy (fuzzy inference) + Streamlit. "
    "Not a substitute for professional advice."
)