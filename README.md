# 🧠 Digital Wellness & Burnout Risk Check

A hosted mini-project combining **LangChain + Groq (LLM reasoning)** and a
**genuine Fuzzy Inference System (scikit-fuzzy)** to estimate a "digital
wellness / burnout risk" score from a free-text description of your day.
Groq gives you a generous free tier and very fast inference, so no paid
API key is needed to run this.

> ⚠️ Educational/demo project only. Not a medical or diagnostic tool.

## How it works

```
User free text
     │
     ▼
[LangChain extraction chain]  ← LLM reads the text and infers
     │                           sleep_hours, screen_time_hours, stress_level
     ▼
[Fuzzy Inference System - scikit-fuzzy]
   1. Fuzzification    → membership in "low/moderate/high" sets
   2. Rule evaluation  → ~12 fuzzy IF-THEN rules (fuzzy AND/OR)
   3. Aggregation      → combine rule outputs
   4. Defuzzification  → centroid method → crisp risk score (0-100)
     │
     ▼
[LangChain explanation chain]  ← LLM turns the numeric result back into
     │                            a conversational explanation + suggestions
     ▼
Streamlit UI (gauge chart + explanation)
```

## File structure

```
digital-wellness-fuzzy-ai/
├── app.py                        # Streamlit UI - orchestrates everything
├── modules/
│   ├── __init__.py
│   ├── extraction_chain.py       # LangChain: free text -> structured inputs
│   ├── fuzzy_engine.py           # scikit-fuzzy: fuzzification/rules/defuzzification
│   └── explanation_chain.py      # LangChain: numeric result -> conversational text
├── .streamlit/
│   ├── config.toml               # theme
│   └── secrets.toml.example      # template for your API key
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Step 1 — Run it locally

```bash
# 1. Clone/enter the project
cd digital-wellness-fuzzy-ai

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Get a free Groq API key at https://console.groq.com/keys, then set it
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your real key as GROQ_API_KEY
# OR just paste the key into the sidebar text box when the app opens

# 5. Run
streamlit run app.py
```

It opens at `http://localhost:8501`. Try:
> "Didn't sleep well, maybe 4 hours. Been scrolling my phone almost non-stop
> since I woke up, feel really wired and anxious about a deadline."

Click **Extract signals** → adjust sliders if needed → **Run fuzzy inference**
→ **Explain this result to me**.

*(Want to use OpenAI instead? Swap `ChatGroq` in `modules/extraction_chain.py`
and `modules/explanation_chain.py` for `ChatOpenAI` from `langchain-openai`
— the rest of the pipeline is unchanged.)*

---

## Step 2 — Push to GitHub

```bash
cd digital-wellness-fuzzy-ai
git init
git add .
git commit -m "Initial commit: fuzzy wellness inference app"
git branch -M main
git remote add origin https://github.com/<your-username>/digital-wellness-fuzzy-ai.git
git push -u origin main
```

`.gitignore` already excludes `.streamlit/secrets.toml`, `venv/`, and `.env` —
**never commit your real API key.**

---

## Step 3 — Host it for free (Streamlit Community Cloud)

1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **New app** → pick your `digital-wellness-fuzzy-ai` repo → branch
   `main` → main file path `app.py`.
3. Before/after deploying, open **Settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your-real-key"
   ```
4. Click **Deploy**. In ~1-2 minutes you'll get a public URL like
   `https://your-app-name.streamlit.app`.

The app reads `st.secrets["GROQ_API_KEY"]` automatically as the sidebar's
default value, so users don't need to paste a key themselves once it's set
as a secret.

### Alternative: Hugging Face Spaces
1. Create a new Space → SDK: **Streamlit**.
2. Push the same repo (or link your GitHub repo) to the Space.
3. In **Settings → Repository secrets**, add `GROQ_API_KEY`.
4. The Space builds and hosts automatically at `https://huggingface.co/spaces/<you>/<space-name>`.

---

## Design notes (for your write-up/report)

- **Fuzzy logic is real, not disguised if-else**: `modules/fuzzy_engine.py`
  defines trapezoidal/triangular membership functions over three inputs
  (sleep, screen time, stress), ~12 rules combined with fuzzy AND, Mamdani-style
  aggregation, and **centroid defuzzification** via `skfuzzy.control`. You can
  inspect membership degrees live in the "membership degrees" expander in the UI.
- **LLM does two distinct reasoning jobs** (LangChain): (1) extracting bounded
  structured numbers from unstructured, ambiguous free text with a rationale,
  and (2) generating a grounded, non-generic explanation conditioned on the
  fuzzy system's actual output (score, category, membership degrees) - not a
  canned template.
- **Responsible framing**: the app explicitly disclaims that it is not a
  diagnostic tool, avoids naming clinical conditions, and nudges toward a real
  professional or trusted person when risk is High.
