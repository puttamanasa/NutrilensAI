from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
import json
# ---------------- SINGLE RESPONSE ----------------
def rag_diet_reasoning(caption, nutrition, medical):
    prompt = f"""
You are a professional nutritionist AI. Respond ONLY with valid JSON, no markdown, no extra text.

Food detected: {caption}
Calories: {nutrition['calories']} kcal | Protein: {nutrition['protein']}g | Carbs: {nutrition['carbs']}g | Fat: {nutrition['fat']}g
Medical Condition: {medical['disease']}
Recommended foods: {medical['eat']}
Foods to avoid: {medical['avoid']}

Return this exact JSON structure:
{{
  "score": <number 0-10>,
  "score_reason": "<one sentence why>",
  "summary": "<2 sentence food description>",
  "limit": ["<tip 1>", "<tip 2>"],
  "balance_with": ["<food 1>", "<food 2>"],
  "modifications": ["<tip 1>", "<tip 2>"],
  "hydration": "<one sentence hydration advice>",
  "health_tips": ["<tip 1>", "<tip 2>"],
   
}}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3   # lower = more consistent structure
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return a safe default so the frontend never crashes
        return {
            "score": None,
            "score_reason": "",
            "summary": raw[:300],
            "limit": [],
            "balance_with": [],
            "modifications": [],
            "hydration": "",
            "health_tips": []
        }

# ---------------- WEEKLY ANALYSIS ----------------
def weekly_diet_recommendation(df):

    summary = df.describe().to_string()

    prompt = f"""
    Analyze this weekly nutrition data:

    {summary}

    Give:
    - Overall diet assessment
    - Improvements
    - Health suggestions
    Keep it short, structured, and practical.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content