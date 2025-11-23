import json
from typing import List, Dict
from groq import Groq
import os


client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a strict fact-checking AI.

YOU MUST ALWAYS OUTPUT VALID JSON.
NO explanations.
NO natural language outside JSON.

The ONLY valid output format is:

{
  "verdict": "True | Likely True | False | Likely False | Unverifiable",
  "evidence": ["string1", "string2"],
  "reasoning": "2-4 sentences"
}
}

If you cannot produce valid JSON, output:

{
  "verdict": "Unverifiable",
  "evidence": [],
  "reasoning": "Unable to determine due to insufficient or incorrect data."
}
"""

def build_prompt(claim: str, facts: List[Dict]):
    text = []
    text.append("Return ONLY a valid JSON object. No explanation.")
    text.append("\nCLAIM:")
    text.append(claim)
    text.append("\nRETRIEVED FACTS:")
    for f in facts:
        text.append(f"- " + f["statement"])

    text.append("\nGenerate only JSON. Do NOT include backticks or markdown.")
    return "\n".join(text)


def judge_claim(claim: str, facts: List[Dict]):
    prompt = build_prompt(claim, facts)

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # Free & fast model
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    content = completion.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()


    # Try to parse JSON strictly
    try:
        out = json.loads(content)
    except:
        out = {
            "verdict": "Unverifiable",
            "evidence": [f["statement"] for f in facts],
            "reasoning": f"Model did not return valid JSON. Raw:\n{content}"
        }

    return out
