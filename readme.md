LLM-Powered Fact Checker (RAG System)

A lightweight Retrieval-Augmented Generation (RAG) system that extracts factual claims from text, retrieves relevant evidence using embeddings, and verifies the claim using an LLM.
Outputs: True, False, or Unverifiable with reasoning, evidence, and confidence scoring.

🚀 Features

Claim Extraction (spaCy)

Embedding-Based Retrieval (SentenceTransformers MiniLM)

Top-K Similarity Search

Similarity Threshold for Unverifiable Claims

LLM Judge (Groq Llama 3.1)

Confidence Scoring & Evidence Ranking

Streamlit UI Dashboard

Fully modular pipeline

📁 Project Structure
LLM-FACT-CHECKER/
│
├── app_streamlit.py
│
├── data/
│   ├── facts.csv
│   └── embeddings.npz
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── retriever.py
│   ├── claim_extraction.py
│   ├── llm_judge.py
│   └── build_index.py
│
├── samples/
│   ├── sample_input.txt
│   └── sample_output.json
│
├── requirements.txt
└── README.md

⚙️ Installation
git clone https://github.com/JayDubey19/llm-fact-checker
cd llm-fact-checker

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt


(Optional) rebuild embeddings:

python src/build_index.py


Run Streamlit:

streamlit run app_streamlit.py

🧪 Sample Input
The government launched a new initiative to support solar irrigation pumps for farmers.

📤 Sample Output
{
  "claim": "The government launched a new initiative to support solar irrigation pumps for farmers.",
  "verdict": "Likely True",
  "confidence": 92.1,
  "max_similarity": 0.9909814491702527,
  "evidence": [
    "Government launches new initiative to support solar irrigation pumps for farmers."
  ],
  "reasoning": "The retrieved fact confirms the claim directly."
}

⭐ Highlights

Multi-claim processing

Confidence + similarity scoring

Hallucination-resistant design

Modular architecture (easy to expand)

Fast inference using Groq Llama 3.1

Clean Streamlit UI with tuning controls

