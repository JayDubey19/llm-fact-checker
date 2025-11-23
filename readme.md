# LLM-Powered Fact Checker (RAG System)

A lightweight Retrieval-Augmented Generation (RAG) system that extracts factual claims from text, retrieves relevant evidence using embeddings, and classifies each claim as:

- **True**
- **False**
- **Unverifiable**

Includes a modular backend pipeline and a complete Streamlit UI.

---

## 🚀 Features

- Claim Extraction (spaCy)
- Embedding-Based Retrieval (SentenceTransformers MiniLM)
- Top-K Similarity Search
- Similarity Threshold for Unverifiable Claims
- LLM Judge (Groq Llama 3.1)
- Confidence Scoring & Evidence Ranking
- Streamlit UI Dashboard
- Fully modular, scalable pipeline

---

## 📁 Project Structure

```
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
├── screenshots/
│   ├── ui_home.png
│   ├── verdict_example.png
│   └── settings_sidebar.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <repo-link>
cd llm-fact-checker

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Optional: Rebuild embeddings

```bash
python src/build_index.py
```

---

## ▶️ Run Streamlit

```bash
streamlit run app_streamlit.py
```

---

## 📸 Screenshots

### Main UI
<img width="1917" height="917" alt="Image" src="https://github.com/user-attachments/assets/2b522bc0-54f9-4d0a-9ec9-c087b886f699" />

### Verdict Output
<img width="1918" height="922" alt="Image" src="https://github.com/user-attachments/assets/19b0ebe8-1acd-4c3c-86d4-98f009a07737" />

<img width="1907" height="902" alt="Image" src="https://github.com/user-attachments/assets/f50fdc7d-ccc3-4358-8874-a222f3fbbd73" />

---

## 🧪 Sample Input

```
The government launched a new initiative to support solar irrigation pumps for farmers.
```

## 📤 Sample Output

```json
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
```

---

## ⭐ Highlights

- Multi-claim processing  
- Confidence & similarity scoring  
- Hallucination-resistant pipeline  
- Scalable architecture  
- Fast Groq inference  
- Clean and interactive Streamlit UI

---

## 👤 Author

**Jay Dubey**
