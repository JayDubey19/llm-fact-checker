import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from src.pipeline import fact_check_text



import streamlit as st





st.set_page_config(page_title="LLM Fact Checker", page_icon="✅", layout="wide")

st.title("🔍 LLM-Powered Fact Checker (RAG)")
st.write(
    "Paste a short news post or social media statement. "
    "This tool will extract claims, retrieve matching verified facts, and use an LLM to judge them."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-k facts per claim", min_value=1, max_value=10, value=5)
    sim_threshold = st.slider(
        "Similarity threshold for Unverifiable",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="If max similarity to any fact is below this, the claim is marked as Unverifiable without calling the LLM.",
    )
    st.markdown("---")
    st.caption("Backend: sentence-transformers + Groq (Llama 3.1)")

text = st.text_area(
    "Input text",
    height=180,
    placeholder="Example: The Indian government has announced free electricity to all farmers starting July 2025.",
)

if st.button("✅ Check Facts"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            result = fact_check_text(
                text.strip(),
                k=top_k,
                sim_threshold=sim_threshold,
            )

        st.subheader("Results")

        results = result.get("results", [])
        if not results:
            st.write("No claims detected.")
        else:
            for idx, r in enumerate(results, start=1):
                st.markdown(f"### Claim #{idx}")
                st.markdown(f"**Claim:** {r['claim'] or '—'}")

                verdict = r.get("verdict", "Unverifiable")
                confidence = r.get("confidence", 0.0)
                max_sim = r.get("max_similarity", 0.0)

                # Verdict emoji
                if "True" in verdict:
                    emoji = "✅"
                elif "False" in verdict:
                    emoji = "❌"
                else:
                    emoji = "🤷‍♂️"

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Verdict:** {emoji} {verdict}")
                with col2:
                    st.metric("Confidence", f"{confidence:.1f} %")
                with col3:
                    st.metric("Best Match Score", f"{max_sim:.3f}")

                st.markdown("**Reasoning:**")
                st.write(r.get("reasoning", ""))

                st.markdown("**Key Evidence (LLM-used):**")
                evidence = r.get("evidence", [])
                if evidence:
                    for i, ev in enumerate(evidence, start=1):
                        st.write(f"{i}. {ev}")
                else:
                    st.write("_No specific evidence was selected by the model._")

                with st.expander("🔎 Retrieved Facts & Similarities"):
                    st.json(r.get("retrieved_facts_meta", []))

                st.markdown("---")

        st.subheader("Feedback")
        helpful = st.checkbox("Was this helpful?")
        st.caption(f"Feedback recorded: {'✅ Yes' if helpful else '❓ Not recorded / No'}")

