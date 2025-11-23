# src/pipeline.py

import argparse
import json
from typing import Dict, Any, List

from claim_extraction import extract_claims
from retriever import retrieve_top_k
from llm_judge import judge_claim


def fact_check_single_claim(
    claim: str,
    k: int = 5,
    sim_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Fact-check a single claim:
    - retrieve top-k facts
    - compute max similarity + confidence score
    - if similarity too low -> Unverifiable
    - else call LLM judge
    """
    retrieved = retrieve_top_k(claim, k=k)

    max_sim = max((f["similarity"] for f in retrieved), default=0.0)
    confidence = round(max_sim * 100, 1)  # 0–100

    # If we have no strong matches, mark as Unverifiable without hitting LLM
    if not retrieved or max_sim < sim_threshold:
        return {
            "claim": claim,
            "verdict": "Unverifiable",
            "confidence": confidence,
            "max_similarity": max_sim,
            "evidence": [],
            "reasoning": (
                f"No sufficiently similar facts found for this claim. "
                f"Maximum similarity={max_sim:.2f} which is below the threshold={sim_threshold:.2f}."
            ),
            "retrieved_facts_meta": retrieved,
        }

    # Otherwise, let the LLM judge based on retrieved facts
    llm_result = judge_claim(claim, retrieved)

    return {
        "claim": claim,
        "verdict": llm_result.get("verdict", "Unverifiable"),
        "confidence": confidence,
        "max_similarity": max_sim,
        "evidence": llm_result.get("evidence", []),
        "reasoning": llm_result.get("reasoning", ""),
        "retrieved_facts_meta": retrieved,
    }


def fact_check_text(
    text: str,
    k: int = 5,
    sim_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Main multi-claim entry point.
    - extract all claims
    - run fact check per claim
    """
    claims = extract_claims(text)

    if not claims:
        return {
            "input_text": text,
            "results": [
                {
                    "claim": None,
                    "verdict": "Unverifiable",
                    "confidence": 0.0,
                    "max_similarity": 0.0,
                    "evidence": [],
                    "reasoning": "Could not extract any clear claim from the input text.",
                    "retrieved_facts_meta": [],
                }
            ],
        }

    per_claim_results: List[Dict[str, Any]] = []
    for c in claims:
        per_claim_results.append(
            fact_check_single_claim(c, k=k, sim_threshold=sim_threshold)
        )

    return {
        "input_text": text,
        "results": per_claim_results,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM-powered multi-claim fact checker")
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="Input statement or news text (wrap in quotes). If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k facts to retrieve per claim.",
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.4,
        help="Similarity threshold below which claims are marked Unverifiable.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON file path.",
    )
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        text = input("Enter text or claim(s): ")

    result = fact_check_text(
        text,
        k=args.top_k,
        sim_threshold=args.sim_threshold,
    )
    result_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(result_json)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result_json)


if __name__ == "__main__":
    main()
