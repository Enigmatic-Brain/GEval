"""
Confusion Matrix Evaluator — OpenAI-compatible backend
=======================================================
Unifies Completeness (Recall) and Relevance (Precision) into a single
confusion-matrix framework using TP, FP, and FN.

  TP = expected answer claim covered by the response
  FN = expected answer claim missed by the response
  FP = response claim not supported by the source document (hallucination)

Pipeline:
  Step 1 — Combined extraction (single LLM call):
    Given question + document + model response, extract:
      - doc_claims:      relevant claims from the document
      - response_claims: atomic claims from the model response

  Step 2 — Align and classify (single LLM call):
    Match doc claims to response claims (1:1 bipartite) and
    inline-verify unmatched response claims as FP or valid extra detail.

  Step 3 — Local math (no LLM):
    precision = TP / (TP + FP)   → Relevance   (is the response on-point?)
    recall    = TP / (TP + FN)   → Completeness (did it cover everything?)

No TN, no F1, no accuracy. Designed for large-context models (e.g. Gemini).
"""

import asyncio
import hashlib
import json
import logging
import os
from typing import Optional

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_URL  = "https://api.openai.com/v1/chat/completions"
MODEL           = "gpt-4o"
MAX_TOKENS      = 4096
MAX_RETRIES     = 3
TIMEOUT_SECONDS = 90

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache  {sha256(question||document||model_response): {doc_claims, response_claims}}
# ---------------------------------------------------------------------------

_extraction_cache: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

COMBINED_EXTRACTION_PROMPT = """\
You are an expert information analyst. Given a question, a source document, \
and a model response, your job is to extract two independent lists of claims.

**Question:** {question}

**Source Document:**
{document}

**Model Response:**
{model_response}

--- TASK A: Document Claims ---
Extract every claim from the Source Document that is relevant to answering \
the Question.
1. Extract ONLY claims whose content appears explicitly in the document — do \
not add outside knowledge or infer facts not stated.
2. Each claim must be atomic — one distinct fact per claim. Do not bundle \
two facts into one claim (e.g. "rate is 11.5% and tenure is 12-60 months" \
must be two separate claims).
3. Be exhaustive — capture every figure, rate, threshold, condition, deadline, \
exception, and procedural step that helps answer the question.

--- TASK B: Response Claims ---
Extract every distinct factual claim from the Model Response.
1. Each claim must be atomic — one fact per claim.
2. Include all specific values, numbers, rates, conditions, and terms stated.
3. Do not verify or judge the claims — just extract them faithfully.

Call the `submit_all_claims` function with both lists.
"""

ALIGNMENT_PROMPT = """\
You are an expert evaluator performing two tasks:

**Task 1 — Match claims (TP and FN)**

Document Claims (what the answer should cover):
{doc_claims_json}

Response Claims (what the answer actually said):
{response_claims_json}

Match each Response Claim to the Document Claim it expresses the same fact as.
Rules:
- Matching is semantic — different wording of the same fact counts as a match.
- Numbers, percentages, dates, and thresholds must match EXACTLY for a match. \
A wrong number is NOT a match even if the topic is the same.
- Each Document Claim can match at most one Response Claim (strict 1:1).
- Each Response Claim can match at most one Document Claim.
- Matched pairs go into `tp_pairs`.
- Document Claim IDs with no match go into `fn_claim_ids` (missed by response).

**Task 2 — Rescue or reject unmatched Response Claims**

For each Response Claim not matched in Task 1, check whether it matches \
any Document Claim that is still unmatched (same rules as Task 1 — \
meaning-level match, numbers must be exact):
- If it matches an unmatched Document Claim: move the pair to `tp_pairs` \
and remove that Doc Claim ID from `fn_claim_ids`. This rescues alignment \
misses from Task 1.
- If it matches no Document Claim: it is outside the scope of the expected \
answer → add to `fp_claims`.

Only use the Document Claims list above — do NOT check against the raw \
source document. Anything not in the Document Claims is FP.

Call the `submit_alignment` function with your answer.
"""

# ---------------------------------------------------------------------------
# Function schemas
# ---------------------------------------------------------------------------

COMBINED_EXTRACTION_FUNCTION = {
    "type": "function",
    "function": {
        "name": "submit_all_claims",
        "description": "Submit document claims and response claims extracted in one pass.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_claims": {
                    "type": "array",
                    "description": "Claims extracted from the source document relevant to the question.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "integer"},
                            "claim":    {"type": "string"},
                        },
                        "required": ["claim_id", "claim"],
                    },
                },
                "response_claims": {
                    "type": "array",
                    "description": "Atomic claims extracted from the model response.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id":   {"type": "integer"},
                            "claim_text": {"type": "string"},
                        },
                        "required": ["claim_id", "claim_text"],
                    },
                },
            },
            "required": ["doc_claims", "response_claims"],
        },
    },
}

ALIGNMENT_FUNCTION = {
    "type": "function",
    "function": {
        "name": "submit_alignment",
        "description": "Submit the alignment between document claims and response claims.",
        "parameters": {
            "type": "object",
            "properties": {
                "tp_pairs": {
                    "type": "array",
                    "description": "Matched pairs: doc claim covered by response claim (TP).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doc_claim_id":      {"type": "integer"},
                            "response_claim_id": {"type": "integer"},
                        },
                        "required": ["doc_claim_id", "response_claim_id"],
                    },
                },
                "fn_claim_ids": {
                    "type": "array",
                    "description": "Doc claim IDs not covered by any response claim (FN — missed).",
                    "items": {"type": "integer"},
                },
                "fp_claims": {
                    "type": "array",
                    "description": "Unmatched response claims with no matching doc claim (FP — outside expected answer scope).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "response_claim_id": {"type": "integer"},
                            "claim_text":        {"type": "string"},
                            "reason":            {"type": "string"},
                        },
                        "required": ["response_claim_id", "claim_text", "reason"],
                    },
                },
            },
            "required": ["tp_pairs", "fn_claim_ids", "fp_claims"],
        },
    },
}

# ---------------------------------------------------------------------------
# Low-level: raw POST to OpenAI chat completions with function calling
# ---------------------------------------------------------------------------

async def call_openai_with_function(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt: str,
    function_def: dict,
) -> dict:
    """
    POST to OpenAI /v1/chat/completions with forced function calling.
    Returns the parsed dict from the function call arguments.
    Raises on HTTP errors or missing tool call.
    """
    function_name = function_def["function"]["name"]

    payload = {
        "model":       MODEL,
        "max_tokens":  MAX_TOKENS,
        "temperature": 0,
        "messages":    [{"role": "user", "content": prompt}],
        "tools":       [function_def],
        "tool_choice": {
            "type":     "function",
            "function": {"name": function_name},
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
    async with session.post(
        OPENAI_API_URL, json=payload, headers=headers, timeout=timeout
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"OpenAI API error {resp.status}: {body}")
        data = await resp.json()

    message   = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls", [])
    if not tool_calls:
        raise RuntimeError(f"No tool_calls in response: {message}")

    return json.loads(tool_calls[0]["function"]["arguments"])

# ---------------------------------------------------------------------------
# Step 1: Extract doc claims + response claims in one combined call (with cache)
# ---------------------------------------------------------------------------

async def extract_all_claims(
    session:        aiohttp.ClientSession,
    api_key:        str,
    question:       str,
    document:       str,
    model_response: str,
) -> tuple[list[dict], list[dict]]:
    """
    Single LLM call that extracts both claim sets simultaneously.
    Returns: (doc_claims, response_claims)
      doc_claims:      [{claim_id, claim, importance: Critical|Supporting}]
      response_claims: [{claim_id, claim_text}]
    Cached by sha256("cm_combined_v1||question||document||model_response").
    """
    cache_key = hashlib.sha256(
        f"cm_combined_v1||{question}||{document}||{model_response}".encode()
    ).hexdigest()
    if cache_key in _extraction_cache:
        log.info("Step 1: Claims loaded from cache.")
        cached = _extraction_cache[cache_key]
        return cached["doc_claims"], cached["response_claims"]

    log.info("Step 1: Extracting doc claims and response claims (combined call)...")
    prompt = COMBINED_EXTRACTION_PROMPT.format(
        question=question, document=document, model_response=model_response
    )
    result = await call_openai_with_function(
        session, api_key, prompt, COMBINED_EXTRACTION_FUNCTION
    )

    doc_claims      = result["doc_claims"]
    response_claims = result["response_claims"]

    log.info(f"  {len(doc_claims)} doc claims extracted")
    for c in doc_claims:
        log.info(f"  D{c['claim_id']}: {c['claim']}")
    log.info(f"  {len(response_claims)} response claims")
    for c in response_claims:
        log.info(f"  R{c['claim_id']}: {c['claim_text']}")

    _extraction_cache[cache_key] = {"doc_claims": doc_claims, "response_claims": response_claims}
    return doc_claims, response_claims

# ---------------------------------------------------------------------------
# Step 3: Align and classify (with retry)
# ---------------------------------------------------------------------------

async def align_and_classify(
    session:         aiohttp.ClientSession,
    api_key:         str,
    doc_claims:      list[dict],
    response_claims: list[dict],
    max_retries:     int = MAX_RETRIES,
) -> dict:
    """
    Bipartite claim matching + rescue pass in a single LLM call.
    Task 1: match response claims to doc claims → TP / FN.
    Task 2: check unmatched response claims against doc claims only → rescue TP or FP.
    Returns: {tp_pairs, fn_claim_ids, fp_claims}

    On permanent failure falls back conservatively: all doc claims → FN,
    all response claims → FP.
    """
    doc_claims_json = json.dumps(
        [{"claim_id": c["claim_id"], "claim": c["claim"]} for c in doc_claims],
        indent=2,
    )
    response_claims_json = json.dumps(
        [{"claim_id": c["claim_id"], "claim_text": c["claim_text"]}
         for c in response_claims],
        indent=2,
    )

    prompt = ALIGNMENT_PROMPT.format(
        doc_claims_json=doc_claims_json,
        response_claims_json=response_claims_json,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            log.info(
                f"Step 3: Aligning {len(doc_claims)} doc claims "
                f"with {len(response_claims)} response claims..."
            )
            result = await call_openai_with_function(session, api_key, prompt, ALIGNMENT_FUNCTION)
            log.info(
                f"  TP={len(result['tp_pairs'])}, "
                f"FN={len(result['fn_claim_ids'])}, "
                f"FP={len(result['fp_claims'])}"
            )
            return result
        except Exception as exc:
            last_error = exc
            wait = 2 ** attempt
            log.warning(
                f"  Alignment failed (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {wait}s: {exc}"
            )
            await asyncio.sleep(wait)

    log.error(f"  Alignment permanently failed: {last_error}")
    return {
        "tp_pairs":          [],
        "fn_claim_ids":      [c["claim_id"] for c in doc_claims],
        "fp_claims":         [
            {
                "response_claim_id": c["claim_id"],
                "claim_text":        c["claim_text"],
                "reason":            f"Alignment failed: {last_error}",
            }
            for c in response_claims
        ],
    }

# ---------------------------------------------------------------------------
# Step 4: Compute metrics (local math, no LLM)
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    doc_claims:      list[dict],
    response_claims: list[dict],
    alignment:       dict,
) -> dict:
    """
    Pure math — no I/O.

    precision = TP / (TP + FP)   → Relevance
    recall    = TP / (TP + FN)   → Completeness

    Edge cases:
      TP + FP == 0  → precision = 1.0  (no false positives is perfect relevance)
      TP + FN == 0  → recall    = 1.0  (no expected claims means nothing to miss)
    """
    tp = len(alignment["tp_pairs"])
    fn = len(alignment["fn_claim_ids"])
    fp = len(alignment["fp_claims"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    precision_label = "Relevant"  if precision >= 0.70 else "Not Relevant"
    recall_label    = "Complete"  if recall    >= 0.70 else "Incomplete"

    doc_by_id      = {c["claim_id"]: c for c in doc_claims}
    response_by_id = {c["claim_id"]: c for c in response_claims}

    tp_claims = [
        {
            "doc_claim_id":      pair["doc_claim_id"],
            "response_claim_id": pair["response_claim_id"],
            "doc_claim":         doc_by_id.get(pair["doc_claim_id"], {}).get("claim", ""),
            "response_claim":    response_by_id.get(pair["response_claim_id"], {}).get("claim_text", ""),
        }
        for pair in alignment["tp_pairs"]
    ]

    fn_claims = [
        {
            "doc_claim_id": cid,
            "doc_claim":    doc_by_id.get(cid, {}).get("claim", ""),
        }
        for cid in alignment["fn_claim_ids"]
    ]

    return {
        "tp":              tp,
        "fp":              fp,
        "fn":              fn,
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "precision_label": precision_label,
        "recall_label":    recall_label,
        "tp_claims":       tp_claims,
        "fp_claims":       alignment["fp_claims"],
        "fn_claims":       fn_claims,
        "doc_claims":      doc_claims,
        "response_claims": response_claims,
    }

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def evaluate_confusion_matrix(
    question:       str,
    model_response: str,
    document:       str,
    api_key:        Optional[str] = None,
) -> dict:
    """
    Full confusion matrix evaluation pipeline.

    Args:
        question:       The original user question.
        model_response: The generated answer to evaluate.
        document:       The source document the answer should be based on.
        api_key:        OpenAI API key (falls back to OPENAI_API_KEY env var).

    Returns a dict with:
        tp, fp, fn            — confusion matrix counts
        precision             — TP / (TP + FP), Relevance score
        recall                — TP / (TP + FN), Completeness score
        precision_label       — "Relevant" | "Not Relevant"
        recall_label          — "Complete" | "Incomplete"
        tp_claims             — matched pairs (doc + response claim text)
        fp_claims             — response claims outside expected answer scope
        fn_claims             — doc claims missed by the response
        doc_claims            — raw extracted doc claims
        response_claims       — raw extracted response claims
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required — set OPENAI_API_KEY or pass api_key=...")

    async with aiohttp.ClientSession() as session:
        # Step 1: extract both claim sets in a single call
        doc_claims, response_claims = await extract_all_claims(
            session, api_key, question, document, model_response
        )

        if not doc_claims:
            raise ValueError("No doc claims extracted — check your document and question.")
        if not response_claims:
            raise ValueError("No response claims extracted — check your model response.")

        # Step 3: align and classify
        alignment = await align_and_classify(
            session, api_key, doc_claims, response_claims
        )

    # Step 4: local math
    return compute_confusion_matrix(doc_claims, response_claims, alignment)

# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(result: dict) -> None:
    print("\n" + "=" * 65)
    print("      CONFUSION MATRIX EVALUATION REPORT  (OpenAI)")
    print("=" * 65)
    print(f"  Precision (Relevance)   : {result['precision']:.2f}  →  {result['precision_label']}")
    print(f"  Recall    (Completeness): {result['recall']:.2f}  →  {result['recall_label']}")
    print(f"\n  TP={result['tp']}  |  FP={result['fp']}  |  FN={result['fn']}")

    if result["fn_claims"]:
        print(f"\n  ❌ Missed Claims — FN ({result['fn']}):")
        for c in result["fn_claims"]:
            print(f"     • {c['doc_claim']}")

    if result["fp_claims"]:
        print(f"\n  ⚠️  Hallucinations — FP ({result['fp']}):")
        for c in result["fp_claims"]:
            print(f"     • {c['claim_text']}")
            print(f"       Reason: {c['reason']}")

    if result["tp_claims"]:
        print(f"\n  ✅ Covered Claims — TP ({result['tp']}):")
        for c in result["tp_claims"]:
            print(f"     • {c['doc_claim']}")

    print("=" * 65 + "\n")

# ---------------------------------------------------------------------------
# Demo  (uses same NorthStar Bank fixture as groundedness_evaluator_openai.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    QUESTION = "What are the loan terms and prepayment rules for NorthStar Bank personal loans?"

    DOCUMENT = """
    PERSONAL LOAN POLICY — NORTHSTAR BANK

    Loan amount: ₹50,000 to ₹40,00,000. Tenure: 12-60 months.
    Interest rate: 11.5% p.a. fixed or MCLR + 2.75% floating.
    Processing fee: 1.5% of loan amount (non-refundable).

    EMI is calculated on reducing balance. Bounce charge: ₹500 per instance.
    Penal interest: 2% per month on overdue amount.

    Prepayment: Not allowed in first 6 months. Part-prepayment (up to 25% of
    outstanding) allowed once per year after 6 months, no charge. Full
    foreclosure: 3% charge if closed in months 7-24, 2% thereafter. Written
    application required 15 days in advance.

    Default: 3 missed EMIs = default. On default, full outstanding becomes due.
    Bank reports to CIBIL, Equifax, Experian, CRIF and may initiate legal
    proceedings under SARFAESI Act.

    Credit life insurance equal to outstanding amount is mandatory throughout
    the loan tenure. Premium may be financed within the loan.
    """

    # Intentionally imperfect response: wrong processing fee (2% vs 1.5%),
    # wrong prepayment window (3 months vs 6 months), fabricated grace period.
    MODEL_RESPONSE = """
    NorthStar Bank offers personal loans from ₹50,000 to ₹40,00,000 with
    a tenure of 12 to 60 months. The interest rate is 11.5% per annum fixed
    or MCLR + 2.75% floating. A processing fee of 2% is charged upfront.

    EMI is calculated on a reducing balance basis. There is a bounce charge
    of ₹500 per instance. Prepayment is not allowed during the first
    3 months. After that, part-prepayment up to 25% is allowed once a year.

    If 3 EMIs are missed, it is considered a default and the bank reports
    to credit bureaus. A 30-day grace period is provided before any action.
    """

    async def main():
        result = await evaluate_confusion_matrix(
            question=QUESTION,
            model_response=MODEL_RESPONSE,
            document=DOCUMENT,
        )
        print_report(result)

        with open("confusion_matrix_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Full results saved to confusion_matrix_result.json")

    asyncio.run(main())
