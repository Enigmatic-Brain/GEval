"""
Confusion Matrix Evaluator v2 — OpenAI-compatible backend
==========================================================
Evaluates model responses using two independent metrics:

  Recall (Completeness):
    How many of the claims the document says are needed to answer the question
    did the model actually cover — counting only Must Have response claims.

    recall = TP / total_doc_claims
      where TP = doc claims matched by a Must Have response claim only.

  Precision (Relevance):
    Of all claims the model made, what fraction were on-topic relative to
    the question (Must Have + Good to Have), vs. off-topic (Irrelevant).

    precision = (Must Have + Good to Have) / (Must Have + Good to Have + Irrelevant)

Pipeline:
  Step 1 — Combined extraction (single LLM call):
    Given question + document + model response, extract:
      - doc_claims:      essential claims from the document needed to answer
                         the question (tightened — no labels, every extracted
                         claim is required by definition)
      - response_claims: every atomic claim from the model response, each
                         classified relative to the question as:
                           Must Have  — directly answers the question
                           Good to Have — on-topic but not strictly required
                           Irrelevant — off-topic relative to the question

  Step 2 — Align Must Have claims to doc claims (single LLM call):
    Only Must Have response claims participate in bipartite matching.
    Good to Have and Irrelevant claims never enter the matcher.
    Returns: tp_pairs, fn_claim_ids.
    No FP step — irrelevance is captured by precision formula directly.

  Step 3 — Local math (no LLM):
    recall    = TP / len(doc_claims)
    precision = (must_have + good_to_have) / total_response_claims

No TN, no F1, no accuracy.
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
# In-memory cache
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
Extract ONLY the claims from the Source Document that are directly and \
necessarily required to fully answer the Question. These are the facts a \
complete, correct answer cannot omit.

Rules:
1. Extract ONLY claims whose content appears explicitly in the document — do \
not add outside knowledge or infer facts not stated.
2. Each claim must be atomic — one distinct fact per claim. Do not bundle \
two facts into one claim (e.g. "rate is 11.5% and tenure is 12-60 months" \
must be two separate claims).
3. Be strict — omit background context, prerequisites, and supplementary \
detail that the answer does not require. Every claim you extract should be \
something a reader would miss if it were absent from the answer.

--- TASK B: Response Claims ---
Extract every distinct factual claim from the Model Response and classify \
each one relative to the Question:

  - "Must Have":    This claim directly answers the question. A complete \
answer requires it.
  - "Good to Have": This claim is on-topic and adds useful detail but the \
answer would still be considered complete without it.
  - "Irrelevant":   This claim does not relate to the question being asked.

Rules:
1. Each claim must be atomic — one fact per claim.
2. Include all specific values, numbers, rates, conditions, and terms stated.
3. Classify based on the question and the source document — use the document \
to judge whether the claim is relevant and necessary to answer the question.
4. Do not verify or judge accuracy — just extract and classify faithfully.

Call the `submit_all_claims` function with both lists.
"""

ALIGNMENT_PROMPT = """\
You are an expert evaluator. Your task is to match Must Have response claims \
to document claims.

**Document Claims** (what the answer must cover):
{doc_claims_json}

**Must Have Response Claims** (what the model said that directly answers the question):
{must_have_claims_json}

Match each Must Have Response Claim to the Document Claim it expresses the \
same fact as.

Rules:
- Matching is semantic — different wording of the same fact counts as a match.
- Numbers, percentages, dates, and thresholds must match EXACTLY. A wrong \
number is NOT a match even if the topic is the same.
- Each Document Claim can match at most one Response Claim (strict 1:1).
- Each Response Claim can match at most one Document Claim.
- Matched pairs go into `tp_pairs`.
- Document Claim IDs with no matching Must Have response claim go into \
`fn_claim_ids`.

Call the `submit_alignment` function with your answer.
"""

# ---------------------------------------------------------------------------
# Function schemas
# ---------------------------------------------------------------------------

COMBINED_EXTRACTION_FUNCTION = {
    "type": "function",
    "function": {
        "name": "submit_all_claims",
        "description": "Submit document claims and classified response claims extracted in one pass.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_claims": {
                    "type": "array",
                    "description": "Claims from the source document that are essential to answer the question.",
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
                    "description": "Atomic claims from the model response, each classified relative to the question.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id":  {"type": "integer"},
                            "claim_text": {"type": "string"},
                            "category": {
                                "type": "string",
                                "enum": ["Must Have", "Good to Have", "Irrelevant"],
                            },
                        },
                        "required": ["claim_id", "claim_text", "category"],
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
        "description": "Submit bipartite matching between doc claims and Must Have response claims.",
        "parameters": {
            "type": "object",
            "properties": {
                "tp_pairs": {
                    "type": "array",
                    "description": "Doc claim covered by a Must Have response claim.",
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
                    "description": "Doc claim IDs not covered by any Must Have response claim.",
                    "items": {"type": "integer"},
                },
            },
            "required": ["tp_pairs", "fn_claim_ids"],
        },
    },
}

# ---------------------------------------------------------------------------
# Low-level: raw POST to OpenAI chat completions with function calling
# ---------------------------------------------------------------------------

async def call_openai_with_function(
    session:      aiohttp.ClientSession,
    api_key:      str,
    prompt:       str,
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

    message    = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls", [])
    if not tool_calls:
        raise RuntimeError(f"No tool_calls in response: {message}")

    return json.loads(tool_calls[0]["function"]["arguments"])

# ---------------------------------------------------------------------------
# Step 1: Extract doc claims + categorised response claims (with cache)
# ---------------------------------------------------------------------------

async def extract_all_claims(
    session:        aiohttp.ClientSession,
    api_key:        str,
    question:       str,
    document:       str,
    model_response: str,
) -> tuple[list[dict], list[dict]]:
    """
    Single LLM call that extracts and classifies both claim sets.
    Returns: (doc_claims, response_claims)
      doc_claims:      [{claim_id, claim}]
      response_claims: [{claim_id, claim_text, category: Must Have|Good to Have|Irrelevant}]
    Cached by sha256("cm_v2_combined||question||document||model_response").
    """
    cache_key = hashlib.sha256(
        f"cm_v2_combined||{question}||{document}||{model_response}".encode()
    ).hexdigest()
    if cache_key in _extraction_cache:
        log.info("Step 1: Claims loaded from cache.")
        cached = _extraction_cache[cache_key]
        return cached["doc_claims"], cached["response_claims"]

    log.info("Step 1: Extracting doc claims and categorising response claims...")
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

    log.info(f"  {len(response_claims)} response claims extracted")
    for c in response_claims:
        log.info(f"  R{c['claim_id']} [{c['category']}]: {c['claim_text']}")

    _extraction_cache[cache_key] = {"doc_claims": doc_claims, "response_claims": response_claims}
    return doc_claims, response_claims

# ---------------------------------------------------------------------------
# Step 2: Align Must Have response claims to doc claims (with retry)
# ---------------------------------------------------------------------------

async def align_must_have_claims(
    session:          aiohttp.ClientSession,
    api_key:          str,
    doc_claims:       list[dict],
    response_claims:  list[dict],
    max_retries:      int = MAX_RETRIES,
) -> dict:
    """
    Bipartite matching between doc claims and Must Have response claims only.
    Good to Have and Irrelevant claims never enter the matcher.
    Returns: {tp_pairs, fn_claim_ids}

    On permanent failure falls back conservatively:
      all doc claims → FN, empty tp_pairs.
    """
    must_have_claims = [c for c in response_claims if c["category"] == "Must Have"]

    if not must_have_claims:
        log.warning("Step 2: No Must Have response claims — all doc claims are FN.")
        return {
            "tp_pairs":     [],
            "fn_claim_ids": [c["claim_id"] for c in doc_claims],
        }

    doc_claims_json = json.dumps(
        [{"claim_id": c["claim_id"], "claim": c["claim"]} for c in doc_claims],
        indent=2,
    )
    must_have_claims_json = json.dumps(
        [{"claim_id": c["claim_id"], "claim_text": c["claim_text"]} for c in must_have_claims],
        indent=2,
    )

    prompt = ALIGNMENT_PROMPT.format(
        doc_claims_json=doc_claims_json,
        must_have_claims_json=must_have_claims_json,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            log.info(
                f"Step 2: Aligning {len(doc_claims)} doc claims "
                f"with {len(must_have_claims)} Must Have response claims..."
            )
            result = await call_openai_with_function(
                session, api_key, prompt, ALIGNMENT_FUNCTION
            )
            log.info(
                f"  TP={len(result['tp_pairs'])}, "
                f"FN={len(result['fn_claim_ids'])}"
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
        "tp_pairs":     [],
        "fn_claim_ids": [c["claim_id"] for c in doc_claims],
    }

# ---------------------------------------------------------------------------
# Step 3: Compute metrics (local math, no LLM)
# ---------------------------------------------------------------------------

def compute_metrics(
    doc_claims:      list[dict],
    response_claims: list[dict],
    alignment:       dict,
) -> dict:
    """
    Pure math — no I/O.

    recall    = TP / total_doc_claims
      TP = doc claims covered by a Must Have response claim only.

    precision = (must_have + good_to_have) / total_response_claims
      Purely response-side; measures how on-topic the response is.

    Edge cases:
      total_doc_claims == 0    → recall    = 0.0
      total_response_claims == 0 → precision = 0.0
    """
    doc_by_id      = {c["claim_id"]: c for c in doc_claims}
    response_by_id = {c["claim_id"]: c for c in response_claims}

    # Recall — Must Have matching only
    tp = len(alignment["tp_pairs"])
    fn = len(alignment["fn_claim_ids"])
    total_doc = len(doc_claims)

    recall = tp / total_doc if total_doc > 0 else 0.0

    # Precision — response-side category counts
    must_have_count    = sum(1 for c in response_claims if c["category"] == "Must Have")
    good_to_have_count = sum(1 for c in response_claims if c["category"] == "Good to Have")
    irrelevant_count   = sum(1 for c in response_claims if c["category"] == "Irrelevant")
    total_response     = must_have_count + good_to_have_count + irrelevant_count

    precision = (must_have_count + good_to_have_count) / total_response if total_response > 0 else 0.0

    precision_label = "Relevant"  if precision >= 0.70 else "Not Relevant"
    recall_label    = "Complete"  if recall    >= 0.70 else "Incomplete"

    # Build detail lists
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

    must_have_claims    = [c for c in response_claims if c["category"] == "Must Have"]
    good_to_have_claims = [c for c in response_claims if c["category"] == "Good to Have"]
    irrelevant_claims   = [c for c in response_claims if c["category"] == "Irrelevant"]

    return {
        # Core scores
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "precision_label":    precision_label,
        "recall_label":       recall_label,

        # Recall counts
        "tp":                 tp,
        "fn":                 fn,
        "total_doc_claims":   total_doc,

        # Precision counts
        "must_have_count":    must_have_count,
        "good_to_have_count": good_to_have_count,
        "irrelevant_count":   irrelevant_count,

        # Detail lists
        "tp_claims":          tp_claims,
        "fn_claims":          fn_claims,
        "must_have_claims":   must_have_claims,
        "good_to_have_claims": good_to_have_claims,
        "irrelevant_claims":  irrelevant_claims,
        "doc_claims":         doc_claims,
        "response_claims":    response_claims,
    }

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def evaluate_confusion_matrix_v2(
    question:       str,
    model_response: str,
    document:       str,
    api_key:        Optional[str] = None,
) -> dict:
    """
    Full v2 confusion matrix evaluation pipeline.

    Args:
        question:       The original user question.
        model_response: The generated answer to evaluate.
        document:       The source document the answer should be based on.
        api_key:        OpenAI API key (falls back to OPENAI_API_KEY env var).

    Returns a dict with:
        precision             — (must_have + good_to_have) / total_response, Relevance
        recall                — TP / total_doc_claims, Completeness
        precision_label       — "Relevant" | "Not Relevant"
        recall_label          — "Complete" | "Incomplete"
        tp                    — doc claims covered by Must Have response claims
        fn                    — doc claims not covered by any Must Have response claim
        total_doc_claims      — total doc claims extracted
        must_have_count       — Must Have response claims count
        good_to_have_count    — Good to Have response claims count
        irrelevant_count      — Irrelevant response claims count
        tp_claims             — matched pairs (doc + response claim text)
        fn_claims             — doc claims missed by Must Have response claims
        must_have_claims      — Must Have response claims list
        good_to_have_claims   — Good to Have response claims list
        irrelevant_claims     — Irrelevant response claims list
        doc_claims            — raw extracted doc claims
        response_claims       — raw extracted response claims (with category)
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required — set OPENAI_API_KEY or pass api_key=...")

    async with aiohttp.ClientSession() as session:
        # Step 1: extract and classify both claim sets in a single call
        doc_claims, response_claims = await extract_all_claims(
            session, api_key, question, document, model_response
        )

        if not doc_claims:
            raise ValueError("No doc claims extracted — check your document and question.")
        if not response_claims:
            raise ValueError("No response claims extracted — check your model response.")

        # Step 2: align Must Have response claims to doc claims
        alignment = await align_must_have_claims(
            session, api_key, doc_claims, response_claims
        )

    # Step 3: local math
    return compute_metrics(doc_claims, response_claims, alignment)

# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(result: dict) -> None:
    print("\n" + "=" * 65)
    print("    CONFUSION MATRIX EVALUATION REPORT v2  (OpenAI)")
    print("=" * 65)
    print(f"  Precision (Relevance)   : {result['precision']:.2f}  →  {result['precision_label']}")
    print(f"  Recall    (Completeness): {result['recall']:.2f}  →  {result['recall_label']}")

    print(f"\n  Recall  : TP={result['tp']}  |  FN={result['fn']}  |  Total Doc Claims={result['total_doc_claims']}")
    print(
        f"  Precision: Must Have={result['must_have_count']}  |  "
        f"Good to Have={result['good_to_have_count']}  |  "
        f"Irrelevant={result['irrelevant_count']}"
    )

    if result["fn_claims"]:
        print(f"\n  ❌ Missed Doc Claims — FN ({result['fn']}):")
        for c in result["fn_claims"]:
            print(f"     • {c['doc_claim']}")

    if result["irrelevant_claims"]:
        print(f"\n  ⚠️  Irrelevant Response Claims ({result['irrelevant_count']}):")
        for c in result["irrelevant_claims"]:
            print(f"     • {c['claim_text']}")

    if result["tp_claims"]:
        print(f"\n  ✅ Covered Doc Claims — TP ({result['tp']}):")
        for c in result["tp_claims"]:
            print(f"     • {c['doc_claim']}")

    if result["good_to_have_claims"]:
        print(f"\n  ℹ  Good to Have Claims ({result['good_to_have_count']}):")
        for c in result["good_to_have_claims"]:
            print(f"     • {c['claim_text']}")

    print("=" * 65 + "\n")

# ---------------------------------------------------------------------------
# Demo
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

    # Intentionally imperfect: wrong processing fee (2% vs 1.5%),
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
        result = await evaluate_confusion_matrix_v2(
            question=QUESTION,
            model_response=MODEL_RESPONSE,
            document=DOCUMENT,
        )
        print_report(result)

        with open("confusion_matrix_v2_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Full results saved to confusion_matrix_v2_result.json")

    asyncio.run(main())
