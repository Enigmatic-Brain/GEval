"""
Groundedness Evaluation Pipeline — OpenAI ChatGPT backend
==========================================================
Extracts claims from the model response and verifies each against the
source document in a **single LLM call**.

Designed for large-context models (e.g. Gemini Flash) where one call is
cheaper and faster than multiple batched calls.

Scoring (strict binary):
  Supported = 1 | Not Supported = 0
  groundedness_score = supported_claims / total_claims
  Label: >= 0.70 → "Grounded", else "Not Grounded"

Numbers must be exact — wrong numbers are "Not Supported", not partial.
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
# In-memory cache  {sha256(model_response||document): result_dict}
# ---------------------------------------------------------------------------

_groundedness_cache: dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

GROUNDEDNESS_PROMPT = """\
You are an expert fact-checker. Your job is to extract every factual claim \
from the **Model Response** and verify whether the **Source Document** \
supports it.

**Model Response:**
{model_response}

**Source Document:**
{document}

Instructions:
1. Extract every distinct, atomic factual claim from the Model Response. \
Each claim must contain exactly one fact — do not bundle multiple facts \
into a single claim.
2. For each claim, check whether the Source Document supports it.
3. Assign a verification status — strictly binary:
   - "Supported": The document explicitly states this fact, or states \
information that is clearly equivalent (different wording, same meaning). \
Numbers, percentages, thresholds, dates, and conditions MUST match exactly. \
Paraphrasing is fine but numeric values must be precise.
   - "Not Supported": The document does not contain this information, \
the claim contradicts the document, OR the claim uses a wrong number / \
percentage / threshold / date (even if the topic is mentioned). A wrong \
number is NOT supported — there is no partial credit.
4. In your reason, quote or reference the specific part of the Source \
Document that supports or contradicts the claim. If the claim is not \
found in the document, state that clearly.
5. Match by MEANING, not exact wording. Paraphrasing and synonyms are \
acceptable for non-numeric content. But all numbers, rates, fees, \
thresholds, and dates must be exact.

Call the `submit_groundedness_results` function with your answer.
"""

# ---------------------------------------------------------------------------
# Function schema
# ---------------------------------------------------------------------------

GROUNDEDNESS_FUNCTION = {
    "type": "function",
    "function": {
        "name": "submit_groundedness_results",
        "description": "Submit extracted claims and their verification against the source document.",
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id":   {"type": "integer"},
                            "claim_text": {"type": "string"},
                            "verification_status": {
                                "type": "string",
                                "enum": ["Supported", "Not Supported"],
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["claim_id", "claim_text", "verification_status", "reason"],
                    },
                }
            },
            "required": ["claims"],
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

    message = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls", [])
    if not tool_calls:
        raise RuntimeError(f"No tool_calls in response: {message}")

    arguments_str = tool_calls[0]["function"]["arguments"]
    return json.loads(arguments_str)

# ---------------------------------------------------------------------------
# Single-call extraction + verification (with retry)
# ---------------------------------------------------------------------------

async def evaluate_groundedness_with_retry(
    session: aiohttp.ClientSession,
    api_key: str,
    model_response: str,
    document: str,
    max_retries: int = MAX_RETRIES,
) -> list[dict]:
    cache_key = hashlib.sha256(f"{model_response}||{document}".encode()).hexdigest()
    if cache_key in _groundedness_cache:
        log.info("Groundedness: Claims loaded from cache.")
        return _groundedness_cache[cache_key]

    prompt = GROUNDEDNESS_PROMPT.format(
        model_response=model_response,
        document=document,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            log.info("Groundedness: Extracting and verifying claims (single call)...")
            result = await call_openai_with_function(
                session, api_key, prompt, GROUNDEDNESS_FUNCTION
            )
            claims = result["claims"]
            log.info(f"  Extracted and verified {len(claims)} claims.")
            for c in claims:
                log.info(f"  [{c['verification_status']}] Claim {c['claim_id']}: {c['claim_text']}")

            _groundedness_cache[cache_key] = claims
            return claims
        except Exception as exc:
            last_error = exc
            wait = 2 ** attempt
            log.warning(
                f"  Groundedness call failed (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {wait}s: {exc}"
            )
            await asyncio.sleep(wait)

    raise RuntimeError(
        f"Groundedness evaluation failed after {max_retries} retries: {last_error}"
    )

# ---------------------------------------------------------------------------
# Compute score locally
# ---------------------------------------------------------------------------

def compute_groundedness_score(claims: list[dict]) -> dict:
    if not claims:
        return {
            "groundedness_score":      0.0,
            "groundedness_label":      "Not Grounded",
            "total_claims":            0,
            "supported_claims":        0,
            "not_supported_claims":    0,
            "unsupported_claim_texts": [],
            "verification_details":    [],
        }

    total       = len(claims)
    supported   = sum(1 for c in claims if c["verification_status"] == "Supported")
    unsupported = total - supported
    score       = supported / total

    unsupported_texts = [
        c["claim_text"] for c in claims if c["verification_status"] == "Not Supported"
    ]

    label = "Grounded" if score >= 0.70 else "Not Grounded"

    return {
        "groundedness_score":      round(score, 4),
        "groundedness_label":      label,
        "total_claims":            total,
        "supported_claims":        supported,
        "not_supported_claims":    unsupported,
        "unsupported_claim_texts": unsupported_texts,
        "verification_details":    claims,
    }

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def evaluate_groundedness(
    model_response: str,
    document: str,
    api_key: Optional[str] = None,
) -> dict:
    """
    Full groundedness pipeline — single LLM call.

    Args:
        model_response: The generated answer to evaluate.
        document:       The source document.
        api_key:        OpenAI API key (falls back to OPENAI_API_KEY env var).
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required — set OPENAI_API_KEY or pass api_key=...")

    async with aiohttp.ClientSession() as session:
        claims = await evaluate_groundedness_with_retry(
            session, api_key, model_response, document
        )

    return compute_groundedness_score(claims)

# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(result: dict) -> None:
    icons = {"Supported": "✅", "Not Supported": "❌"}
    print("\n" + "=" * 65)
    print("         GROUNDEDNESS EVALUATION REPORT")
    print("=" * 65)
    print(
        f"  Overall Score : {result['groundedness_score']:.2f}"
        f"  →  {result['groundedness_label']}"
    )
    print(
        f"  Total Claims  : {result['total_claims']}  "
        f"(Supported: {result['supported_claims']}, "
        f"Not Supported: {result['not_supported_claims']})"
    )

    if result["unsupported_claim_texts"]:
        print(f"\n  ❌ Unsupported Claims ({len(result['unsupported_claim_texts'])}):")
        for c in result["unsupported_claim_texts"]:
            print(f"     • {c}")

    print("\n  Per-Claim Breakdown:")
    print(f"  {'ID':<4} {'Status':<24} Reason")
    print("  " + "-" * 75)
    for c in sorted(result["verification_details"], key=lambda x: x["claim_id"]):
        icon = icons.get(c["verification_status"], "?")
        print(
            f"  {c['claim_id']:<4} {icon} {c['verification_status']:<22} {c['reason']}"
        )
    print("=" * 65 + "\n")

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DOCUMENT = """
    PERSONAL LOAN POLICY — NORTHSTAR BANK

    Loan amount: ₹50,000 to ₹40,00,000. Tenure: 12–60 months.
    Interest rate: 11.5% p.a. fixed or MCLR + 2.75% floating.
    Processing fee: 1.5% of loan amount (non-refundable).

    EMI is calculated on reducing balance. Bounce charge: ₹500 per instance.
    Penal interest: 2% per month on overdue amount.

    Prepayment: Not allowed in first 6 months. Part-prepayment (up to 25% of
    outstanding) allowed once per year after 6 months, no charge. Full
    foreclosure: 3% charge if closed in months 7–24, 2% thereafter. Written
    application required 15 days in advance.

    Default: 3 missed EMIs = default. On default, full outstanding becomes due.
    Bank reports to CIBIL, Equifax, Experian, CRIF and may initiate legal
    proceedings under SARFAESI Act.

    Credit life insurance equal to outstanding amount is mandatory throughout
    the loan tenure. Premium may be financed within the loan.
    """

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
        result = await evaluate_groundedness(
            model_response=MODEL_RESPONSE,
            document=DOCUMENT,
        )
        print_report(result)

    asyncio.run(main())
