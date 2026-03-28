"""
Optimised Completeness Evaluation Pipeline — OpenAI ChatGPT backend
=====================================================================
Calls OpenAI via raw aiohttp POST requests (no SDK).
Uses OpenAI function calling for guaranteed structured output.

Optimisations:
  1. Step 3 LLM call removed  → weighted score computed locally (pure math)
  2. Claim evaluation batched  → N calls reduced to ceil(N / BATCH_SIZE)
  3. Function calling          → no brittle JSON string parsing
  4. Retry with exponential backoff → failed batches are retried, not dropped
  5. Claim extraction cached   → same doc+question never re-extracted twice
"""

import asyncio
import hashlib
import json
import logging
import os

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_URL  = "https://api.openai.com/v1/chat/completions"
MODEL           = "gpt-4o"
MAX_TOKENS      = 2048
MAX_CONCURRENT  = 5       # semaphore cap for parallel batch calls
BATCH_SIZE      = 5       # claims per batch evaluation call
MAX_RETRIES     = 3       # retry attempts per batch on failure
TIMEOUT_SECONDS = 60      # aiohttp request timeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory claim cache  {sha256(question+doc): [claim_dict, ...]}
# ---------------------------------------------------------------------------

_claim_cache: dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are an expert information analyst. Extract all claims from the document \
that are necessary to fully answer the question.

**Question:** {question}
**Document:** {document}

Instructions:
- Break relevant content into atomic claims (one distinct fact per claim).
- Label each as "Critical" (answer meaningfully incomplete without it) or
  "Supporting" (useful detail, not essential).

Call the `submit_claims` function with your answer.
"""

BATCH_COVERAGE_PROMPT = """\
You are an expert evaluator. For each claim below, assess whether the model \
response covers it.

**Question:** {question}

**Model Response:**
{model_response}

**Claims to evaluate:**
{claims_json}

Scoring rubric:
  2 = Fully Covered     — explicitly addressed with sufficient detail
  1 = Partially Covered — touched on but lacking detail or precision
  0 = Not Covered       — not addressed at all

Rules:
- Do not penalise extra information in the response.
- Coverage must be reasonably explicit — do not infer.
- Provide a brief reason per claim referencing the response.

Call the `submit_coverage_results` function with your answer.
"""

# ---------------------------------------------------------------------------
# OpenAI function schemas  (equivalent to Anthropic tool schemas)
# ---------------------------------------------------------------------------

EXTRACTION_FUNCTION = {
    "type": "function",
    "function": {
        "name": "submit_claims",
        "description": "Submit the list of extracted claims.",
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id":   {"type": "integer"},
                            "claim":      {"type": "string"},
                            "importance": {
                                "type": "string",
                                "enum": ["Critical", "Supporting"],
                            },
                        },
                        "required": ["claim_id", "claim", "importance"],
                    },
                }
            },
            "required": ["claims"],
        },
    },
}

COVERAGE_FUNCTION = {
    "type": "function",
    "function": {
        "name": "submit_coverage_results",
        "description": "Submit coverage scores for all evaluated claims.",
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id":       {"type": "integer"},
                            "coverage_score": {"type": "integer", "enum": [0, 1, 2]},
                            "coverage_label": {
                                "type": "string",
                                "enum": [
                                    "Not Covered",
                                    "Partially Covered",
                                    "Fully Covered",
                                ],
                            },
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "claim_id",
                            "coverage_score",
                            "coverage_label",
                            "reason",
                        ],
                    },
                }
            },
            "required": ["results"],
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
    Returns the parsed dict from the function_call arguments.
    Raises on HTTP errors or missing tool call.
    """
    function_name = function_def["function"]["name"]

    payload = {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "messages":   [{"role": "user", "content": prompt}],
        "tools":      [function_def],
        # Force the model to call exactly this function
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

    # Extract the function call arguments
    message = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls", [])
    if not tool_calls:
        raise RuntimeError(f"No tool_calls in response: {message}")

    # arguments is a JSON string — parse it
    arguments_str = tool_calls[0]["function"]["arguments"]
    return json.loads(arguments_str)

# ---------------------------------------------------------------------------
# Step 1: Extract claims  (with in-memory cache)
# ---------------------------------------------------------------------------

async def extract_claims(
    session: aiohttp.ClientSession,
    api_key: str,
    question: str,
    document: str,
) -> list[dict]:
    """
    Returns a list of claim dicts:
      {"claim_id": int, "claim": str, "importance": "Critical"|"Supporting"}

    Cached by (question, document) hash — same inputs never re-extracted.
    """
    cache_key = hashlib.sha256(f"{question}||{document}".encode()).hexdigest()
    if cache_key in _claim_cache:
        log.info("Step 1: Claims loaded from cache.")
        return _claim_cache[cache_key]

    log.info("Step 1: Extracting claims from document...")
    prompt = EXTRACTION_PROMPT.format(question=question, document=document)
    result = await call_openai_with_function(session, api_key, prompt, EXTRACTION_FUNCTION)
    claims = result["claims"]

    n_critical   = sum(1 for c in claims if c["importance"] == "Critical")
    n_supporting = len(claims) - n_critical
    log.info(
        f"  Extracted {len(claims)} claims "
        f"({n_critical} critical, {n_supporting} supporting)"
    )

    _claim_cache[cache_key] = claims
    return claims

# ---------------------------------------------------------------------------
# Step 2a: Evaluate a single batch  (with retry + exponential backoff)
# ---------------------------------------------------------------------------

async def evaluate_batch_with_retry(
    session: aiohttp.ClientSession,
    api_key: str,
    semaphore: asyncio.Semaphore,
    question: str,
    model_response: str,
    batch: list[dict],
    max_retries: int = MAX_RETRIES,
) -> list[dict]:
    """
    Evaluates a batch of claims in one API call.
    Retries up to `max_retries` times with exponential backoff (1s, 2s, 4s…).
    On permanent failure injects placeholder "Not Covered" entries (conservative).
    """
    ids = [c["claim_id"] for c in batch]
    claims_json = json.dumps(
        [{"claim_id": c["claim_id"], "claim": c["claim"]} for c in batch],
        indent=2,
    )
    prompt = BATCH_COVERAGE_PROMPT.format(
        question=question,
        model_response=model_response,
        claims_json=claims_json,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            async with semaphore:
                data = await call_openai_with_function(
                    session, api_key, prompt, COVERAGE_FUNCTION
                )
            return data["results"]
        except Exception as exc:
            last_error = exc
            wait = 2 ** attempt   # 1 s → 2 s → 4 s
            log.warning(
                f"  Batch {ids} failed (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {wait}s: {exc}"
            )
            await asyncio.sleep(wait)

    # All retries exhausted — return conservative placeholder results
    log.error(f"  Batch {ids} permanently failed: {last_error}")
    return [
        {
            "claim_id":       c["claim_id"],
            "coverage_score": 0,
            "coverage_label": "Not Covered",
            "reason":         f"Evaluation failed after {max_retries} retries: {last_error}",
        }
        for c in batch
    ]

# ---------------------------------------------------------------------------
# Step 2b: Evaluate all claims in parallel batches
# ---------------------------------------------------------------------------

async def evaluate_all_claims(
    session: aiohttp.ClientSession,
    api_key: str,
    question: str,
    model_response: str,
    claims: list[dict],
    batch_size: int = BATCH_SIZE,
    max_concurrent: int = MAX_CONCURRENT,
) -> list[dict]:
    """
    Splits claims into batches, fires all batches concurrently (capped at
    `max_concurrent`), and returns a flat list of coverage result dicts.
    """
    batches = [claims[i : i + batch_size] for i in range(0, len(claims), batch_size)]
    log.info(
        f"Step 2: Evaluating {len(claims)} claims across "
        f"{len(batches)} batch(es) (batch_size={batch_size}, "
        f"max_concurrent={max_concurrent})..."
    )

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        evaluate_batch_with_retry(
            session, api_key, semaphore, question, model_response, batch
        )
        for batch in batches
    ]
    batch_results = await asyncio.gather(*tasks)

    # Flatten list-of-lists
    coverage_results = [item for batch in batch_results for item in batch]
    log.info(f"  Evaluation complete for {len(coverage_results)} claims.")
    return coverage_results

# ---------------------------------------------------------------------------
# Step 3: Compute score locally  (zero LLM calls — pure math)
# ---------------------------------------------------------------------------

def compute_completeness_score(
    claims: list[dict],
    coverage_results: list[dict],
) -> dict:
    """
    Weighted aggregation — no LLM required.

    Weighted Score = Σ(coverage_score × weight) / Σ(2 × weight)
      Critical weight = 1.0
      Supporting weight = 0.5

    Returns a plain result dict.
    """
    importance_map = {c["claim_id"]: c["importance"] for c in claims}
    claim_text_map = {c["claim_id"]: c["claim"]       for c in claims}
    weight_map     = {"Critical": 1.0, "Supporting": 0.5}

    numerator   = 0.0
    denominator = 0.0
    for r in coverage_results:
        weight      = weight_map.get(importance_map.get(r["claim_id"], "Supporting"), 0.5)
        numerator  += r["coverage_score"] * weight
        denominator += 2 * weight

    score = numerator / denominator if denominator > 0 else 0.0

    label = (
        "Complete"           if score >= 0.85 else
        "Mostly Complete"    if score >= 0.60 else
        "Partially Complete" if score >= 0.35 else
        "Incomplete"
    )

    n_critical   = sum(1 for c in claims if importance_map.get(c["claim_id"]) == "Critical")
    n_supporting = len(claims) - n_critical

    missing_critical = [
        claim_text_map[r["claim_id"]]
        for r in coverage_results
        if importance_map.get(r["claim_id"]) == "Critical" and r["coverage_score"] == 0
    ]
    missing_supporting = [
        claim_text_map[r["claim_id"]]
        for r in coverage_results
        if importance_map.get(r["claim_id"]) == "Supporting" and r["coverage_score"] == 0
    ]

    return {
        "weighted_completeness_score": round(score, 4),
        "completeness_label":          label,
        "total_claims":                len(claims),
        "critical_claims":             n_critical,
        "supporting_claims":           n_supporting,
        "missing_critical_claims":     missing_critical,
        "missing_supporting_claims":   missing_supporting,
        "coverage_details":            coverage_results,
    }

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def evaluate_completeness(
    question: str,
    model_response: str,
    document: str,
    api_key: str | None = None,
    batch_size: int = BATCH_SIZE,
    max_concurrent: int = MAX_CONCURRENT,
) -> dict:
    """
    Full optimised completeness pipeline using OpenAI via raw HTTP POST.
    Returns a plain result dict.

    Args:
        question:       The original user question.
        model_response: The generated answer to evaluate.
        document:       The source document the answer should be based on.
        api_key:        OpenAI API key (falls back to OPENAI_API_KEY env var).
        batch_size:     How many claims to evaluate per API call.
        max_concurrent: Max parallel API calls in flight at once.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required — set OPENAI_API_KEY or pass api_key=...")

    # One shared aiohttp session for the lifetime of this pipeline call
    async with aiohttp.ClientSession() as session:
        claims = await extract_claims(session, api_key, question, document)
        if not claims:
            raise ValueError("No claims extracted — check your document and question.")

        coverage_results = await evaluate_all_claims(
            session, api_key, question, model_response, claims, batch_size, max_concurrent
        )

    result = compute_completeness_score(claims, coverage_results)
    return result

# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(result: dict) -> None:
    icons = {2: "✅", 1: "🟡", 0: "❌"}
    print("\n" + "=" * 65)
    print("         COMPLETENESS EVALUATION REPORT  (OpenAI)")
    print("=" * 65)
    print(
        f"  Overall Score : {result['weighted_completeness_score']:.2f}"
        f"  →  {result['completeness_label']}"
    )
    print(
        f"  Total Claims  : {result['total_claims']}  "
        f"(Critical: {result['critical_claims']}, "
        f"Supporting: {result['supporting_claims']})"
    )

    if result["missing_critical_claims"]:
        print(f"\n  ❌ Missing Critical ({len(result['missing_critical_claims'])}):")
        for c in result["missing_critical_claims"]:
            print(f"     • {c}")

    if result["missing_supporting_claims"]:
        print(f"\n  ⚠️  Missing Supporting ({len(result['missing_supporting_claims'])}):")
        for c in result["missing_supporting_claims"]:
            print(f"     • {c}")

    print("\n  Per-Claim Breakdown:")
    print(f"  {'ID':<4} {'Sc':<4} {'Label':<22} Reason")
    print("  " + "-" * 75)
    for r in sorted(result["coverage_details"], key=lambda x: x["claim_id"]):
        icon = icons.get(r["coverage_score"], "?")
        print(
            f"  {r['claim_id']:<4} {icon}{r['coverage_score']:<3}  "
            f"{r['coverage_label']:<22} {r['reason']}"
        )
    print("=" * 65 + "\n")

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    QUESTION = "What are the main causes of climate change and the recommended mitigation strategies?"

    DOCUMENT = """
    Climate change refers to long-term shifts in global temperatures and weather patterns.
    While some climate change is natural, since the 1800s, human activities have been the
    main driver. Burning fossil fuels like coal, oil, and gas generates greenhouse gas
    emissions that act like a blanket around the Earth, trapping heat and raising temperatures.

    The primary causes include:
    1. Burning fossil fuels for energy (electricity, heat, transport) — the largest source
    2. Deforestation — trees absorb CO2; cutting them releases stored carbon
    3. Industrial processes — cement and steel production release significant CO2
    4. Agriculture — livestock produce methane; rice paddies and fertilizers release N2O

    Key mitigation strategies recommended by the IPCC include:
    - Transitioning to renewable energy sources (solar, wind, hydro)
    - Improving energy efficiency in buildings and industry
    - Electrifying transportation and shifting to low-carbon fuels
    - Protecting and restoring forests and ecosystems
    - Carbon capture and storage (CCS) technologies
    - Reducing methane emissions from agriculture and waste
    """

    MODEL_RESPONSE = """
    Climate change is primarily driven by human activities, especially burning fossil fuels
    such as coal, oil, and natural gas, which release greenhouse gases into the atmosphere.
    Deforestation also plays a significant role as forests that once absorbed CO2 are removed.

    To address climate change, experts recommend switching to renewable energy sources like
    solar and wind power, and improving energy efficiency across sectors. Protecting forests
    is also considered an important strategy.
    """

    async def main():
        result = await evaluate_completeness(
            question=QUESTION,
            model_response=MODEL_RESPONSE,
            document=DOCUMENT,
            # api_key="sk-..."   ← or set OPENAI_API_KEY env var
        )
        print_report(result)

        # Save full results
        with open("completeness_result_openai.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Saved to completeness_result_openai.json")

    asyncio.run(main())
