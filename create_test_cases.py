"""
create_test_cases.py
Writes test_cases.xlsx for run_tests.py.

1. (Optional) Legacy banking fixtures — set INCLUDE_LEGACY_BANKING_FIXTURES.
2. All *.html / *.htm under HTML_DOCS_DIR — content via html_to_md_conversion().
   By default, questions are generated from each document with OpenAI
   (GENERATE_QUESTIONS_FROM_HTML_DOCUMENT). Set False to use a single preset
   question per file. Requires OPENAI_API_KEY when generation is on.

Run: python create_test_cases.py
Then: python run_tests.py
"""

import asyncio
import logging
import os
from pathlib import Path

import aiohttp
import openpyxl

from question_gen_openai import generate_questions_for_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML sources (edit HTML_DOCS_DIR and html_to_md_conversion)
# ---------------------------------------------------------------------------

# Folder containing your HTML documents (created next to this script by default).
HTML_DOCS_DIR = Path(__file__).resolve().parent / "html_docs"

# Set False if you only want rows from HTML_DOCS_DIR.
INCLUDE_LEGACY_BANKING_FIXTURES = True


def html_to_md_conversion(html_path: str) -> str:
    """
    Read one HTML file and return the document text used in the `document`
    column (Markdown or plain text). Implement your table-preserving
    conversion here.

    Args:
        html_path: Absolute or relative path to a .html file.

    Returns:
        String passed to test cases and later to run_tests.py / the evaluator.
    """
    # --- Replace this body with your HTML → Markdown pipeline ---
    with open(html_path, encoding="utf-8", errors="replace") as f:
        return f.read()


# Used for every HTML file unless overridden in HTML_QUESTIONS_BY_STEM.
DEFAULT_HTML_QUESTION = (
    "What are the key facts, figures, obligations, and any tabular or numeric "
    "details in this document that a reader must know to answer operational "
    "or compliance questions about it?"
)

# Optional: map file stem (filename without extension) → specific question.
# Example: {"annual_report": "What revenue figures appear in the tables? ..."}
HTML_QUESTIONS_BY_STEM: dict[str, str] = {}

# True: LLM generates QUESTIONS_PER_HTML_DOC questions per HTML file (needs OPENAI_API_KEY).
# False: one question per file from HTML_QUESTIONS_BY_STEM or DEFAULT_HTML_QUESTION (no API).
GENERATE_QUESTIONS_FROM_HTML_DOCUMENT = True
QUESTIONS_PER_HTML_DOC = 3


def _iter_html_files() -> list[Path]:
    if not HTML_DOCS_DIR.is_dir():
        return []
    paths = list(HTML_DOCS_DIR.glob("*.html")) + list(HTML_DOCS_DIR.glob("*.htm"))
    return sorted(paths, key=lambda p: p.name.lower())

# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

DOCUMENTS = {
"Personal Loan": """
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
""",

"KYC / AML Policy": """
KYC & AML POLICY — NORTHSTAR BANK

Individual onboarding documents:
- Identity: Aadhaar, PAN, Passport, Voter ID, or Driving Licence (any one).
- Address: Utility bill (≤3 months old), bank statement, or registered rent
  agreement (any one).
Non-individual entities must provide: Certificate of Incorporation, MoA/AoA,
board resolution, and beneficial owner declaration.

CDD Levels:
- Basic CDD: applied to standard retail customers.
- Enhanced Due Diligence (EDD): mandatory for PEPs and their immediate family,
  customers from FATF high-risk jurisdictions, accounts with monthly credits
  >₹5,00,000, and inconclusive video-KYC cases.

Verification: In-person required for accounts with expected monthly transactions
>₹10,00,000. Video KYC (V-CIP) permitted for resident individuals with Aadhaar
biometric consent. Re-KYC: every 2 years (high-risk), every 10 years (low-risk).

Transaction monitoring: Cash Transaction Reports (CTRs) filed for cash
transactions >₹10,00,000 in a single day. Structuring is auto-flagged.

STR filing: Within 7 working days of identifying a suspicious transaction.
Tipping off the customer is prohibited under PMLA 2002.

Record retention: KYC documents for 5 years after account closure.
Transaction records for 5 years from transaction date.
""",

"Credit Card T&C": """
CREDIT CARD TERMS — NORTHSTAR BANK PLATINUM

Annual fee: ₹2,999 + GST. Waived if annual spends exceed ₹1,50,000.

Interest: 3.49% per month (41.88% p.a.) on daily outstanding if full
balance not paid by due date. Calculated from transaction date. No
interest-free period on cash advances.

Minimum payment: 5% of outstanding or ₹200, whichever is higher.
Late fee: ₹0 (outstanding ≤₹500), ₹500 (₹501–₹5,000), ₹750
(₹5,001–₹10,000), ₹1,200 (₹10,001–₹25,000), ₹1,500 (above ₹25,000).

Cash advance: Limit is 30% of credit limit. Fee: 2.5% of amount (min ₹500).
Interest accrues from date of withdrawal.

Disputes: Must be reported within 60 days of statement date. Provisional
credit applied within 10 working days. Resolution in 30–45 days.

Rewards: 1 point per ₹100 spent (excludes fuel, EMI, cash advances,
government transactions). Points expire after 24 months. Redemption:
₹0.25 per point or 1 airline mile. 3x points on dining, travel, grocery.
""",

"Fixed Deposit": """
FIXED DEPOSIT POLICY — NORTHSTAR BANK

Tenure: 7 days to 10 years. Minimum: ₹10,000 (general); ₹5,000 (senior
citizens aged 60+). Senior citizens earn 0.50% p.a. additional interest.

Interest: Compounded quarterly for tenures ≥6 months; simple interest for
<6 months. Rate locked on booking date for the full tenure. Payout options:
monthly (discounted rate), quarterly, or at maturity.

Premature withdrawal (full): Allowed any time with 1% p.a. penalty on the
rate applicable for the actual holding period. No penalty if held <7 days.
Partial withdrawal: NOT permitted on standard FDs — full FD must be closed.
Flexi-FD (linked to savings account) allows partial withdrawal in ₹1,000
multiples with no penalty.

Auto-renewal: FD auto-renews at maturity rate unless customer gives written
instruction 3 working days before maturity. Unclaimed proceeds earn savings
account rate after 7 days.

Loan against FD: Up to 90% of FD value at FD rate + 1.5% p.a. No processing
fee. Loan tenure cannot exceed residual FD tenure.

TDS: 10% if annual interest >₹40,000 (₹50,000 for senior citizens). 20%
without PAN. Avoid TDS by submitting Form 15G/15H at the start of each year.
""",

"Home Loan": """
HOME LOAN AGREEMENT — NORTHSTAR BANK

LTV ratios (per RBI):
- Up to ₹30,00,000: maximum 90% LTV
- ₹30,00,001 to ₹75,00,000: maximum 80% LTV
- Above ₹75,00,000: maximum 75% LTV
LTV is on the lower of cost or market value by bank-approved valuer.

Insurance: Property insurance for full replacement value is mandatory throughout
the loan. Must cover fire, earthquake, flood, and allied perils. Bank named as
co-insured / loss payee. Failure to maintain = event of default.

Legal: Independent legal opinion from bank-empanelled advocate mandatory before
disbursement. MODT (Memorandum of Deposit of Title Deed) must be stamped per
state stamp duty on deposit of title documents.

Default and SARFAESI: NPA after 90 days overdue. Bank issues 60-day demand
notice under Section 13(2) SARFAESI Act 2002. If unpaid, bank may take
possession under Section 13(4) and sell property by public auction without
court intervention.

Prepayment: Floating rate — no charges (RBI directive). Fixed rate — 2% of
outstanding principal. Part-prepayment minimum = 3 EMIs equivalent.

CERSAI: Bank must register the equitable mortgage on CERSAI within 30 days of
mortgage creation.
""",

"Savings Account": """
SAVINGS ACCOUNT TERMS — NORTHSTAR BANK

Minimum balance (Monthly Average Balance):
- Regular: ₹5,000 (metro/urban), ₹2,000 (semi-urban), ₹1,000 (rural)
- Premium: ₹10,000 (all locations)
Non-maintenance fee: Regular ₹150/₹100/₹50 per month by location.
Premium: ₹300/month.

ATM usage:
- Own-bank ATMs: 5 free cash withdrawals/month; ₹20 per extra transaction.
- Other-bank ATMs: 3 free (metro) / 5 free (non-metro) per month.
  Charges after free limit: ₹21 (financial), ₹8.50 (non-financial).

Account closure: Free after 12 months. Within 12 months: ₹500 (Regular),
₹1,000 (Premium). Must surrender debit card and unused cheques.

Dormant: Inoperative after 24 months of no customer-initiated transaction.
Unclaimed balances transferred to RBI DEA Fund after 10 years. Reactivation
requires branch visit with KYC documents or video-KYC (online).

Online transfers: NEFT/RTGS/IMPS free via net/mobile banking. Branch NEFT:
₹2–₹25 by amount. IMPS daily limit: ₹5,00,000. UPI: ₹1,00,000/transaction.
RTGS minimum: ₹2,00,000.

Auto-sweep: Balances above ₹25,000 (Regular) / ₹50,000 (Premium) swept to
linked FD in ₹5,000 multiples. Sweep-back on LIFO basis. No penalty on
sweep FDs.
""",
}

# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

QUESTIONS = {
"Personal Loan": (
    "What are the interest rate, processing fee, EMI calculation method "
    "and bounce charges, prepayment and foreclosure terms, conditions and "
    "consequences of default, and the insurance requirement?"
),
"KYC / AML Policy": (
    "What documents are required for individual and non-individual onboarding, "
    "what are the CDD levels and when does EDD apply, what are the verification "
    "methods, transaction monitoring thresholds, STR filing rules, and record "
    "retention period?"
),
"Credit Card T&C": (
    "What are the annual fee and waiver conditions, how is interest calculated "
    "on balances and cash advances, what are the minimum payment and late fee "
    "structure, cash advance limits and charges, dispute window, and rewards "
    "programme rules?"
),
"Fixed Deposit": (
    "What are the tenure options and minimum deposit amounts, how is interest "
    "calculated and what are the payout options, what penalty applies on "
    "premature withdrawal and is partial withdrawal allowed, what are the "
    "auto-renewal rules, loan-against-FD terms, and TDS rules?"
),
"Home Loan": (
    "What are the LTV ratios for different loan amounts, what insurance is "
    "mandatory and what must it cover, what legal steps are required before "
    "disbursement, when is an account NPA and what SARFAESI action follows, "
    "what are the prepayment terms for floating vs fixed rate, and when must "
    "CERSAI registration happen?"
),
"Savings Account": (
    "What are the minimum balance requirements and non-maintenance charges "
    "for different account types and locations, what are the ATM free transaction "
    "limits and charges, what are the account closure fees, how is a dormant "
    "account defined and reactivated, what are the online transfer limits, "
    "and how does auto-sweep work?"
),
}

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("complete",        "Complete",            "Hard"),
    ("mostly_complete", "Mostly Complete",     "Medium"),
    ("partial",         "Partially Complete",  "Easy"),
    ("incomplete",      "Incomplete",          "Easy"),
]

# ---------------------------------------------------------------------------
# Build rows and write Excel
# ---------------------------------------------------------------------------

HEADERS = ["test_id", "document_type", "difficulty", "scenario_type",
           "question", "document", "expected_label"]

def _append_rows_for_document(
    rows: list,
    counter: int,
    doc_type: str,
    question: str,
    document_text: str,
) -> int:
    for scenario_type, expected_label, difficulty in SCENARIOS:
        rows.append([
            f"TC_{counter:03d}",
            doc_type,
            difficulty,
            scenario_type,
            question,
            document_text.strip(),
            expected_label,
        ])
        counter += 1
    return counter


async def _add_rows_for_html_files_llm(
    rows: list,
    counter: int,
    html_files: list[Path],
    api_key: str,
) -> int:
    async with aiohttp.ClientSession() as session:
        for html_path in html_files:
            stem = html_path.stem
            document_text = html_to_md_conversion(str(html_path.resolve()))
            if stem in HTML_QUESTIONS_BY_STEM:
                questions = [HTML_QUESTIONS_BY_STEM[stem]]
            else:
                questions = await generate_questions_for_document(
                    session, api_key, document_text, QUESTIONS_PER_HTML_DOC
                )
                if not questions:
                    log.warning(
                        "No LLM questions for %s; using DEFAULT_HTML_QUESTION.", stem
                    )
                    questions = [DEFAULT_HTML_QUESTION]
            for question in questions:
                counter = _append_rows_for_document(
                    rows, counter, stem, question, document_text
                )
    return counter


def main():
    rows: list[list] = []
    counter = 1

    if INCLUDE_LEGACY_BANKING_FIXTURES:
        for doc_type in DOCUMENTS:
            counter = _append_rows_for_document(
                rows,
                counter,
                doc_type,
                QUESTIONS[doc_type],
                DOCUMENTS[doc_type],
            )

    html_files = _iter_html_files()
    if html_files:
        if GENERATE_QUESTIONS_FROM_HTML_DOCUMENT:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise SystemExit(
                    "OPENAI_API_KEY is required when HTML files are present and "
                    "GENERATE_QUESTIONS_FROM_HTML_DOCUMENT is True. "
                    "Set the env var, or set GENERATE_QUESTIONS_FROM_HTML_DOCUMENT = False "
                    "to use DEFAULT_HTML_QUESTION / HTML_QUESTIONS_BY_STEM only."
                )
            counter = asyncio.run(
                _add_rows_for_html_files_llm(rows, counter, html_files, api_key)
            )
        else:
            for html_path in html_files:
                stem = html_path.stem
                question = HTML_QUESTIONS_BY_STEM.get(stem, DEFAULT_HTML_QUESTION)
                document_text = html_to_md_conversion(str(html_path.resolve()))
                counter = _append_rows_for_document(
                    rows, counter, stem, question, document_text
                )

    if not rows:
        raise SystemExit(
            "No test cases generated. Add .html files under HTML_DOCS_DIR, "
            "or set INCLUDE_LEGACY_BANKING_FIXTURES = True."
        )

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Test Cases"
    ws.append(HEADERS)
    for row in rows:
        ws.append(row)

    # Basic column widths
    for col, width in zip("ABCDEFG", [10, 20, 10, 18, 60, 80, 18]):
        ws.column_dimensions[col].width = width

    wb.save("test_cases.xlsx")
    n_html = len(html_files)
    gen = GENERATE_QUESTIONS_FROM_HTML_DOCUMENT and bool(html_files)
    print(
        f"Written {len(rows)} test cases → test_cases.xlsx "
        f"({n_html} HTML source(s), llm_questions={gen}, "
        f"legacy_banking={INCLUDE_LEGACY_BANKING_FIXTURES})"
    )

if __name__ == "__main__":
    main()
