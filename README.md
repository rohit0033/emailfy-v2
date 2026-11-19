# Emalify v2: LLM support-email classifier with guardrails

Classifies customer support emails for a (fictional) Indian logistics company, *Kirana Logistics*, and decides which ones can be **handled automatically** and which must go to a **human**.

A rewrite of my earlier Emalify project, **without LangChain**: direct API calls, strict output validation, and safety rules enforced in code.

## Results

Synthetic dataset of 50 emails (Hinglish, typos, multi-intent, legal notices, prompt injection), stratified 50/50 dev/test split.

| Metric | Dev (25) | Test (25) |
|---|---|---|
| Accuracy | 100% | 100% |
| **Automation rate** (no human needed) | 44% | 48% |
| **Automated but wrong** (reaches a customer unchecked) | **0** | **0** |
| Latency p50 / p95 | 661 / 1071 ms | 599 / 1225 ms |

Model: `openai/gpt-oss-20b` on Groq. A prompt fix raised automation from **27% → 43%** on the original 30 emails with zero automated-wrong answers.

> ⚠️ **Limitation:** I wrote these emails and their labels myself, so the numbers show the model agrees with *my* labels. An honest production number needs real customer emails labeled by the customer's support leads, run once on a locked test set.

## Design: the model proposes, code verifies

```
email ──► LLM (JSON mode) ──► pydantic validation ──► safety rules ──► route
                                 │ invalid                │
                                 ▼                        ▼
                        retry once WITH the error   refund/other → always human
                                 │ still invalid    (model can raise needs_human,
                                 ▼                   never lower it)
                        fallback: other + human
```

- **Strict schema** (`models.py`): categories map to downstream actions, with an `other` escape hatch and a `reason` field for debugging misroutes.
- **Prompt injection defense in code.** Delimiting the email in the prompt helps, but the real defense is `ALWAYS_HUMAN`: in one run the model *was* fooled into answering "refund," and the code rule still sent it to a human.
- **Two kinds of retry:** a bad model answer → retry with the validation error; an API 429/5xx → exponential backoff + jitter + `Retry-After`; a 401/404 → fail fast.
- **Honest reporting:** the report separates *system health* (API unavailable) from *AI quality* (accuracy only on emails the model answered) and *business value* (automation vs automated-but-wrong).
- **Observability:** every LLM call is logged to `logs/calls.jsonl` (tokens, latency, outcome). The logs are gitignored because they contain customer data.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # set LLM_PROVIDER=fake to run with no API key

python split.py                 # stratified dev/test split (fixed seed)
python classify.py              # dev set (tune here)
python classify.py --set test   # locked test set: run once per finished change
python classify.py 1 18         # debug specific emails
```

Providers: `fake` (canned responses, including deliberately broken ones), `gemini`, `groq`. Both real providers go through their OpenAI-compatible endpoints, so switching is one line in `.env`.

## Files
| File | What |
|---|---|
| `models.py` | Pydantic output contract |
| `classify.py` | Prompt, LLM call, validation, safety rules, retries, logging, report |
| `split.py` | Stratified dev/test split |
| `emails.json` | 50 synthetic emails with `expected_category` + `split` |
