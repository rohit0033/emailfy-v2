"""split.py — give every email a "split" field: "dev" or "test".

WHY (D1 lesson):
  - dev  = the "mock test". Tune the prompt on it as often as you like.
  - test = the "final exam". Locked. Run it ONCE after a change, never tune on it.
    The moment you tune on test, it secretly becomes dev and your numbers lie (overfitting).

HOW: a STRATIFIED split.
  We split INSIDE each category, so rare categories (e.g. only 7 spam emails)
  appear in BOTH dev and test. A plain random split could put all legal/spam
  emails in dev by bad luck, and then the test never checks them.

Run:  python split.py          (safe to re-run: same seed -> same split every time)
"""
import json
import random
from collections import defaultdict  # a dict that creates a default value for missing keys

EMAILS_PATH = "emails.json"
DEV_FRACTION = 0.5  # prompt-based system: no training set, so just dev + test (50/50)

# A fixed SEED makes "random" repeatable: the same shuffle every run.
# Without it, every run would move emails between dev and test -> results not comparable over time.
# (Like seeding a random generator in JS tests so failures are reproducible.)
SEED = 42


def main() -> None:
    with open(EMAILS_PATH, encoding="utf-8") as f:
        emails = json.load(f)

    # Group emails by category.
    # defaultdict(list): if a key doesn't exist yet, it starts as an empty list [].
    # JS equivalent:  const groups = {};  (groups[cat] ??= []).push(email)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for email in emails:
        by_category[email["expected_category"]].append(email)

    # random.Random(SEED) = our own random generator with a fixed seed,
    # so we don't affect (or get affected by) other random calls in the program.
    rng = random.Random(SEED)

    # Python's string formatting: {x:<16} = pad x to 16 characters, left-aligned (like padEnd(16)).
    print(f"{'category':<16} {'total':>5} {'dev':>5} {'test':>5}")

    # sorted(...) so the order is always the same (dict order could differ if the file changes order).
    for category in sorted(by_category):
        group = by_category[category]

        # Sort by id first, THEN shuffle, so the result doesn't depend on the order emails
        # happen to be in the file. Only the seed decides.
        # key=lambda e: e["id"]  -> sort by each email's id. (JS: group.sort((a, b) => a.id - b.id))
        group.sort(key=lambda e: e["id"])
        rng.shuffle(group)  # shuffles the list IN PLACE (changes it, returns nothing)

        # How many go to dev. round(7 * 0.5) = round(3.5) = 4 (Python rounds .5 to the nearest EVEN number,
        # so it can be 3 or 4; either is fine). max(1, ...) guarantees at least 1 in dev...
        n_dev = max(1, round(len(group) * DEV_FRACTION))
        # ...and min(..., len-1) guarantees at least 1 in TEST, even for tiny categories.
        n_dev = min(n_dev, len(group) - 1)

        # enumerate gives (index, item). The first n_dev shuffled emails -> dev, the rest -> test.
        for i, email in enumerate(group):
            email["split"] = "dev" if i < n_dev else "test"

        print(f"{category:<16} {len(group):>5} {n_dev:>5} {len(group) - n_dev:>5}")

    # Write back in the ORIGINAL id order, so the file stays easy to read.
    emails.sort(key=lambda e: e["id"])
    with open(EMAILS_PATH, "w", encoding="utf-8") as f:
        # ensure_ascii=False keeps Hindi/₹ characters readable instead of ₹ escapes.
        # indent=1 = pretty-print with 1 space (like JSON.stringify(x, null, 1)).
        json.dump(emails, f, ensure_ascii=False, indent=1)

    dev = sum(1 for e in emails if e["split"] == "dev")  # count of dev emails (JS: filter(...).length)
    print(f"\nTotal: {len(emails)}  ->  dev {dev}, test {len(emails) - dev}")
    print("Rule: tune on dev only. Run test ONCE, after a change is finished.")


if __name__ == "__main__":
    main()
