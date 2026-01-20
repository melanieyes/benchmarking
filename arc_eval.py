import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

# -------------------------
# Client / Data
# -------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

arc = load_dataset("allenai/ai2_arc", "ARC-Challenge")
val = arc["validation"]   # 299
test = arc["test"]        # 2590
train = arc["train"]

import string
LETTERS = list(string.ascii_uppercase)
LETTER_SET = set(LETTERS)

# -------------------------
# Prompt formatting
# -------------------------
def format_arc_prompt(ex) -> tuple[str, list[str]]:
    """
    Returns:
      prompt_str: question + choices
      choice_letters: letters we used (A, B, C, ...) aligned to printed choices
    """
    question = ex["question"].strip()
    texts = ex["choices"]["text"]
    n = len(texts)
    choice_letters = LETTERS[:n]

    lines = [question]
    for L, t in zip(choice_letters, texts):
        lines.append(f"{L}. {str(t).strip()}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines), choice_letters

# -------------------------
# Few-shot builder (deterministic)
# -------------------------
def make_fewshot_prefix(train_split, k: int, seed: int = 42) -> str:
    if k <= 0:
        return ""
    rng = random.Random(seed)
    idxs = rng.sample(range(len(train_split)), k)
    blocks = []
    for i in idxs:
        ex = train_split[i]
        prompt, _ = format_arc_prompt(ex)
        gold = ex["answerKey"].strip().upper()
        blocks.append(prompt + " " + gold)
    return "\n\n".join(blocks) + "\n\n"

# -------------------------
# Logprobs helpers
# -------------------------
def _pick_letter_logprobs(top_logprobs):
    """
    top_logprobs: list of objects with .token and .logprob
    Returns dict like {'A': -0.1, 'B': -2.3, ...} for any letters found
    """
    out = {}
    for item in top_logprobs:
        tok_stripped = str(item.token).strip()
        if tok_stripped in LETTER_SET:
            out[tok_stripped] = float(item.logprob)
    return out

def call_with_logprobs(model: str, full_prompt: str, top_k: int = 5):
    """
    Returns:
      raw_text (str),
      letter_logprobs (dict),
      first_token_top_logprobs (list)
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0,
        max_completion_tokens=1,
        logprobs=True,
        top_logprobs=top_k,
    )

    msg = resp.choices[0].message.content
    top = resp.choices[0].logprobs.content[0].top_logprobs
    letter_lps = _pick_letter_logprobs(top)
    return msg, letter_lps, top

STRICT_SUFFIX = (
    "\n\nYou MUST answer with exactly one capital letter from the listed options. "
    "Do not write words, punctuation, or explanation.\nAnswer:"
)

def call_with_logprobs_robust(model: str, full_prompt: str, needed_letters: list[str], top_k: int = 5, max_retries: int = 2):
    """
    Retries until the needed letters appear in top_logprobs (as much as possible).
    Note: top_logprobs is capped at 5, so do NOT set needed_letters > top_k.
    """
    prompt = full_prompt
    last = None

    for attempt in range(max_retries + 1):
        msg, lps, top = call_with_logprobs(model, prompt, top_k=top_k)
        last = (msg, lps, top)

        if all(L in lps for L in needed_letters):
            return msg, lps, top

        # Retry with stricter suffix
        prompt = full_prompt + STRICT_SUFFIX

    # Return best-effort last call
    return last

def normalize_over_choices(letter_lps: dict, choice_letters: list[str]) -> dict:
    """
    Normalizes over the *choices* for this question.
    Missing letters get ~0 mass. This is still an approximation when top_logprobs
    doesn't include all choices, but robust retry helps a lot for A-D.
    """
    lps = np.array([letter_lps.get(L, -100.0) for L in choice_letters], dtype=float)
    m = lps.max()
    probs = np.exp(lps - m)
    probs = probs / probs.sum()
    return {L: float(p) for L, p in zip(choice_letters, probs)}


def eval_split(
    dataset_split,
    model: str,
    n: int | None = None,
    fewshot_k: int = 0,
    seed: int = 42,
    cache_path: str | None = None,
):
    # Optional: load cached results if present
    if cache_path is not None and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        summary = {
            "model": model,
            "shots": fewshot_k,
            "n": len(df),
            "accuracy": float(df["correct"].mean()),
            "mean_p_gold": float(df["p_gold"].mean()),
            "mean_p_pred": float(df["p_pred"].mean()),
        }
        return df, summary

    fewshot_prefix = make_fewshot_prefix(train, fewshot_k, seed=seed)

    rows = []
    it = range(len(dataset_split)) if n is None else range(min(n, len(dataset_split)))

    for i in tqdm(it, desc=f"Eval {model} | shots={fewshot_k}"):
        ex = dataset_split[i]
        base_prompt, choice_letters = format_arc_prompt(ex)
        full_prompt = fewshot_prefix + base_prompt

        # We can only demand up to top_k letters. For ARC, prioritize A-D if present.
        k_need = min(4, len(choice_letters), 5)
        needed_letters = choice_letters[:k_need]

        raw, letter_lps, _ = call_with_logprobs_robust(
            model=model,
            full_prompt=full_prompt,
            needed_letters=needed_letters,
            top_k=5,
            max_retries=2,
        )

        probs = normalize_over_choices(letter_lps, choice_letters)

        gold = ex["answerKey"].strip().upper()
        pred = max(probs.items(), key=lambda kv: kv[1])[0]
        p_gold = probs.get(gold, 0.0)

        raw_clean = (raw or "").strip()
        raw_valid = int(raw_clean in choice_letters)

        rows.append({
            "idx": i,
            "model": model,
            "shots": fewshot_k,
            "gold": gold,
            "pred": pred,
            "correct": int(pred == gold),
            "p_gold": p_gold,
            "p_pred": probs[pred],
            "raw": raw_clean,
            "raw_valid": raw_valid,
            "question": ex["question"],
        })

    df = pd.DataFrame(rows)
    summary = {
        "model": model,
        "shots": fewshot_k,
        "n": len(df),
        "accuracy": float(df["correct"].mean()),
        "mean_p_gold": float(df["p_gold"].mean()),
        "mean_p_pred": float(df["p_pred"].mean()),
        "raw_valid_rate": float(df["raw_valid"].mean()),
    }

    if cache_path is not None:
        df.to_csv(cache_path, index=False)

    return df, summary


def paired_analysis(df_a: pd.DataFrame, df_b: pd.DataFrame):
    needed_cols = {"idx", "correct", "pred", "gold", "question"}
    missing_a = needed_cols - set(df_a.columns)
    missing_b = needed_cols - set(df_b.columns)
    if missing_a:
        raise ValueError(f"df_a missing columns: {missing_a}")
    if missing_b:
        raise ValueError(f"df_b missing columns: {missing_b}")

    merged = df_a.merge(df_b, on="idx", suffixes=("_a", "_b"))

    only_a = merged[(merged["correct_a"] == 1) & (merged["correct_b"] == 0)]
    only_b = merged[(merged["correct_a"] == 0) & (merged["correct_b"] == 1)]
    both_wrong = merged[(merged["correct_a"] == 0) & (merged["correct_b"] == 0)]
    both_right = merged[(merged["correct_a"] == 1) & (merged["correct_b"] == 1)]

    return only_a, only_b, both_wrong, both_right


def main():
    MODELS = ["gpt-4.1-nano", "gpt-4o-mini"]

    dfs = []
    summaries = []

    # Validation 0-shot
    for m in MODELS:
        df_m, summ_m = eval_split(
            val, m, n=None, fewshot_k=0, seed=42,
            cache_path=f"val_{m}_shots0.csv",
        )
        dfs.append(df_m)
        summaries.append(summ_m)

    print(pd.DataFrame(summaries))

    only_nano, only_mini, both_wrong, both_right = paired_analysis(dfs[0], dfs[1])
    print("Only nano correct:", len(only_nano))
    print("Only mini correct:", len(only_mini))
    print("Both wrong:", len(both_wrong))
    print("Both right:", len(both_right))

    # Few-shot sweep on validation
    shots_grid = [0, 1, 2, 4, 8]
    all_summaries = []

    for m in MODELS:
        for k in shots_grid:
            df_mk, summ_mk = eval_split(
                val, m, n=None, fewshot_k=k, seed=42,
                cache_path=f"val_{m}_shots{k}.csv",
            )
            all_summaries.append(summ_mk)

    summ_df = pd.DataFrame(all_summaries)
    print(summ_df)

    pivot = summ_df.pivot(index="shots", columns="model", values="mean_p_gold")
    print(pivot)

    print(summ_df[["model", "shots", "accuracy", "mean_p_gold", "raw_valid_rate"]].sort_values(["shots", "model"]))

    # Full test run (costly) — cached
    df_nano, _ = eval_split(test, "gpt-4.1-nano", n=None, fewshot_k=0, seed=42, cache_path="test_gpt-4.1-nano_shots0.csv")
    df_mini, _ = eval_split(test, "gpt-4o-mini", n=None, fewshot_k=0, seed=42, cache_path="test_gpt-4o-mini_shots0.csv")

    # also export the filenames your writeup might reference
    df_nano.to_csv("arc_test_gpt-4.1-nano_shots0.csv", index=False)
    df_mini.to_csv("arc_test_gpt-4o-mini_shots0.csv", index=False)

if __name__ == "__main__":
    main()
