"""Stage 3 — text normalisation and redaction.

Two jobs, in this order.

**Redaction, for ethics and for generalisation.** 105,971 commit messages
(18.5%) contain an email address and 69,201 contain a URL. Email addresses and
author names are personal data under UK GDPR, so they must not be carried into
a published feature set or quoted in the dissertation. They are also harmful
to the model: a specific address is a near-unique identifier that invites
memorisation of individual contributors rather than learning of change risk.
Replacing each span with a type marker keeps the *signal* ("this message
mentions an email") and discards the identity.

**Normalisation, for tokeniser stability.** 20,149 messages contain non-ASCII
characters and 10 contain C0 control characters. Bot-generated changelogs
reach tens of thousands of characters. Unicode is normalised to NFC rather
than stripped, because CJK and accented identifiers are legitimate content;
control characters are removed; and length is capped so a handful of records
cannot dominate tokenisation cost.

Order is deliberate and was corrected after an audit of the output: Unicode is
canonicalised *first*, then spans are redacted, then the text is truncated.

Normalising first matters because NFC folds fullwidth and compatibility
characters onto their ASCII equivalents, so `alice＠example.com` only becomes
matchable as an email after normalisation. Redacting before normalising left 7
addresses and 15 URLs in the processed corpus. Redacting before truncating
matters because otherwise an address near the end of a long message would be
removed from short records and silently retained in long ones.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

from .config import (
    MAX_BRANCH_CHARS,
    MAX_COMMIT_MESSAGE_CHARS,
    MAX_DESCRIPTION_CHARS,
    REDACTION_TOKENS,
)

# Ordered most specific first: a GitHub token inside a URL should be caught as
# a secret, and an email inside a URL should not be split across two rules.
SECRET_RE = re.compile(
    r"(gh[pousr]_[A-Za-z0-9]{16,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"
)
# The final segment must be an alphabetic TLD. Without that constraint the
# pattern also matches dependency and version references that are common in
# this corpus — `actions/checkout@v3...v4`, `calcite-components@1.10.0`,
# `python@3.11` — and redacting those would destroy exactly the
# dependency-bump signal that makes `renovate/` and `dependabot/` branches
# predictive. The trade-off is that addresses with a non-alphabetic host
# (`bob@192.168.0.1`) are not matched; they do not occur in this dataset and
# are not personal data in the sense the redaction exists to protect.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://\S*|www\.\S+")
# Bare 7-40 hex characters: commit SHAs, which are unique per commit and so
# pure memorisation fodder.
SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
MENTION_RE = re.compile(r"(?<![\w/])@[A-Za-z0-9][A-Za-z0-9-]{0,38}\b")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"[ \t]+")
BLANKLINES_RE = re.compile(r"\n{3,}")


def redact(text: str) -> str:
    """Replace identifying and memorisable spans with type markers.

    Applied before any feature is computed, so no downstream stage can see the
    raw identifiers. The markers are retained rather than deleted because
    "contains a URL" and "contains a commit SHA" are themselves plausible
    signals, and keeping them makes token-level explanations readable.
    """
    text = SECRET_RE.sub(REDACTION_TOKENS["secret"], text)
    text = EMAIL_RE.sub(REDACTION_TOKENS["email"], text)
    text = URL_RE.sub(REDACTION_TOKENS["url"], text)
    text = MENTION_RE.sub(REDACTION_TOKENS["mention"], text)
    text = SHA_RE.sub(REDACTION_TOKENS["sha"], text)
    return text


def redact_identifiers(text: str) -> str:
    """Restricted redaction for identifier columns (branch and workflow names).

    Only addresses and credentials are removed. URLs, mentions, and SHAs are
    left alone, because a full redaction pass would rewrite refs such as
    `renovate/lodash-4.x` and destroy the single strongest static signal in
    the dataset (measured 27.4% failure rate on `renovate/` branches versus
    12.9% on `main`).

    This pass exists because an output audit found a contributor's real email
    address used as a branch name in 9 runs. Identifier columns are not exempt
    from privacy obligations just because they are usually machine-generated.
    The pattern over-matches in rare cases — one workflow named
    `prisma-schema-wasm@5.6.0-31.integration` is redacted spuriously — which is
    the right direction to err for 1 row in 567,814.
    """
    text = SECRET_RE.sub(REDACTION_TOKENS["secret"], text)
    return EMAIL_RE.sub(REDACTION_TOKENS["email"], text)


def canonicalise(text: str) -> str:
    """Fold Unicode to a canonical form and strip control characters.

    Runs before redaction. NFKC rather than NFC is used deliberately: it maps
    compatibility characters such as the fullwidth `＠` and `．` onto their
    ASCII equivalents, so obfuscated addresses become matchable by the
    redaction patterns instead of slipping through. Legitimate non-ASCII
    content (CJK text, accented identifiers) is preserved either way.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return CONTROL_RE.sub(" ", text)


def tidy(text: str, max_chars: int) -> str:
    """Collapse whitespace and cap length. Runs last, after redaction.

    Newline structure is preserved up to two consecutive breaks, because the
    blank line separating a commit subject from its body is a real convention
    and a feature worth keeping.
    """
    if not text:
        return ""
    text = WHITESPACE_RE.sub(" ", text)
    text = BLANKLINES_RE.sub("\n\n", text)
    text = text.strip()
    if len(text) > max_chars:
        # Mark the cut so that a truncated message is distinguishable from one
        # that genuinely ended there.
        text = text[:max_chars].rstrip() + " <TRUNC>"
    return text


def normalise(text: str, max_chars: int, mode: str = "prose") -> str:
    """Full text pipeline: canonicalise, redact, then tidy and truncate.

    `mode` selects the redaction strength. "prose" applies the full pass and
    is used for commit messages, run titles, and repository descriptions.
    "identifier" applies addresses and credentials only, preserving ref and
    workflow naming conventions.
    """
    cleaned = canonicalise(text)
    if mode == "prose":
        cleaned = redact(cleaned)
    elif mode == "identifier":
        cleaned = redact_identifiers(cleaned)
    else:
        raise ValueError(f"unknown redaction mode: {mode!r}")
    return tidy(cleaned, max_chars)


def pseudonymise(value: str | None, salt: str) -> str:
    """Stable, non-reversible identifier for a person or account.

    Keeps per-actor history features computable (the same actor maps to the
    same token) while removing the identity. The salt must be held outside the
    repository; without it the mapping cannot be reproduced, which is the
    point.
    """
    if not value:
        return ""
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return digest[:16]


def clean_text_columns(runs: pd.DataFrame, salt: str) -> pd.DataFrame:
    """Redact, normalise, and pseudonymise every text column in place.

    Raw columns are replaced rather than kept alongside, so that no later
    stage can reach the unredacted value. The one derived quantity taken from
    the raw text before redaction is its original length, because truncation
    and redaction both change it and "this was a 40,000-character changelog"
    is a legitimate signal.
    """
    runs = runs.copy()

    runs["raw_commit_message_chars"] = (
        runs["commit_message"].fillna("").astype(str).str.len().astype("int32")
    )

    text_limits = {
        "commit_message": MAX_COMMIT_MESSAGE_CHARS,
        "display_title": MAX_COMMIT_MESSAGE_CHARS,
        "head_branch": MAX_BRANCH_CHARS,
        "workflow_name": MAX_BRANCH_CHARS,
        "repo_description": MAX_DESCRIPTION_CHARS,
    }
    prose_columns = ("commit_message", "display_title", "repo_description")
    for column, limit in text_limits.items():
        mode = "prose" if column in prose_columns else "identifier"
        values = runs[column].fillna("").astype(str)
        runs[column] = [normalise(value, limit, mode) for value in values]

    # Identity columns: replaced by salted hashes, never kept in the clear.
    for column in ("actor_login", "triggering_actor_login"):
        # Bot detection must happen before hashing, since it reads the suffix.
        runs[f"{column}_is_bot"] = (
            runs[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(r"\[bot\]$|^dependabot$|^renovate", regex=True)
            .astype("int8")
        )
        runs[column] = runs[column].map(lambda v: pseudonymise(v, salt))

    runs["commit_author_id"] = runs["commit_author_email"].map(
        lambda v: pseudonymise(v, salt)
    )
    runs = runs.drop(columns=["commit_author_email", "commit_author_name"])
    return runs


def build_transformer_input(runs: pd.DataFrame) -> pd.Series:
    """Assemble the single string fed to a pre-trained encoder.

    Field order and the explicit labels matter. Workflow identity is placed
    first because 47.7% of failures share a commit with a success, so the
    model cannot do better than chance on those cases unless it knows which
    workflow is running. The commit message goes last because it is the only
    field that can be long, so truncation at the tokeniser's maximum length
    removes the least important content rather than the most.
    """
    parts = [
        "workflow: " + runs["workflow_name"].fillna(""),
        " | event: " + runs["event"].fillna("unknown"),
        " | branch: " + runs["head_branch"].fillna(""),
        " | title: " + runs["display_title"].fillna(""),
        " | message: " + runs["commit_message"].fillna(""),
    ]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + part
    return joined
