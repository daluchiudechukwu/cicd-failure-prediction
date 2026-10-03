"""Pre-execution CI/CD failure prediction from GHALogs metadata.

Pipeline stages, in order:

    config     thresholds and vocabularies, in one place
    ingest     streaming JSON-lines -> flattened Parquet
    quality    noise filtering with an auditable exclusion ledger
    textproc   PII redaction, Unicode normalisation, pseudonymisation
    features   static (history-free) feature engineering
    history    causally ordered execution-history features
    contract   executable enforcement of the pre-execution feature contract
    pipeline   orchestration and CLI
"""

__all__ = [
    "config",
    "contract",
    "features",
    "history",
    "ingest",
    "pipeline",
    "quality",
    "textproc",
]
