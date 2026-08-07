"""Guardrails: PII detection/anonymization (Presidio) plus lightweight
input/output safety checks (prompt-injection heuristics, blocked terms).
"""

from __future__ import annotations

import logging
import os
import re

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.utils.schemas import PIIEntity

logger = logging.getLogger("guardrails")

PII_ENABLED = os.getenv("MASK_PII_ON_INGEST", "true").lower() == "true"

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|previous|the above) instructions", re.IGNORECASE),
    re.compile(r"disregard (all|previous|your) (system|instructions)", re.IGNORECASE),
    re.compile(r"you are now (in )?(dan|developer) mode", re.IGNORECASE),
    re.compile(r"reveal (your|the) (system prompt|instructions)", re.IGNORECASE),
]

DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "IP_ADDRESS",
]


class PIIGuard:
    def __init__(self, entities: list[str] | None = None):
        self.entities = entities or DEFAULT_ENTITIES
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def detect(self, text: str, language: str = "en") -> list[PIIEntity]:
        results = self._analyzer.analyze(text=text, entities=self.entities, language=language)
        return [
            PIIEntity(
                entity_type=r.entity_type,
                text=text[r.start:r.end],
                start=r.start,
                end=r.end,
                score=r.score,
            )
            for r in results
        ]

    def anonymize(self, text: str, language: str = "en") -> tuple[str, list[PIIEntity]]:
        if not PII_ENABLED:
            return text, []

        analyzer_results = self._analyzer.analyze(text=text, entities=self.entities, language=language)
        if not analyzer_results:
            return text, []

        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=analyzer_results)
        entities = [
            PIIEntity(entity_type=r.entity_type, text=text[r.start:r.end], start=r.start, end=r.end, score=r.score)
            for r in analyzer_results
        ]
        logger.info("anonymized %d PII entities", len(entities))
        return anonymized.text, entities


def detect_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def guard_input(text: str, pii_guard: PIIGuard) -> tuple[str, dict]:
    """Run all input-side guardrails. Returns (safe_text, report)."""

    report: dict = {"injection_detected": detect_prompt_injection(text)}
    safe_text, pii_entities = pii_guard.anonymize(text)
    report["pii_entities"] = [e.model_dump() for e in pii_entities]
    return safe_text, report
