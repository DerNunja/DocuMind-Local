from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CategoryStatus(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    ARCHIVED = "archived"


class DocumentStatus(StrEnum):
    RECEIVED = "received"
    EXTRACTION_QUALITY_FAILED = "extraction_quality_failed"
    PROFILE_READY = "profile_ready"
    CLASSIFICATION_SUGGESTED = "classification_suggested"
    NEEDS_REVIEW = "needs_review"
    NEEDS_TAXONOMY_REVIEW = "needs_taxonomy_review"
    APPROVED = "approved"
    CLASSIFICATION_ERROR = "classification_error"


class Category(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cat"))
    name: str
    description: str
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    near_misses: list[str] = Field(default_factory=list)
    example_document_ids: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    status: CategoryStatus = CategoryStatus.ACTIVE
    embedding: list[float] | None = None
    embedding_model: str | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    def embedding_text(self) -> str:
        parts = [
            f"Name: {self.name}",
            f"Description: {self.description}",
            "Inclusion criteria: " + "; ".join(self.inclusion_criteria),
            "Exclusion criteria: " + "; ".join(self.exclusion_criteria),
            "Near misses: " + "; ".join(self.near_misses),
        ]
        return "\n".join(part for part in parts if part.strip())


class ClassificationProfile(BaseModel):
    summary: str
    document_type: str
    business_purpose: str = ""
    key_entities: dict[str, list[str]] = Field(default_factory=dict)
    key_references: dict[str, list[str]] = Field(default_factory=dict)
    dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    suggested_tags: dict[str, list[str]] = Field(default_factory=dict)
    evidence_snippets: list[str] = Field(default_factory=list)

    @field_validator("dates", "amounts", "keywords", "evidence_snippets", mode="before")
    @classmethod
    def normalize_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            return [str(value)]
        normalized = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(", ".join(f"{key}: {val}" for key, val in item.items()))
            else:
                normalized.append(str(item))
        return normalized

    @field_validator("key_entities", "key_references", "suggested_tags", mode="before")
    @classmethod
    def normalize_string_list_dict(cls, value: Any) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {}
        normalized = {}
        for key, items in value.items():
            if items is None:
                normalized[str(key)] = []
            elif isinstance(items, list):
                normalized[str(key)] = [str(item) for item in items]
            else:
                normalized[str(key)] = [str(items)]
        return normalized

    def embedding_text(self) -> str:
        return self.model_dump_json(exclude_none=True)


class CandidateRanking(BaseModel):
    category_id: str
    category_name: str | None = None
    similarity: float | None = None
    fit: str | None = None
    reason: str | None = None


class NoneFitsProposal(BaseModel):
    name: str = ""
    description: str = ""
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)


class ClassificationDecision(BaseModel):
    decision: str
    selected_category_id: str | None = None
    ranking: list[CandidateRanking] = Field(default_factory=list)
    rationale: str = ""
    evidence_snippets: list[str] = Field(default_factory=list)
    proposed_tags: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    diagnostic_confidence: float | None = None
    none_fits_proposal: NoneFitsProposal | None = None

    @field_validator("none_fits_proposal", mode="before")
    @classmethod
    def normalize_none_fits_proposal(cls, value: Any) -> Any:
        if value is None or isinstance(value, dict):
            return value
        return {"name": str(value), "description": str(value)}


class DocumentRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("doc"))
    filename: str
    source_path: str | None = None
    extracted_text: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.RECEIVED
    profile: ClassificationProfile | None = None
    decision: ClassificationDecision | None = None
    primary_category_id: str | None = None
    tags: dict[str, list[str]] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
