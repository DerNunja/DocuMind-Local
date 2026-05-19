from __future__ import annotations

import json
from pathlib import Path

import requests

from .lm_studio import LMStudioClient
from .models import (
    CandidateRanking,
    Category,
    CategoryStatus,
    ClassificationDecision,
    ClassificationProfile,
    DocumentRecord,
    DocumentStatus,
)
from .prompts import DECISION_SYSTEM_PROMPT, DECISION_USER_TEMPLATE, PROFILE_SYSTEM_PROMPT, PROFILE_USER_TEMPLATE
from .store import PostgresStore
from .vector import fallback_embedding


MIN_TEXT_CHARS = 80


class CategorisationService:
    def __init__(self, store: PostgresStore, client: LMStudioClient) -> None:
        self.store = store
        self.client = client

    def add_category(
        self,
        name: str,
        description: str,
        inclusion_criteria: list[str] | None = None,
        exclusion_criteria: list[str] | None = None,
        proposed: bool = False,
    ) -> Category:
        category = Category(
            name=name,
            description=description,
            inclusion_criteria=inclusion_criteria or [],
            exclusion_criteria=exclusion_criteria or [],
            status=CategoryStatus.PROPOSED if proposed else CategoryStatus.ACTIVE,
        )
        category.embedding = self._embed(category.embedding_text())
        category.embedding_model = self.client.embedding_model
        return self.store.add_category(category)

    def categorise_text(
        self,
        text: str,
        filename: str = "manual-input.txt",
        source_path: str | None = None,
    ) -> DocumentRecord:
        document = DocumentRecord(
            filename=filename,
            source_path=source_path,
            extracted_text=text,
        )
        if len(text.strip()) < MIN_TEXT_CHARS:
            document.status = DocumentStatus.EXTRACTION_QUALITY_FAILED
            return self.store.add_document(document)

        try:
            profile = self.build_profile(text, filename)
        except (ValueError, KeyError, requests.RequestException) as exc:
            document.status = DocumentStatus.CLASSIFICATION_ERROR
            document.errors.append(str(exc))
            return self.store.add_document(document)

        document.profile = profile
        document.status = DocumentStatus.PROFILE_READY

        candidates = self.retrieve_candidates(profile)
        try:
            decision = self.classify(profile, candidates)
        except (ValueError, KeyError, requests.RequestException) as exc:
            document.status = DocumentStatus.CLASSIFICATION_ERROR
            document.errors.append(str(exc))
            return self.store.add_document(document)

        document.decision = decision
        document.tags = decision.proposed_tags
        if decision.decision == "none_fits":
            document.status = DocumentStatus.NEEDS_TAXONOMY_REVIEW
        else:
            document.status = DocumentStatus.NEEDS_REVIEW
            document.primary_category_id = decision.selected_category_id
        return self.store.add_document(document)

    def categorise_file(self, path: Path) -> DocumentRecord:
        text = path.read_text(encoding="utf-8")
        return self.categorise_text(text=text, filename=path.name, source_path=str(path))

    def build_profile(self, text: str, filename: str) -> ClassificationProfile:
        system_prompt = PROFILE_SYSTEM_PROMPT.strip()
        user_prompt = render_profile_prompt(filename, text)
        return ClassificationProfile.model_validate(
            self.client.chat_json(system_prompt, user_prompt)
        )

    def retrieve_candidates(
        self, profile: ClassificationProfile, limit: int = 10
    ) -> list[CandidateRanking]:
        categories = [
            category
            for category in self.store.load_categories()
            if category.status == CategoryStatus.ACTIVE
        ]
        if not categories:
            return []
        profile_embedding = self._embed(profile.embedding_text())
        embeddings_changed = False
        for category in categories:
            if category.embedding_model != self.client.embedding_model:
                category.embedding = self._embed(category.embedding_text())
                category.embedding_model = self.client.embedding_model
                embeddings_changed = True
        if embeddings_changed:
            all_categories = self.store.load_categories()
            updated_categories = {category.id: category for category in categories}
            self.store.save_categories(
                [updated_categories.get(category.id, category) for category in all_categories]
            )
        return self.store.search_categories(
            profile_embedding,
            embedding_model=self.client.embedding_model,
            limit=limit,
        )

    def classify(
        self, profile: ClassificationProfile, candidates: list[CandidateRanking]
    ) -> ClassificationDecision:
        categories = {category.id: category for category in self.store.load_categories()}
        candidate_payload = []
        for candidate in candidates:
            category = categories[candidate.category_id]
            candidate_payload.append(
                {
                    "id": category.id,
                    "name": category.name,
                    "description": category.description,
                    "inclusion_criteria": category.inclusion_criteria,
                    "exclusion_criteria": category.exclusion_criteria,
                    "near_misses": category.near_misses,
                    "similarity": candidate.similarity,
                }
            )

        system_prompt = DECISION_SYSTEM_PROMPT.strip()
        user_prompt = render_decision_prompt(profile, candidate_payload)
        decision = ClassificationDecision.model_validate(
            self.client.chat_json(system_prompt, user_prompt)
        )
        if decision.decision == decision.selected_category_id:
            decision.decision = "category"
        if decision.decision == "category" and not decision.selected_category_id:
            raise ValueError("LLM selected category decision without selected_category_id")
        if decision.selected_category_id and decision.selected_category_id not in categories:
            raise ValueError("LLM selected a category that does not exist")
        return decision

    def _embed(self, text: str) -> list[float]:
        try:
            return self.client.embed(text)
        except requests.RequestException:
            return fallback_embedding(text)


def render_profile_prompt(filename: str, text: str) -> str:
    return (
        PROFILE_USER_TEMPLATE.replace("{{filename}}", filename)
        .replace("{{text[:12000]}}", text[:12000])
        .strip()
    )


def render_decision_prompt(
    profile: ClassificationProfile, candidate_payload: list[dict]
) -> str:
    return (
        DECISION_USER_TEMPLATE.replace(
            "{{profile.model_dump_json(indent=2)}}",
            profile.model_dump_json(indent=2),
        )
        .replace(
            "{{json.dumps(candidate_payload, indent=2)}}",
            json.dumps(candidate_payload, indent=2, ensure_ascii=False),
        )
        .strip()
    )
