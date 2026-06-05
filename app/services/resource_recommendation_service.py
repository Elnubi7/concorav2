import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.core.config import Settings
from app.core.policy_loader import load_policy
from app.services.conversation_intelligence import ConversationIntelligenceResult
from app.services.resource_intent_classifier import ResourceIntentResult

ResourceDecisionAction = Literal["recommend_now", "ask_permission", "no_recommendation"]


@dataclass(frozen=True)
class ResourceRecord:
    id: str
    mbti: str
    resource_title: str
    url: str
    media_type: str
    language: str
    issue_tags: list[str]
    trigger_keywords: list[str]

    def public_payload(self) -> dict:
        return {
            "id": self.id,
            "title": self.resource_title,
            "url": self.url,
            "media_type": self.media_type,
            "language": self.language,
            "issue_tags": self.issue_tags,
        }


@dataclass(frozen=True)
class ResourceDecision:
    action: ResourceDecisionAction
    resources: list[ResourceRecord]
    match_score: float
    reason: str
    conversation_depth_score: float
    issue_thread_turn_count: int
    detected_issue_tags: list[str]
    stable_issue_detected: bool
    user_requested_resource: bool
    requested_media_types: list[str]
    requested_topic: str | None


class ResourceRecommendationService:
    def __init__(self, settings: Settings, catalog_path: Path | None = None) -> None:
        self.settings = settings
        self.policy = load_policy("resource_rag_policy")
        catalog_paths = self.policy["catalog_paths"]
        self.catalog_path = catalog_path or Path(catalog_paths["jsonl"])
        self.csv_path = Path(catalog_paths["csv"])

    def decide(
        self,
        *,
        message: str,
        mbti: str,
        intelligence: ConversationIntelligenceResult,
        resource_intent: ResourceIntentResult,
    ) -> ResourceDecision:
        explicit = (
            resource_intent.user_requested_resource
            and resource_intent.confidence >= self.settings.resource_intent_confidence_threshold
        )
        if not self.settings.enable_resource_rag or not intelligence.should_consider_resource:
            return self._decision("no_recommendation", [], 0.0, "resource_rag_disabled_or_not_needed", intelligence, resource_intent)
        if intelligence.context.user_declined_resources:
            return self._decision("no_recommendation", [], 0.0, "user_declined_resources", intelligence, resource_intent)
        if self._cooldown_active(intelligence):
            return self._decision("no_recommendation", [], 0.0, "cooldown_active", intelligence, resource_intent)

        matches = self.search(
            message=message,
            mbti=mbti,
            issue_tags=intelligence.issue_tags,
            requested_media_types=resource_intent.requested_media_types,
            requested_topic=resource_intent.requested_topic,
        )
        if not matches:
            return self._decision("no_recommendation", [], 0.0, "no_catalog_match", intelligence, resource_intent)
        best_score = matches[0][1]
        max_items = self._max_items(message, explicit=explicit)
        resources = [record for record, _ in matches[:max_items]]

        if explicit and best_score >= self.settings.resource_rag_permission_threshold:
            self._remember_recommendation(intelligence, resources)
            return self._decision("recommend_now", resources, best_score, "explicit_resource_request", intelligence, resource_intent)
        if explicit:
            return self._decision("no_recommendation", [], best_score, "no_catalog_match", intelligence, resource_intent)
        if intelligence.context.issue_thread_turn_count < self.settings.resource_rag_min_turns_before_suggest:
            return self._decision("no_recommendation", [], best_score, "issue_thread_too_short", intelligence, resource_intent)
        if not intelligence.stable_issue_detected:
            return self._decision("no_recommendation", [], best_score, "unstable_issue", intelligence, resource_intent)
        if intelligence.depth_score < self.settings.resource_rag_permission_threshold:
            return self._decision("no_recommendation", [], best_score, "low_depth", intelligence, resource_intent)
        if intelligence.depth_score < self.settings.resource_rag_min_depth_before_recommend:
            if best_score >= self.settings.resource_rag_permission_threshold:
                return self._decision("ask_permission", [], best_score, "permission_before_recommendation", intelligence, resource_intent)
            return self._decision("no_recommendation", [], best_score, "weak_match", intelligence, resource_intent)
        if best_score >= self.settings.resource_rag_strong_match_threshold:
            self._remember_recommendation(intelligence, resources[: self.settings.resource_rag_max_default])
            return self._decision("recommend_now", resources[: self.settings.resource_rag_max_default], best_score, "deep_stable_issue", intelligence, resource_intent)
        return self._decision("no_recommendation", [], best_score, "weak_match", intelligence, resource_intent)

    def search(
        self,
        *,
        message: str,
        mbti: str,
        issue_tags: list[str],
        requested_media_types: list[str] | None = None,
        requested_topic: str | None = None,
    ) -> list[tuple[ResourceRecord, float]]:
        text = message.lower()
        topic = (requested_topic or "").lower()
        requested_media_types = requested_media_types or []
        weights = self.policy["ranking_weights"]
        compatible_tags = {key: set(value) for key, value in self.policy["compatible_issue_tags"].items()}
        scored: list[tuple[ResourceRecord, float]] = []
        for record in self.load_catalog():
            score = 0.0
            if record.mbti.upper() == mbti.upper():
                score += weights["same_mbti"]
            tag_overlap = set(record.issue_tags).intersection(issue_tags)
            if tag_overlap:
                score += min(0.45, weights["tag_overlap_base"] + weights["tag_overlap_increment"] * len(tag_overlap))
            compatible_overlap = set(record.issue_tags).intersection(
                set().union(*(compatible_tags.get(tag, set()) for tag in issue_tags)) if issue_tags else set()
            )
            if compatible_overlap:
                score += min(0.38, weights["compatible_tag_base"] + weights["compatible_tag_increment"] * len(compatible_overlap))
            if record.language == "ar":
                score += weights["arabic_language"]
            if requested_media_types and record.media_type in requested_media_types:
                score += 0.08
            if requested_topic and (topic in record.resource_title.lower() or any(topic in tag.lower() for tag in record.issue_tags)):
                score += 0.3
            if any(keyword and keyword.lower() in text for keyword in record.trigger_keywords):
                score += weights["catalog_trigger_keyword"]
            if "general_support" in record.issue_tags and issue_tags:
                score += weights["general_support"]
            if score:
                scored.append((record, min(score, 1.0)))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def load_catalog(self) -> list[ResourceRecord]:
        if self.catalog_path.exists():
            return [self._record_from_mapping(json.loads(line)) for line in self.catalog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        with self.csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
            return [self._record_from_mapping(row) for row in csv.DictReader(csv_file)]

    def _record_from_mapping(self, raw: dict) -> ResourceRecord:
        return ResourceRecord(
            id=str(raw["id"]),
            mbti=str(raw.get("mbti", "")),
            resource_title=str(raw.get("resource_title", "")),
            url=str(raw.get("url", "")),
            media_type=str(raw.get("media_type", "")),
            language=str(raw.get("language", "")),
            issue_tags=self._list_value(raw.get("issue_tags", [])),
            trigger_keywords=self._list_value(raw.get("trigger_keywords", [])),
        )

    def _list_value(self, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            if not value:
                return []
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return []

    def _max_items(self, message: str, *, explicit: bool) -> int:
        more_terms = self.policy["deterministic_patterns"]["more_results"]
        if explicit and any(term.lower() in message.lower() for term in more_terms):
            return self.settings.resource_rag_max_explicit
        return self.settings.resource_rag_max_default

    def _cooldown_active(self, intelligence: ConversationIntelligenceResult) -> bool:
        turns_since = intelligence.context.turn_index - intelligence.context.last_resource_recommended_at_turn
        return turns_since < self.settings.resource_rag_cooldown_turns

    def _remember_recommendation(self, intelligence: ConversationIntelligenceResult, resources: list[ResourceRecord]) -> None:
        intelligence.context.last_recommended_resource_ids = [resource.id for resource in resources]
        intelligence.context.last_resource_recommended_at_turn = intelligence.context.turn_index

    def _decision(
        self,
        action: ResourceDecisionAction,
        resources: list[ResourceRecord],
        score: float,
        reason: str,
        intelligence: ConversationIntelligenceResult,
        resource_intent: ResourceIntentResult,
    ) -> ResourceDecision:
        return ResourceDecision(
            action=action,
            resources=resources,
            match_score=score,
            reason=reason,
            conversation_depth_score=intelligence.depth_score,
            issue_thread_turn_count=intelligence.context.issue_thread_turn_count,
            detected_issue_tags=intelligence.issue_tags,
            stable_issue_detected=intelligence.stable_issue_detected,
            user_requested_resource=resource_intent.user_requested_resource,
            requested_media_types=resource_intent.requested_media_types,
            requested_topic=resource_intent.requested_topic,
        )
