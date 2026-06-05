from collections.abc import AsyncIterator
from dataclasses import replace
from time import perf_counter
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import AppError
from app.db.repositories.chat_repository import ChatRepository
from app.schemas.chat import ChatRequest, ChatResponse, SessionOut
from app.services.conversation_intelligence import ConversationIntelligence, ConversationIntelligenceResult
from app.services.fast_intent_hints import FastIntentHintService
from app.services.hybrid_router import HybridRouteDecision
from app.services.intent_service import IntensityService, IntentService
from app.services.llm_service import LLMProviderError, LLMService
from app.services.mbti_service import MbtiService
from app.services.memory_service import MemoryService
from app.services.resource_intent_classifier import ResourceIntentClassifier
from app.services.resource_recommendation_service import ResourceDecision, ResourceRecommendationService


class ChatService:
    def __init__(self, settings: Settings, db: Session | None) -> None:
        self.settings = settings
        self.db = db

    def chat(self, request: ChatRequest, request_id: str | None = None) -> ChatResponse:
        if self.settings.simple_chat_mode:
            intelligence = ConversationIntelligence(self.settings).analyze(
                user_id=request.user_id,
                session_id=request.session_id,
                message=request.message,
                metadata=request.metadata,
            )
            resource_intent = ResourceIntentClassifier(self.settings).classify(request.message)
            explicit_resource = (
                resource_intent.user_requested_resource
                and resource_intent.confidence >= self.settings.resource_intent_confidence_threshold
            )
            should_consider_resource = self.settings.enable_resource_rag and not intelligence.user_requested_document and (
                explicit_resource
                or (intelligence.stable_issue_detected and intelligence.depth_score >= self.settings.resource_rag_permission_threshold)
            )
            intelligence = replace(
                intelligence,
                user_requested_resource=explicit_resource,
                user_need="resource_request" if explicit_resource else intelligence.user_need,
                should_consider_resource=should_consider_resource,
                route_reason="explicit_resource_request" if explicit_resource else intelligence.route_reason,
            )
            if intelligence.user_requested_document:
                if self.settings.enable_rag:
                    return self._graph_chat(request, self._route("rag", intelligence), request_id=request_id, intelligence=intelligence)
                return self._simple_chat(request, self._route("rag_disabled", intelligence), intelligence=intelligence)
            resource_decision = ResourceRecommendationService(self.settings).decide(
                message=request.message,
                mbti=request.mbti,
                intelligence=intelligence,
                resource_intent=resource_intent,
            )
            if resource_decision.action in {"ask_permission", "recommend_now"}:
                return self._resource_chat(request, intelligence, resource_decision)
            if intelligence.should_use_langgraph:
                return self._graph_chat(request, self._route("langgraph", intelligence), request_id=request_id, intelligence=intelligence)
            return self._simple_chat(request, self._route("simple", intelligence), intelligence=intelligence)
        hints = FastIntentHintService().analyze(request.message)
        if self._should_use_fast_path(request, hints.likely_rag_request):
            return self._fast_chat(request)
        return self._full_graph_chat(request, request_id=request_id)

    def _should_use_fast_path(self, request: ChatRequest, likely_rag_request: bool) -> bool:
        if not self.settings.always_fast_chat:
            return False
        if request.metadata.get("force_full_graph") or request.metadata.get("debug"):
            return False
        return not likely_rag_request

    def _simple_chat(
        self,
        request: ChatRequest,
        route: HybridRouteDecision | None = None,
        intelligence: ConversationIntelligenceResult | None = None,
    ) -> ChatResponse:
        started = perf_counter()
        message = request.message.strip()
        if not message:
            raise AppError("Message is required", 422)
        if len(message) > 12000:
            raise AppError("Message is too long", 413)
        try:
            MbtiService().validate(request.mbti)
        except ValueError as exc:
            raise AppError(str(exc), 422) from exc

        hints = FastIntentHintService().analyze(message)
        intent = IntentService().classify(message).intent
        intensity = IntensityService().classify(intent, message)
        rag_disabled_for_request = bool(route and route.route == "rag_disabled") or (hints.likely_rag_request and not self.settings.enable_rag)
        llm = LLMService(
            self.settings.llm_provider,
            self.settings.llm_model,
            self.settings.effective_llm_api_key,
            base_url=self.settings.effective_llm_base_url,
            timeout_seconds=self.settings.llm_timeout_seconds,
            max_retries=self.settings.llm_max_retries,
            app_env=self.settings.app_env,
        )
        try:
            reply = llm.generate_simple_reply_sync(
                message=message,
                rag_disabled=rag_disabled_for_request,
                intelligence=self._intelligence_payload(intelligence),
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
            )
        except LLMProviderError as exc:
            raise AppError("LLM provider unavailable", 503) from exc

        return ChatResponse(
            reply=reply,
            intent=intent,
            intensity=intensity,
            used_rag=False,
            path_used="simple",
            model=self.settings.llm_model,
            latency_ms=int((perf_counter() - started) * 1000),
            route_reason=route.route_reason if route else "simple:fast_default",
            conversation_stage=intelligence.conversation_stage if intelligence else None,
            depth_score=intelligence.depth_score if intelligence else None,
            complexity_score=intelligence.complexity_score if intelligence else route.complexity_score if route else None,
            resource_reason=None,
            session_id=request.session_id or f"simple-{uuid4()}",
            message_id=f"simple-{uuid4()}",
        )

    def _resource_chat(
        self,
        request: ChatRequest,
        intelligence: ConversationIntelligenceResult,
        resource_decision: ResourceDecision,
    ) -> ChatResponse:
        started = perf_counter()
        message = request.message.strip()
        try:
            MbtiService().validate(request.mbti)
        except ValueError as exc:
            raise AppError(str(exc), 422) from exc
        llm = LLMService(
            self.settings.llm_provider,
            self.settings.llm_model,
            self.settings.effective_llm_api_key,
            base_url=self.settings.effective_llm_base_url,
            timeout_seconds=self.settings.llm_timeout_seconds,
            max_retries=self.settings.llm_max_retries,
            app_env=self.settings.app_env,
        )
        payload = self._intelligence_payload(intelligence)
        resources = [resource.public_payload() for resource in resource_decision.resources]
        try:
            if resource_decision.action == "ask_permission":
                reply = llm.generate_resource_permission_sync(
                    message=message,
                    intelligence=payload,
                    max_tokens=self.settings.llm_max_tokens,
                    temperature=self.settings.llm_temperature,
                )
            else:
                reply = llm.generate_resource_recommendation_sync(
                    message=message,
                    intelligence=payload,
                    resources=resources,
                    max_tokens=self.settings.llm_max_tokens,
                    temperature=self.settings.llm_temperature,
                )
        except LLMProviderError as exc:
            raise AppError("LLM provider unavailable", 503) from exc
        intent = IntentService().classify(message).intent
        return ChatResponse(
            reply=reply,
            intent=intent,
            intensity=IntensityService().classify(intent, message),
            used_rag=False,
            used_resource_rag=True,
            resource_action=resource_decision.action,
            recommended_resources=resources,
            path_used="resource_rag",
            model=self.settings.llm_model,
            latency_ms=int((perf_counter() - started) * 1000),
            route_reason=resource_decision.reason,
            resource_reason=resource_decision.reason,
            conversation_stage=intelligence.conversation_stage,
            depth_score=intelligence.depth_score,
            complexity_score=intelligence.complexity_score,
            session_id=request.session_id or f"resource-{uuid4()}",
            message_id=f"resource-{uuid4()}",
        )

    def _fast_chat(self, request: ChatRequest) -> ChatResponse:
        started = perf_counter()
        message = request.message.strip()
        if not message:
            raise AppError("Message is required", 422)
        if len(message) > 12000:
            raise AppError("Message is too long", 413)
        try:
            mbti = MbtiService().validate(request.mbti)
            style = MbtiService().interpret(mbti)
        except ValueError as exc:
            raise AppError(str(exc), 422) from exc

        repo = ChatRepository(self.db)
        session = repo.get_or_create_session(request.user_id, request.session_id, request.metadata)
        previous_messages = repo.list_session_messages(session.id, limit=8)
        session_summary = self._compact_session_summary(previous_messages)
        user_message = repo.add_message(
            user_id=request.user_id,
            session_id=session.id,
            role="user",
            content=message,
            mbti=mbti,
            metadata=request.metadata,
        )
        hints = FastIntentHintService().analyze(message)
        intent = IntentService().classify(message).intent
        if intent == "needs_rag" and not hints.likely_rag_request:
            intent = hints.intent_hint
        intensity = IntensityService().classify(intent, message)
        llm = LLMService(
            self.settings.llm_provider,
            self.settings.llm_model,
            self.settings.effective_llm_api_key,
            base_url=self.settings.effective_llm_base_url,
            timeout_seconds=self.settings.llm_fast_timeout_seconds,
            max_retries=0,
            app_env=self.settings.app_env,
        )
        try:
            reply = llm.generate_fast_reply_sync(
                message=message,
                intent_hint=intent,
                intensity_hint=intensity,
                clarification_may_be_needed=hints.likely_unclear,
                style=style,
                session_summary=session_summary,
                max_tokens=self.settings.llm_fast_max_tokens,
                temperature=self.settings.llm_fast_temperature,
            )
        except LLMProviderError as exc:
            self.db.rollback()
            raise AppError("LLM provider unavailable", 503) from exc

        assistant_message = repo.add_message(
            user_id=request.user_id,
            session_id=session.id,
            role="assistant",
            content=reply,
            intent=intent,
            intensity=intensity,
            used_rag=False,
            mbti=mbti,
            metadata={"path_used": "fast", "user_message_id": user_message.id},
        )
        MemoryService(self.settings, self.db).maybe_write(request.user_id, message)
        self.db.commit()
        return ChatResponse(
            reply=reply,
            intent=intent,
            intensity=intensity,
            used_rag=False,
            path_used="fast",
            model=self.settings.llm_model,
            latency_ms=int((perf_counter() - started) * 1000),
            route_reason="fast:legacy",
            session_id=session.id,
            message_id=assistant_message.id,
        )

    def _full_graph_chat(self, request: ChatRequest, request_id: str | None = None) -> ChatResponse:
        return self._graph_chat(request, None, request_id=request_id)

    def _graph_chat(
        self,
        request: ChatRequest,
        route: HybridRouteDecision | None,
        request_id: str | None = None,
        intelligence: ConversationIntelligenceResult | None = None,
    ) -> ChatResponse:
        if not self.settings.enable_langgraph:
            raise AppError("LangGraph workflow is disabled", 503)
        if route and route.route == "rag" and self.db is None:
            raise AppError("Database is required for RAG workflow", 503)
        started = perf_counter()
        from app.graph.workflow import build_graph

        graph = build_graph()
        final_state = graph.invoke(
            {
                "user_id": request.user_id,
                "session_id": request.session_id,
                "message": request.message,
                "mbti": request.mbti,
                "metadata": request.metadata,
                "request_id": request_id,
                "db": self.db,
                "settings": self.settings,
            }
        )
        return ChatResponse(
            reply=final_state["reply"],
            intent=final_state["intent"],
            intensity=final_state["intensity"],
            used_rag=final_state.get("used_rag", False),
            path_used="rag" if route and route.route == "rag" else "langgraph",
            model=self.settings.llm_model,
            latency_ms=int((perf_counter() - started) * 1000),
            route_reason=route.route_reason if route else "langgraph:legacy",
            conversation_stage=intelligence.conversation_stage if intelligence else None,
            depth_score=intelligence.depth_score if intelligence else None,
            complexity_score=intelligence.complexity_score if intelligence else route.complexity_score if route else None,
            resource_reason=None,
            session_id=final_state["session_id"],
            message_id=final_state["assistant_message_id"],
        )

    def _route(self, route: str, intelligence: ConversationIntelligenceResult) -> HybridRouteDecision:
        route_reason = {
            "simple": "simple:fast_default",
            "langgraph": "langgraph:complex_message",
            "rag": "rag:explicit_document_reference",
            "rag_disabled": "rag_disabled:explicit_document_reference",
        }[route]
        return HybridRouteDecision(route, intelligence.complexity_score, intelligence.user_need, route_reason)

    def _intelligence_payload(self, intelligence: ConversationIntelligenceResult | None) -> dict:
        if not intelligence:
            return {}
        return {
            "conversation_stage": intelligence.conversation_stage,
            "user_need": intelligence.user_need,
            "depth_score": intelligence.depth_score,
            "complexity_score": intelligence.complexity_score,
            "issue_tags": intelligence.issue_tags,
            "stable_issue_detected": intelligence.stable_issue_detected,
            "user_requested_resource": intelligence.user_requested_resource,
            "user_requested_document": intelligence.user_requested_document,
            "should_consider_resource": intelligence.should_consider_resource,
            "route_reason": intelligence.route_reason,
        }

    async def stream_chat(self, request: ChatRequest, request_id: str | None = None) -> AsyncIterator[str]:
        response = self.chat(request, request_id=request_id)
        async for chunk in self.stream_response(response):
            yield chunk

    async def stream_response(self, response: ChatResponse) -> AsyncIterator[str]:
        for word in response.reply.split():
            yield word + " "

    def _compact_session_summary(self, messages: list) -> str:
        if not messages:
            return ""
        parts: list[str] = []
        for msg in messages[-6:]:
            content = " ".join(msg.content.split())
            if len(content) > 120:
                content = content[:117].rstrip() + "..."
            parts.append(f"{msg.role}: {content}")
        return " | ".join(parts)

    def get_session(self, session_id: str) -> SessionOut | None:
        repo = ChatRepository(self.db)
        session = repo.get_session(session_id)
        if not session:
            return None
        messages = repo.list_session_messages(session_id, limit=200)
        return SessionOut(
            id=session.id,
            user_id=session.user_id,
            is_archived=session.is_archived,
            messages=[
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "intent": msg.intent,
                    "intensity": msg.intensity,
                    "used_rag": msg.used_rag,
                    "created_at": msg.created_at.isoformat(),
                }
                for msg in messages
            ],
        )

    def delete_session(self, session_id: str) -> bool:
        deleted = ChatRepository(self.db).archive_or_delete_session(session_id, archive=self.settings.archive_deleted_sessions)
        self.db.commit()
        return deleted
