import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api.v1.routes.chat import chat_stream
from app.api.v1.routes.system import ready
from app.core.errors import AppError
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.hybrid_router import HybridRouter
from app.services.resource_intent_classifier import ResourceIntentClassifier


def post_chat(db, settings, message: str, mbti: str = "INTP", session_id: str | None = None):
    payload = ChatRequest(user_id="u1", session_id=session_id, message=message, mbti=mbti, metadata={})
    return ChatService(settings, db).chat(payload)


def fake_graph(reply: str = "graph reply", used_rag: bool = False):
    class FakeGraph:
        def invoke(self, state):
            return {
                "reply": reply,
                "intent": state.get("route_intent", "casual_chat"),
                "intensity": 3,
                "used_rag": used_rag,
                "session_id": "s-graph",
                "assistant_message_id": "m-graph",
            }

    return FakeGraph()


@pytest.mark.parametrize("message", ["أنا زهقانة", "مش عارف أنام", "؟", "يعني إيه vector database؟"])
def test_simple_messages_route_to_simple(db, settings, message):
    body = post_chat(db, settings, message)

    assert body.path_used == "simple"
    assert body.used_rag is False
    assert body.route_reason == "simple:fast_default"


def test_unclear_message_does_not_use_fixed_python_reply(db, settings, monkeypatch):
    calls = []

    def fake_simple_reply(self, **kwargs):
        calls.append(kwargs)
        return "model generated clarification"

    monkeypatch.setattr("app.services.llm_service.LLMService.generate_simple_reply_sync", fake_simple_reply)
    body = post_chat(db, settings, "؟")

    assert body.path_used == "simple"
    assert body.reply == "model generated clarification"
    assert len(calls) == 1


def test_repeated_long_sleep_problem_routes_to_langgraph(db, settings, monkeypatch):
    monkeypatch.setattr("app.graph.workflow.build_graph", lambda: fake_graph())
    message = "بقالي شهر مش بعرف أنام وبفضل أفكر كل يوم ومش قادرة أرتب شغلي ودراستي ومسؤولياتي"

    body = post_chat(db, settings, message, "INFP")

    assert body.path_used == "langgraph"
    assert body.route_reason == "langgraph:complex_message"


def test_decision_problem_with_multiple_constraints_routes_to_langgraph(db, settings, monkeypatch):
    monkeypatch.setattr("app.graph.workflow.build_graph", lambda: fake_graph())
    message = "محتارة أسيب الشغل ولا أكمل عشان الفلوس والوقت والدراسة ومسؤولية البيت"

    body = post_chat(db, settings, message, "INTJ")

    assert body.path_used == "langgraph"


def test_step_by_step_analysis_routes_to_langgraph(db, settings, monkeypatch):
    monkeypatch.setattr("app.graph.workflow.build_graph", lambda: fake_graph())

    body = post_chat(db, settings, "حللي المشكلة خطوة خطوة", "INTJ")

    assert body.path_used == "langgraph"


def test_explicit_file_request_with_rag_disabled_routes_simple_without_retriever(db, settings, monkeypatch):
    def fail_retrieve(*args, **kwargs):
        raise AssertionError("Retriever should not run when RAG is disabled")

    monkeypatch.setattr(settings, "enable_rag", False)
    monkeypatch.setattr("app.services.rag_service.RagService.retrieve", fail_retrieve)
    body = post_chat(db, settings, "ارجعي للملف اللي رفعته وقولي الخلاصة", "INTJ")

    assert body.path_used == "simple"
    assert body.used_rag is False
    assert body.route_reason == "rag_disabled:explicit_document_reference"


def test_explicit_file_request_with_rag_enabled_routes_to_rag(db, settings, monkeypatch):
    monkeypatch.setattr(settings, "enable_rag", True)
    monkeypatch.setattr("app.graph.workflow.build_graph", lambda: fake_graph(used_rag=True))

    body = post_chat(db, settings, "حسب المستند، إيه الخلاصة؟", "INTJ")

    assert body.path_used == "rag"
    assert body.used_rag is True
    assert body.route_reason == "rag:explicit_document_reference"


def test_simple_path_makes_exactly_one_llm_call(db, settings, monkeypatch):
    calls = []

    def fake_simple_reply(self, **kwargs):
        calls.append(kwargs)
        return "model generated reply"

    monkeypatch.setattr("app.services.llm_service.LLMService.generate_simple_reply_sync", fake_simple_reply)
    body = post_chat(db, settings, "أنا محتاجة أرتب يومي", "INTP")

    assert body.path_used == "simple"
    assert body.reply == "model generated reply"
    assert len(calls) == 1


def test_simple_path_does_not_call_langgraph_rag_or_embeddings(db, settings, monkeypatch):
    def fail_build_graph():
        raise AssertionError("LangGraph should not run for simple chat")

    def fail_retrieve(*args, **kwargs):
        raise AssertionError("RAG should not run for simple chat")

    def fail_embedding(*args, **kwargs):
        raise AssertionError("Embeddings should not run for simple chat")

    monkeypatch.setattr("app.graph.workflow.build_graph", fail_build_graph)
    monkeypatch.setattr("app.services.rag_service.RagService.retrieve", fail_retrieve)
    monkeypatch.setattr("app.services.embedding_service.EmbeddingService.embed_query", fail_embedding)
    body = post_chat(db, settings, "مش عارف أنام", "INTP")

    assert body.path_used == "simple"


def test_simple_path_does_not_require_db_or_redis(settings):
    response = post_chat(None, settings, "مرحبا", "INTP")

    assert response.path_used == "simple"
    assert response.used_rag is False


def test_first_issue_turn_does_not_recommend(db, settings):
    body = post_chat(db, settings, "مشاعري متلخبطة ومش عارفة أتعامل معها", "INFP", session_id="issue-1")

    assert body.resource_action == "no_recommendation"
    assert body.used_resource_rag is False


def test_second_issue_turn_does_not_recommend(db, settings):
    session_id = "issue-2"
    post_chat(db, settings, "مشاعري متلخبطة ومش عارفة أتعامل معها", "INFP", session_id=session_id)
    body = post_chat(db, settings, "كل يوم بحس إن المشاعر بتغرقني", "INFP", session_id=session_id)

    assert body.resource_action == "no_recommendation"
    assert body.used_resource_rag is False


def test_third_issue_turn_medium_depth_asks_permission_with_llm(db, settings, monkeypatch):
    calls = []

    def fake_permission(self, **kwargs):
        calls.append(kwargs)
        return "permission generated by model"

    monkeypatch.setattr("app.services.llm_service.LLMService.generate_resource_permission_sync", fake_permission)
    session_id = "issue-3"
    post_chat(db, settings, "مشاعري متلخبطة ومش عارفة أتعامل معها", "INFP", session_id=session_id)
    post_chat(db, settings, "كل يوم بحس إن المشاعر بتغرقني", "INFP", session_id=session_id)
    body = post_chat(db, settings, "لسه نفس المشاعر ومحتاجة أفهمها بهدوء", "INFP", session_id=session_id)

    assert body.path_used == "resource_rag"
    assert body.resource_action == "ask_permission"
    assert body.recommended_resources == []
    assert body.reply == "permission generated by model"
    assert len(calls) == 1


def test_high_depth_repeated_issue_can_recommend_from_catalog(db, settings, monkeypatch):
    monkeypatch.setattr(settings, "resource_rag_cooldown_turns", 0)
    session_id = "issue-4"
    post_chat(db, settings, "مشاعري متلخبطة ومش عارفة أتعامل معها", "INFP", session_id=session_id)
    post_chat(db, settings, "كل يوم بحس إن المشاعر بتغرقني", "INFP", session_id=session_id)
    body = post_chat(db, settings, "بقالي شهر مشاعري مضغوطة وكل يوم بفضل أفكر ومش قادرة أوقف", "INFP", session_id=session_id)

    assert body.path_used == "resource_rag"
    assert body.resource_action == "recommend_now"
    assert len(body.recommended_resources) == 1
    assert body.recommended_resources[0]["url"]


def test_explicit_resource_request_recommends_immediately(db, settings):
    body = post_chat(db, settings, "ابعتي حاجة أسمعها عن المشاعر", "INFP", session_id="resource-explicit")

    assert body.path_used == "resource_rag"
    assert body.resource_action == "recommend_now"
    assert len(body.recommended_resources) == 1


def test_overthinking_video_request_recommends_one_catalog_resource(db, settings):
    body = post_chat(db, settings, "رشحيلي فيديو عن التفكير المفرط", "INTP", session_id="resource-overthinking")
    catalog = Path("mbti_resource_catalog.jsonl").read_text(encoding="utf-8")

    assert body.path_used == "resource_rag"
    assert body.used_resource_rag is True
    assert body.resource_action == "recommend_now"
    assert len(body.recommended_resources) == 1
    assert body.recommended_resources[0]["url"] in catalog
    assert "youtube.com/results" not in body.reply
    assert "search" not in body.reply.lower()


def test_max_three_only_when_user_asks_for_more(db, settings):
    default_body = post_chat(db, settings, "ابعتي فيديو عن المشاعر", "INFP", session_id="resource-one")
    more_body = post_chat(db, settings, "ابعتي 3 فيديوهات عن المشاعر", "INFP", session_id="resource-three")

    assert len(default_body.recommended_resources) == 1
    assert 1 <= len(more_body.recommended_resources) <= 3


def test_cooldown_blocks_repeated_recommendations(db, settings):
    session_id = "resource-cooldown"
    first = post_chat(db, settings, "ابعتي فيديو عن المشاعر", "INFP", session_id=session_id)
    second = post_chat(db, settings, "ابعتي فيديو تاني عن المشاعر", "INFP", session_id=session_id)

    assert first.resource_action == "recommend_now"
    assert second.resource_action == "no_recommendation"


def test_declined_resources_block_recommendation(db, settings):
    session_id = "resource-decline"
    post_chat(db, settings, "مشاعري صعبة بس بلاش لينكات", "INFP", session_id=session_id)
    body = post_chat(db, settings, "ابعتي فيديو عن المشاعر", "INFP", session_id=session_id)

    assert body.resource_action == "no_recommendation"


def test_issue_change_resets_issue_thread(db, settings):
    session_id = "issue-change"
    post_chat(db, settings, "مشاعري متلخبطة ومش عارفة أتعامل معها", "INFP", session_id=session_id)
    post_chat(db, settings, "كل يوم بحس إن المشاعر بتغرقني", "INFP", session_id=session_id)
    body = post_chat(db, settings, "محتارة أسيب الشغل ولا أكمل", "INFP", session_id=session_id)

    assert body.resource_action == "no_recommendation"
    assert body.conversation_stage in {"initial_problem", "action_planning", "deepening"}


def test_resource_urls_exist_in_catalog_and_no_invented_links(db, settings):
    body = post_chat(db, settings, "ابعتي فيديو عن المشاعر", "INFP", session_id="resource-catalog")
    catalog = Path("mbti_resource_catalog.jsonl").read_text(encoding="utf-8")

    assert body.recommended_resources
    for resource in body.recommended_resources:
        assert resource["url"] in catalog


def test_deterministic_resource_classifier_loads_patterns_from_yaml(settings, monkeypatch):
    monkeypatch.setattr(settings, "resource_intent_classifier", "deterministic")
    result = ResourceIntentClassifier(settings).classify("رشحيلي فيديو عن التفكير المفرط")

    assert result.user_requested_resource is True
    assert "video" in result.requested_media_types
    assert result.confidence == 1.0


def test_semantic_resource_classifier_returns_structured_json(settings, monkeypatch):
    calls = []

    def fake_classify(self, **kwargs):
        calls.append(kwargs)
        return {
            "user_requested_resource": True,
            "requested_media_types": ["video"],
            "requested_topic": "overthinking",
            "confidence": 0.91,
            "reason": "semantic_match",
        }

    monkeypatch.setattr(settings, "resource_intent_classifier", "semantic")
    monkeypatch.setattr("app.services.llm_service.LLMService.classify_resource_intent_sync", fake_classify)
    result = ResourceIntentClassifier(settings).classify("anything")

    assert result.user_requested_resource is True
    assert result.requested_media_types == ["video"]
    assert result.requested_topic == "overthinking"
    assert result.confidence == 0.91
    assert calls


def test_langgraph_invokes_graph_only_for_complex_messages(db, settings, monkeypatch):
    calls = []

    def build_fake_graph():
        calls.append("graph")
        return fake_graph()

    monkeypatch.setattr("app.graph.workflow.build_graph", build_fake_graph)
    simple = post_chat(db, settings, "مرحبا", "INTP")
    complex_response = post_chat(db, settings, "حللي المشكلة خطوة خطوة", "INTJ")

    assert simple.path_used == "simple"
    assert complex_response.path_used == "langgraph"
    assert calls == ["graph"]


def test_router_outputs_requested_fields(settings):
    decision = HybridRouter(settings).route("حللي المشكلة خطوة خطوة")

    assert decision.route == "langgraph"
    assert decision.complexity_score >= settings.complexity_threshold
    assert decision.intent_hint
    assert decision.route_reason


def test_stream_provider_error_returns_before_stream_starts(db, settings, monkeypatch):
    from app.services.llm_service import LLMProviderError

    def fail_simple_reply(self, **kwargs):
        raise LLMProviderError("provider down")

    monkeypatch.setattr("app.services.llm_service.LLMService.generate_simple_reply_sync", fail_simple_reply)
    request_body = ChatRequest(user_id="u-stream", message="مرحبا", mbti="INTP", metadata={})
    request = SimpleNamespace(state=SimpleNamespace(request_id="stream-test"))

    with pytest.raises(AppError) as exc:
        asyncio.run(chat_stream(request_body=request_body, request=request, db=db, settings=settings))

    assert exc.value.status_code == 503
    assert exc.value.message == "LLM provider unavailable"


def test_invalid_mbti(db, settings):
    with pytest.raises(AppError):
        post_chat(db, settings, "مرحبا", "ABCD")


def test_missing_mbti():
    with pytest.raises(ValidationError):
        ChatRequest(user_id="u1", message="مرحبا", metadata={})


def test_ready_skips_db_and_redis_when_disabled(settings, monkeypatch):
    monkeypatch.setattr(settings, "enable_db", False)
    monkeypatch.setattr(settings, "enable_redis", False)
    monkeypatch.setattr(settings, "enable_rag", False)
    monkeypatch.setattr(settings, "enable_memory", False)

    response = ready(db=None, settings=settings)

    assert response.status == "ok"
    assert response.database == "skipped"
    assert response.redis == "skipped"


def test_env_example_contains_github_models_defaults():
    env_example = Path(".env.example").read_text(encoding="utf-8")

    assert "SIMPLE_CHAT" + "_MODE=true" in env_example
    assert "ENABLE" + "_LANGGRAPH=true" in env_example
    assert "ENABLE" + "_RAG=false" in env_example
    assert "ENABLE" + "_DB=false" in env_example
    assert "LLM" + "_PROVIDER=github_models" in env_example
    assert "GITHUB" + "_TOKEN=" in env_example
    assert "GITHUB" + "_MODELS_BASE_URL=https://models.github.ai/inference" in env_example
    assert "LLM" + "_MODEL=open" + "ai/gpt" + "-4.1-mini" in env_example


def test_invalid_provider_failure_returns_clean_503(db, settings, monkeypatch):
    from app.services.llm_service import LLMProviderError

    def fail_simple_reply(self, **kwargs):
        raise LLMProviderError("connection refused")

    monkeypatch.setattr("app.services.llm_service.LLMService.generate_simple_reply_sync", fail_simple_reply)

    with pytest.raises(AppError) as exc:
        post_chat(db, settings, "مرحبا", "INTP")

    assert exc.value.status_code == 503
    assert exc.value.message == "LLM provider unavailable"


def test_no_hardcoded_arabic_chatbot_replies_in_production_python():
    root = Path("app")
    allowed_files = {
        Path("app/services/intent_service.py"),
        Path("app/services/fast_intent_hints.py"),
        Path("app/services/hybrid_router.py"),
        Path("app/services/rag_service.py"),
        Path("app/services/quality_service.py"),
        Path("app/services/memory_service.py"),
        Path("app/graph/nodes/problem_detector.py"),
    }
    for path in root.rglob("*.py"):
        if path in allowed_files or path.parts[:2] == ("app", "prompts"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                returned = ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
                assert not any("\u0600" <= char <= "\u06ff" for char in returned), path


def test_no_hardcoded_resource_trigger_phrase_lists_in_python():
    forbidden = ("رشحيلي", "حاجة أسمعها", "حاجة أشوفها", "ابعتي حاجة")
    for path in Path("app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(phrase in text for phrase in forbidden), path
