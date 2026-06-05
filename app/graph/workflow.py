try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover - local fallback when dependencies are not installed
    END = "__end__"
    START = "__start__"

    class _CompiledFallbackGraph:
        def __init__(self, nodes, edges, conditional_edges):
            self.nodes = nodes
            self.edges = edges
            self.conditional_edges = conditional_edges

        def invoke(self, state):
            current = self.edges[START][0]
            while current != END:
                state = self.nodes[current](state)
                if current in self.conditional_edges:
                    route_func, mapping = self.conditional_edges[current]
                    current = mapping[route_func(state)]
                else:
                    current = self.edges[current][0]
            return state

    class StateGraph:
        def __init__(self, _state_type):
            self.nodes = {}
            self.edges = {}
            self.conditional_edges = {}

        def add_node(self, name, func):
            self.nodes[name] = func

        def add_edge(self, source, target):
            self.edges.setdefault(source, []).append(target)

        def add_conditional_edges(self, source, route_func, mapping):
            self.conditional_edges[source] = (route_func, mapping)

        def compile(self):
            return _CompiledFallbackGraph(self.nodes, self.edges, self.conditional_edges)

from app.graph.nodes.common import timed_node
from app.graph.nodes.conversation_intent_router import conversation_intent_router
from app.graph.nodes.conversation_mode_selector import conversation_mode_selector
from app.graph.nodes.gemma_response_generator import gemma_response_generator
from app.graph.nodes.input_guard import input_guard
from app.graph.nodes.mbti_interpreter import mbti_interpreter
from app.graph.nodes.mbti_validator import mbti_validator
from app.graph.nodes.memory_writer import memory_writer
from app.graph.nodes.problem_detector import problem_detector
from app.graph.nodes.quality_guard import quality_guard
from app.graph.nodes.rag_decider import rag_decider
from app.graph.nodes.response_intensity_classifier import response_intensity_classifier
from app.graph.nodes.response_planner import response_planner
from app.graph.nodes.retriever import retriever
from app.graph.nodes.session_loader import session_loader
from app.graph.state import GraphState


def route_rag(state: GraphState) -> str:
    return "retriever" if state.get("rag_required") else "response_planner"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("input_guard", timed_node("input_guard", input_guard))
    graph.add_node("session_loader", timed_node("session_loader", session_loader))
    graph.add_node("mbti_validator", timed_node("mbti_validator", mbti_validator))
    graph.add_node("mbti_interpreter", timed_node("mbti_interpreter", mbti_interpreter))
    graph.add_node("conversation_intent_router", timed_node("conversation_intent_router", conversation_intent_router))
    graph.add_node("response_intensity_classifier", timed_node("response_intensity_classifier", response_intensity_classifier))
    graph.add_node("problem_detector", timed_node("problem_detector", problem_detector))
    graph.add_node("conversation_mode_selector", timed_node("conversation_mode_selector", conversation_mode_selector))
    graph.add_node("rag_decider", timed_node("rag_decider", rag_decider))
    graph.add_node("retriever", timed_node("retriever", retriever))
    graph.add_node("response_planner", timed_node("response_planner", response_planner))
    graph.add_node("gemma_response_generator", timed_node("gemma_response_generator", gemma_response_generator))
    graph.add_node("quality_guard", timed_node("quality_guard", quality_guard))
    graph.add_node("memory_writer", timed_node("memory_writer", memory_writer))

    graph.add_edge(START, "input_guard")
    graph.add_edge("input_guard", "session_loader")
    graph.add_edge("session_loader", "mbti_validator")
    graph.add_edge("mbti_validator", "mbti_interpreter")
    graph.add_edge("mbti_interpreter", "conversation_intent_router")
    graph.add_edge("conversation_intent_router", "response_intensity_classifier")
    graph.add_edge("response_intensity_classifier", "problem_detector")
    graph.add_edge("problem_detector", "conversation_mode_selector")
    graph.add_edge("conversation_mode_selector", "rag_decider")
    graph.add_conditional_edges("rag_decider", route_rag, {"retriever": "retriever", "response_planner": "response_planner"})
    graph.add_edge("retriever", "response_planner")
    graph.add_edge("response_planner", "gemma_response_generator")
    graph.add_edge("gemma_response_generator", "quality_guard")
    graph.add_edge("quality_guard", "memory_writer")
    graph.add_edge("memory_writer", END)
    return graph.compile()
