class MetricsService:
    requests_total = 0
    graph_runs_total = 0
    rag_queries_total = 0

    @classmethod
    def increment_request(cls) -> None:
        cls.requests_total += 1

    @classmethod
    def increment_graph_run(cls) -> None:
        cls.graph_runs_total += 1

    @classmethod
    def increment_rag_query(cls) -> None:
        cls.rag_queries_total += 1
