# Query Rewrite Measurement Notes

Phase 4 responses expose:

- `original_query`
- `rewritten_query`
- `query_rewrite_used_context`
- `query_rewrite_latency_ms`
- `query_rewrite_method`

For a controlled comparison, run the same query set with and without an explicit
`conversation_context` and compare retrieved source IDs, final ranks, scores,
and downstream answer quality. The original query must remain the user-visible
question in both runs.
