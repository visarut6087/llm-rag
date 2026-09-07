# Phase 5 Conversation-to-RAG Test Cases

These cases exercise query rewriting with the UI's `conversation_context` request field. Previous assistant messages may be present in the payload, but the backend rewriter uses only previous user turns for query coreference. Retrieved documents remain the only factual evidence.

| Case | Conversation / question | Expected retrieval behavior |
|---|---|---|
| A — standalone | No previous turns; ask a complete question | Retrieval query equals the original question |
| B — follow-up | User asks about an article, then asks `แล้วเรื่องนี้เกิดขึ้นเมื่อไหร่?` | Retrieval query includes the latest relevant user turn and the follow-up |
| C — pronoun/coreference | Previous user turn names an event; next question uses `เหตุการณ์นี้` | Retrieval query includes the named event; assistant history is not used as evidence |
| D — topic change | Previous topic is X; next question is a complete question about Y | Query is not contaminated with X when no follow-up marker is present |
| E — ambiguous follow-up | Previous user turn establishes a topic; next question is incomplete/uses a deictic marker | Rewrite uses only the latest user-turn context and reports rewrite metadata |
| F — unanswered | Rewritten query retrieves no supporting article | The existing grounded prompt instructs the model to abstain; no conversation turn becomes a citation |

For each case, record `original_query`, `rewritten_query`, `query_rewrite_used_context`, `sources`, and citation validity. Compare Case B/C/E with a request whose `conversation_context` is empty to measure the effect of memory-assisted rewriting.
