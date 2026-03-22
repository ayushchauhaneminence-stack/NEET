import re
import json
import logging
import requests
import ollama
from pymilvus import MilvusClient
from src.settings.config import settings

logger = logging.getLogger(__name__)

# ── Milvus client (singleton) ────────────────────────────────────────────────
_client: MilvusClient | None = None

OUTPUT_FIELDS = [
    "item_id", "type", "subject", "chapter", "topic",
    "content", "question", "options", "answer", "solution",
    "images", "page", "doc_type",
]


def _get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=settings.ZILLIZ_URI, token=settings.ZILLIZ_TOKEN)
        logger.info("Connected to Zilliz Cloud")
    return _client


# ── Query classifier ─────────────────────────────────────────────────────────
def classify_query(query: str) -> str:
    q = query.lower().strip()

    if any(kw in q for kw in [
        "derive", "prove", "show that", "derivation",
        "proof", "deduce", "establish that",
    ]):
        return "derivation"

    if any(kw in q for kw in [
        "what is", "what are", "define", "explain",
        "meaning", "concept", "difference between",
        "describe", "why is", "why does", "why do",
        "how does", "how is", "tell me about",
        "state the", "what do you mean", "definition",
    ]):
        return "conceptual"

    if re.search(r'\d', q) or any(kw in q for kw in [
        "find", "calculate", "solve", "compute",
        "determine", "how many", "how much",
        "how far", "how long", "how fast",
    ]):
        return "numerical"

    return "conceptual"


def _get_type_filter(query_type: str) -> str | None:
    if query_type == "conceptual":
        return 'type in ["theory"]'
    elif query_type == "derivation":
        return 'type in ["theory", "example_solved"]'
    elif query_type == "numerical":
        return 'type in ["example_solved", "question_unsolved", "question_mcq"]'
    return None


# ── DeepSeek helper ──────────────────────────────────────────────────────────
def _deepseek_chat(messages: list[dict], max_tokens: int = 256,
                   temperature: float = 0, timeout: int = 30) -> str | None:
    try:
        resp = requests.post(
            f"{settings.DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            },
            timeout=timeout,
        )
        if resp.ok:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("DeepSeek call failed: %s", e)
    return None


def _parse_json_response(raw: str) -> any:
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())


# ── LLM Re-ranking ───────────────────────────────────────────────────────────
def llm_rerank(results: list, query: str, top_k: int = 5) -> list[dict]:
    if not results:
        return []

    candidates = results[:min(10, len(results))]

    chunk_summaries = []
    for i, hit in enumerate(candidates):
        e = hit["entity"]
        parts = []
        if e.get("topic"):
            parts.append(f"Topic: {e['topic']}")
        if e.get("content"):
            parts.append(f"Content: {e['content'][:400]}")
        if e.get("question"):
            parts.append(f"Question: {e['question'][:200]}")
        if e.get("solution"):
            parts.append(f"Solution: {e['solution'][:200]}")
        chunk_summaries.append(f"[Chunk {i}]\n" + "\n".join(parts))

    chunks_text = "\n\n".join(chunk_summaries)

    rerank_prompt = f"""You are a search relevance judge for a physics education system.

QUERY: "{query}"

Below are {len(candidates)} candidate chunks retrieved from a textbook. Rank them by how directly and precisely they answer the query.

Ranking criteria:
- A chunk that contains the EXACT definition, concept, or derivation asked for should rank highest
- A chunk that explains the concept with formulas and intuition ranks higher than one with just formulas
- A chunk that only tangentially mentions the topic should rank lowest
- Prefer chunks with explanations over chunks with only equations

CHUNKS:
{chunks_text}

Return ONLY a JSON array of chunk indices ordered from most relevant to least relevant.
Example: [3, 0, 7, 1, 2]

Your response must be ONLY the JSON array, nothing else."""

    raw = _deepseek_chat([{"role": "user", "content": rerank_prompt}])
    if raw:
        try:
            ranked_indices = _parse_json_response(raw)
            reranked = []
            seen = set()
            for idx in ranked_indices:
                if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                    hit = candidates[idx]
                    reranked.append({
                        "id": hit["id"],
                        "distance": hit["distance"],
                        "entity": hit["entity"],
                        "semantic_score": hit["distance"],
                        "llm_rank": len(reranked) + 1,
                    })
                    seen.add(idx)

            for i, hit in enumerate(candidates):
                if i not in seen:
                    reranked.append({
                        "id": hit["id"],
                        "distance": hit["distance"],
                        "entity": hit["entity"],
                        "semantic_score": hit["distance"],
                        "llm_rank": len(reranked) + 1,
                    })

            return reranked[:top_k]
        except (json.JSONDecodeError, TypeError):
            logger.warning("LLM re-rank JSON parse failed, using semantic order")

    return [
        {
            "id": hit["id"],
            "distance": hit["distance"],
            "entity": hit["entity"],
            "semantic_score": hit["distance"],
            "llm_rank": i + 1,
        }
        for i, hit in enumerate(candidates[:top_k])
    ]


# ── Confidence scoring ───────────────────────────────────────────────────────
def assess_confidence(top_results: list, query: str) -> tuple[str, str, str | None]:
    if not top_results:
        return "low", "No results found", "Try rephrasing your query"

    summaries = []
    for i, hit in enumerate(top_results[:3]):
        e = hit["entity"]
        parts = [f"Topic: {e.get('topic', 'N/A')}"]
        if e.get("content"):
            parts.append(f"Content: {e['content'][:300]}")
        if e.get("question"):
            parts.append(f"Question: {e['question'][:150]}")
        summaries.append(f"[Result {i+1}]\n" + "\n".join(parts))

    results_text = "\n\n".join(summaries)

    confidence_prompt = f"""You are a retrieval quality judge for a physics education system.

STUDENT QUERY: "{query}"

TOP 3 RETRIEVED RESULTS:
{results_text}

Assess how well these results answer the student's query. Respond with ONLY a JSON object:
{{
  "confidence": "high" | "medium" | "low",
  "reason": "one-line explanation",
  "suggestion": "null if high confidence, else a clarification question or rephrased query to help the student"
}}

Rules:
- "high": Result 1 directly and precisely answers the query
- "medium": Results are related but not a precise match
- "low": Results are off-topic or the query is too vague

Return ONLY the JSON object, nothing else."""

    raw = _deepseek_chat([{"role": "user", "content": confidence_prompt}], timeout=20)
    if raw:
        try:
            result = _parse_json_response(raw)
            return (
                result.get("confidence", "medium"),
                result.get("reason", ""),
                result.get("suggestion"),
            )
        except (json.JSONDecodeError, TypeError):
            logger.warning("Confidence parse failed")

    return "medium", "Could not assess confidence", None


# ── LLM answer generation ────────────────────────────────────────────────────
def generate_answer(top_results: list, query: str,
                    confidence: str) -> str | None:
    if not top_results:
        return None

    context_parts = []
    for idx, hit in enumerate(top_results[:3]):
        e = hit["entity"]
        parts = []
        if e.get("topic"):
            parts.append(f"Topic: {e['topic']}")
        if e.get("content"):
            parts.append(e["content"])
        if e.get("question"):
            parts.append(f"Question: {e['question']}")
        if e.get("solution"):
            parts.append(f"Solution: {e['solution']}")
        if e.get("answer"):
            parts.append(f"Answer: {e['answer']}")
        context_parts.append(f"[Source {idx+1}]\n" + "\n".join(parts))

    context = "\n\n---\n\n".join(context_parts)

    confidence_note = ""
    if confidence == "low":
        confidence_note = (
            "\nIMPORTANT: The retrieved context may not directly answer the question. "
            "If the context is insufficient, clearly state what's missing and what "
            "additional material the student should look for.\n"
        )
    elif confidence == "medium":
        confidence_note = (
            "\nNote: The retrieved context is partially relevant. "
            "Answer what you can from the context and note any gaps.\n"
        )

    system_msg = f"""You are a physics teacher answering a NEET student's question using ONLY the textbook excerpts provided below.

STRICT GROUNDING RULES — follow these in order:
1. READ the context carefully. Find sentences, phrases, or reasoning in the context that directly address the student's question.
2. BUILD your answer by quoting or closely paraphrasing the textbook. For each key claim, cite which Source it comes from (e.g., "As stated in Source 1, ...").
3. If the context contains the relevant explanation — even implicitly — extract and highlight that reasoning. Do NOT say "missing from context" if the idea is present in different words.
4. Add formulas from the context in LaTeX.
5. You may add a short physical intuition or analogy AFTER presenting the textbook's own reasoning — but label it clearly as "Intuition:" so the student knows it's supplementary.
6. If and ONLY if the context truly does not address the question at all, say: "The provided textbook excerpts do not cover this. You should refer to [specific topic/section]."

FORMAT:
- Use clear headings
- Quote or closely paraphrase the textbook (cite Source number)
- Add formulas in LaTeX
- Keep supplementary intuition brief and clearly labeled
{confidence_note}
TEXTBOOK CONTEXT:
{context}"""

    answer = _deepseek_chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ],
        max_tokens=2048,
        temperature=0.3,
        timeout=120,
    )
    return answer


# ── Format result for API response ────────────────────────────────────────────
def _format_hit(hit: dict) -> dict:
    e = hit["entity"]
    images = json.loads(e["images"]) if e.get("images") else []
    options = json.loads(e["options"]) if e.get("options") else None

    return {
        "item_id": e.get("item_id", ""),
        "type": e.get("type", ""),
        "subject": e.get("subject", ""),
        "chapter": e.get("chapter", ""),
        "topic": e.get("topic", ""),
        "content": e.get("content") or None,
        "question": e.get("question") or None,
        "options": options,
        "answer": e.get("answer") or None,
        "solution": e.get("solution") or None,
        "images": images,
        "page": e.get("page", 0),
        "doc_type": e.get("doc_type", ""),
        "semantic_score": hit.get("semantic_score", hit.get("distance", 0)),
        "llm_rank": hit.get("llm_rank", 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Main retrieval function
# ══════════════════════════════════════════════════════════════════════════════
def retrieve(query: str) -> dict:
    """
    Full retrieval pipeline:
      1. Classify query
      2. Embed + vector search (filtered)
      3. LLM re-rank top candidates
      4. Confidence check (+ fallback if low)
      5. Generate grounded answer (conceptual/derivation)
    Returns a structured dict ready for the API response.
    """
    client = _get_client()

    # 1. Classify
    query_type = classify_query(query)
    type_filter = _get_type_filter(query_type)
    logger.info("Query: %s | Type: %s | Filter: %s", query, query_type, type_filter)

    # 2. Embed + search
    query_embedding = ollama.embed(
        model=settings.EMBED_MODEL, input=query
    )["embeddings"][0]

    search_kwargs = dict(
        collection_name=settings.COLLECTION_NAME,
        data=[query_embedding],
        limit=15,
        output_fields=OUTPUT_FIELDS,
    )
    if type_filter:
        search_kwargs["filter"] = type_filter

    results = client.search(**search_kwargs)
    raw_results = results[0] if results else []

    # 3. LLM re-rank
    top_results = llm_rerank(raw_results, query, top_k=5)

    # 4. Confidence
    confidence, reason, suggestion = assess_confidence(top_results, query)
    logger.info("Confidence: %s — %s", confidence, reason)

    # 4b. Fallback if low confidence
    if confidence == "low" and type_filter:
        logger.info("Low confidence — broadening search (no type filter)")
        broad = client.search(
            collection_name=settings.COLLECTION_NAME,
            data=[query_embedding],
            limit=20,
            output_fields=OUTPUT_FIELDS,
        )
        if broad and broad[0]:
            top_results = llm_rerank(broad[0], query, top_k=5)
            confidence, reason, suggestion = assess_confidence(top_results, query)
            logger.info("After broadening — Confidence: %s — %s", confidence, reason)

    # 5. Generate answer for conceptual/derivation
    answer = None
    if query_type in ("conceptual", "derivation") and top_results:
        answer = generate_answer(top_results, query, confidence)

    # Build response
    return {
        "query": query,
        "query_type": query_type,
        "confidence": confidence,
        "confidence_reason": reason,
        "suggestion": suggestion if confidence != "high" else None,
        "results": [_format_hit(h) for h in top_results],
        "answer": answer,
    }