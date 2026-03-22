import re
import json
import time
import logging
import requests
from collections import Counter

from src.settings.config import settings

logger = logging.getLogger(__name__)


# ── 2a. Inject page markers ──────────────────────────────────────────────────

def inject_page_markers(ocr_text: str) -> str:
    """Split on '---' separators and inject [PAGE N] markers."""
    page_parts = re.split(r'\n-{3,}\n', ocr_text)
    if len(page_parts) > 1:
        processing_text = ""
        for idx, page_content in enumerate(page_parts):
            page_num = idx + 1
            processing_text += f"\n\n[PAGE {page_num}]\n\n{page_content}"
        processing_text = processing_text.strip()
        logger.info("Detected %d pages, injected [PAGE N] markers", len(page_parts))
        return processing_text
    logger.info("No page separators detected — proceeding without page markers")
    return ocr_text


# ── 2b. Smart chunking ───────────────────────────────────────────────────────

def smart_chunk(text: str, max_size: int = 6000) -> list[str]:
    """Split at headings and example markers so complete examples stay intact."""
    split_pattern = re.compile(
        r'(?='
        r'\n#{1,4}\s+#?\s*Illustrative Example'
        r'|\n#{1,4}\s+\d+\.\d+\s'
        r'|\n#{1,4}\s+#?\s*Practice Exercise'
        r'|\n\[PAGE \d+\]'
        r')'
    )
    sections = split_pattern.split(text)

    chunks = []
    current = ""
    for section in sections:
        if len(current) + len(section) > max_size and current.strip():
            chunks.append(current.strip())
            current = section
        else:
            current += section
    if current.strip():
        chunks.append(current.strip())

    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_size * 1.5:
            sub_start = 0
            while sub_start < len(chunk):
                sub_end = min(sub_start + max_size, len(chunk))
                if sub_end < len(chunk):
                    nl = chunk.rfind("\n\n", sub_start, sub_end)
                    if nl > sub_start:
                        sub_end = nl
                final_chunks.append(chunk[sub_start:sub_end].strip())
                sub_start = sub_end
        else:
            final_chunks.append(chunk)

    return final_chunks


# ── 2c. System prompt ────────────────────────────────────────────────────────

STRUCTURE_PROMPT = """You are a physics textbook content extractor. Given OCR text, extract EVERY content item into structured JSON.

ITEM TYPES:
- "theory": Concepts, definitions, derivations, formulas, explanations
- "example_solved": Illustrative examples WITH complete solutions
- "question_unsolved": Practice problems (may have final answer but NO detailed solution)
- "question_mcq": Multiple choice questions with options A/B/C/D

CRITICAL RULES:
1. Preserve ALL image references exactly as they appear: ![alt](hash_img.jpg)
2. Extract image filenames (e.g. abc123_img.jpg) into the "images" array
3. Keep each Illustrative Example as ONE complete item with FULL question AND FULL solution — never split across items
4. Use CONSISTENT chapter name from the text headers throughout (e.g. "Kinematics")
5. If [PAGE N] markers exist in the text, set "page" to that number N
6. For Practice Exercises, each numbered sub-problem (i), (ii), etc. is a SEPARATE item
7. Answers in square brackets like [25.75 kph] after a practice problem belong to that problem
8. "subject" is always "Physics"
9. Do NOT omit any content — extract everything
10. IMPORTANT — SPLIT THEORY INTO SMALL FOCUSED ITEMS:
    - Each theory item must cover ONE specific concept, definition, or formula
    - Maximum ~500-800 characters per theory item
    - If a section covers multiple concepts (e.g. "average velocity" then "instantaneous velocity" then "speed vs velocity"), create SEPARATE theory items for EACH concept
    - Each sub-item gets its own specific "topic" name (e.g. "Definition of Instantaneous Velocity", "Relationship between Speed and Velocity Magnitude")
    - Include the relevant formula(s) WITH the explanation in each item — don't separate formulas from their explanations

Return ONLY a valid JSON array:
[
  {
    "type": "theory",
    "subject": "Physics",
    "chapter": "Kinematics",
    "topic": "Specific concept name",
    "content": "Focused explanation of ONE concept with its formula...",
    "question": null,
    "options": null,
    "answer": null,
    "solution": null,
    "images": ["hash_img.jpg"],
    "page": 3
  },
  {
    "type": "example_solved",
    "subject": "Physics",
    "chapter": "Kinematics",
    "topic": "Topic name",
    "content": "Context if any...",
    "question": "Full question text...",
    "options": null,
    "answer": "Final answer value",
    "solution": "Complete step-by-step solution — never truncate...",
    "images": [],
    "page": 4
  },
  {
    "type": "question_unsolved",
    "subject": "Physics",
    "chapter": "Kinematics",
    "topic": "Topic name",
    "content": null,
    "question": "Full question text...",
    "options": null,
    "answer": "25.75 kph",
    "solution": null,
    "images": [],
    "page": 5
  }
]"""


# ── Helper: parse LLM JSON ───────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> list[dict]:
    """Strip markdown fences and parse JSON."""
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()
    items = json.loads(raw)
    if isinstance(items, dict):
        items = items.get("items", [items])
    if not isinstance(items, list):
        items = [items]
    return items


# ── 2d. Send chunks to DeepSeek ──────────────────────────────────────────────

def _structure_chunks(text_parts: list[str]) -> list[dict]:
    """Send each chunk to DeepSeek and collect all structured items."""
    all_items = []

    for i, part in enumerate(text_parts):
        logger.info("Processing chunk %d/%d...", i + 1, len(text_parts))

        resp = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{settings.DEEPSEEK_BASE_URL}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek-chat",
                        "max_tokens": 8192,
                        "messages": [
                            {"role": "system", "content": STRUCTURE_PROMPT},
                            {"role": "user", "content": part},
                        ],
                    },
                    timeout=300,
                )
                if resp.ok:
                    break
                logger.warning("Attempt %d failed: %d", attempt + 1, resp.status_code)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                wait = 5 * (attempt + 1)
                logger.warning("Attempt %d timed out — retrying in %ds", attempt + 1, wait)
                time.sleep(wait)

        if resp is None or not resp.ok:
            logger.error("DeepSeek structuring failed on chunk %d after 3 attempts", i + 1)
            continue

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        try:
            items = _parse_llm_json(raw)
            all_items.extend(items)
            logger.info("Extracted %d items from chunk %d", len(items), i + 1)
        except json.JSONDecodeError as e:
            logger.error("JSON parse error on chunk %d: %s", i + 1, e)

    return all_items


# ── 2e. Merge fragmented examples ────────────────────────────────────────────

def _merge_fragmented_examples(all_items: list[dict]) -> list[dict]:
    """Merge example_solved items missing questions into their predecessor."""
    merged = []
    for item in all_items:
        is_continuation = (
            item.get("type") == "example_solved"
            and not item.get("question")
            and merged
            and merged[-1].get("type") == "example_solved"
        )
        if is_continuation:
            prev = merged[-1]
            extra = item.get("solution") or item.get("content") or ""
            if extra:
                prev["solution"] = (prev.get("solution") or "") + "\n" + extra
            prev["images"] = list(
                set((prev.get("images") or []) + (item.get("images") or []))
            )
            if not prev.get("answer") and item.get("answer"):
                prev["answer"] = item["answer"]
            if not prev.get("page") and item.get("page"):
                prev["page"] = item["page"]
        else:
            merged.append(item)
    return merged


# ── 2f. Split oversized theory ───────────────────────────────────────────────

MAX_THEORY_CHARS = 1200

SPLIT_PROMPT = """You are splitting a large physics theory chunk into smaller, focused sub-items.
Each sub-item should cover ONE specific concept, definition, or formula.
Target size: 400-800 characters per item.
Keep all formulas, image references, and explanations intact — just separate them by concept.

Return ONLY a JSON array of items with the same structure as the input.
Each item must have: type, subject, chapter, topic, content, question, options, answer, solution, images, page.
The "topic" for each sub-item must be specific (e.g. "Definition of Instantaneous Velocity" not just "Velocity").

ORIGINAL ITEM:
"""


def _split_oversized_theory(all_items: list[dict]) -> list[dict]:
    """Find theory items > MAX_THEORY_CHARS, ask DeepSeek to split them."""
    oversized = [
        (i, item) for i, item in enumerate(all_items)
        if item.get("type") == "theory"
        and len(item.get("content", "")) > MAX_THEORY_CHARS
    ]

    if not oversized:
        logger.info("No oversized theory items found (all <=%d chars)", MAX_THEORY_CHARS)
        return all_items

    logger.info("Found %d oversized theory items (>%d chars). Splitting...",
                len(oversized), MAX_THEORY_CHARS)

    for idx, item in reversed(oversized):
        logger.info("Splitting '%s' (%d chars)...",
                     item.get("topic", "?"), len(item.get("content", "")))
        item_json = json.dumps(item, ensure_ascii=False)

        try:
            resp = requests.post(
                f"{settings.DEEPSEEK_BASE_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "max_tokens": 4096,
                    "temperature": 0,
                    "messages": [
                        {"role": "user", "content": SPLIT_PROMPT + item_json},
                    ],
                },
                timeout=120,
            )

            if resp.ok:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                sub_items = _parse_llm_json(raw)
                if len(sub_items) > 1:
                    all_items[idx:idx + 1] = sub_items
                    logger.info("Split into %d sub-items", len(sub_items))
                else:
                    logger.info("LLM returned 1 item — keeping original")
            else:
                logger.warning("Split API failed: %d", resp.status_code)
        except Exception as e:
            logger.warning("Split failed: %s", e)

    return all_items


# ── Chapter normalization ────────────────────────────────────────────────────

def _normalize_chapters(all_items: list[dict]) -> list[dict]:
    """Set all items to the most common chapter name."""
    if not all_items:
        return all_items
    chapter_counts = Counter(item.get("chapter", "") for item in all_items)
    dominant_chapter = chapter_counts.most_common(1)[0][0]
    for item in all_items:
        item["chapter"] = dominant_chapter
    logger.info("Normalized chapters to '%s'", dominant_chapter)
    return all_items


# ── Assign unique IDs ────────────────────────────────────────────────────────

def _assign_ids(all_items: list[dict]) -> list[dict]:
    for i, item in enumerate(all_items):
        item["id"] = f"item_{i:04d}"
    return all_items


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Full structuring pipeline
# ══════════════════════════════════════════════════════════════════════════════

def structure_ocr_text(ocr_text: str) -> list[dict]:
    """
    Full structuring pipeline:
      page markers -> chunk -> DeepSeek LLM -> merge fragments ->
      split oversized theory -> normalize chapters -> assign IDs
    """
    # 2a. Page markers
    processing_text = inject_page_markers(ocr_text)

    # 2b. Smart chunking
    text_parts = smart_chunk(processing_text, max_size=6000)
    logger.info("Split into %d chunks for LLM processing", len(text_parts))

    # 2d. Send to DeepSeek
    all_items = _structure_chunks(text_parts)
    logger.info("Raw extraction: %d items", len(all_items))

    # 2e. Merge fragmented examples
    all_items = _merge_fragmented_examples(all_items)
    logger.info("After merging fragments: %d items", len(all_items))

    # 2f. Split oversized theory
    all_items = _split_oversized_theory(all_items)
    logger.info("After splitting: %d items", len(all_items))

    # Chapter normalization
    all_items = _normalize_chapters(all_items)

    # Assign IDs
    all_items = _assign_ids(all_items)

    # Summary
    type_counts = Counter(item.get("type") for item in all_items)
    logger.info("Final: %d items — %s", len(all_items), dict(type_counts))

    return all_items