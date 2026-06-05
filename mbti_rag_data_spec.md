# MBTI Resource RAG Data Specification

## Goal
Use only the curated resources extracted from `أنماط الشخصية 16 for AI bot.docx` to recommend a video/book/podcast link when it is relevant.
The bot must not invent links and must not recommend resources on every message.

## Retrieval policy
Recommend a resource only when at least one condition is true:
1. The user explicitly asks for a recommendation, book, video, podcast, source, link, or something to watch/read.
2. The user describes a repeated or strong problem and a resource is clearly useful.
3. The assistant has already given a short answer and the user asks for deeper material.

Do not recommend when:
- the user is just chatting casually.
- the user asks for a simple direct answer.
- the match score is weak.
- there is no resource matching the MBTI + problem.

## Data record schema
Each resource record should contain:

```json
{
  "id": "string",
  "mbti": "INFJ|INFP|ENFJ|...",
  "resource_title": "string",
  "url": "string",
  "media_type": "video|podcast|book_summary_video|link",
  "language": "ar|en",
  "issue_tags": ["decision_paralysis", "emotional_absorption"],
  "trigger_keywords": ["optional keywords"],
  "recommendation_policy": "recommend_only_if_user_asks_or_message_strongly_matches_issue",
  "source_file": "أنماط الشخصية 16 for AI bot.docx",
  "source_paragraph_index": 0,
  "notes": "optional"
}
```

## Ranking logic
Preferred ranking:
1. Same MBTI type as user.
2. Issue tag matches detected problem.
3. Arabic resources first for Arabic users, unless user asks for English.
4. Prefer video/podcast for emotional/coaching issues.
5. Prefer book summary for planning, productivity, decision-making, and habits.
6. Return 1 resource by default, maximum 3 if the user asks for more.

## Output style
When recommending:
- give one short sentence explaining why this resource fits.
- include title + URL.
- do not list too many links.
- do not claim the resource is medically/clinically authoritative.
- say “ده مناسب كبداية” rather than overstating certainty.

## RAG trigger categories
Use resource RAG only for these intents:
- recommend_resource
- explicit_video_request
- explicit_book_request
- explicit_link_request
- deep_problem_with_resource_need

Do not use resource RAG for:
- casual_chat
- simple_advice unless user asks for a resource
- unclear_message
- normal knowledge question

## Ingestion files
- `mbti_resource_catalog.jsonl`: use for vector/document ingestion.
- `mbti_resource_catalog.csv`: use for manual QA and review.

## Safety note
The resources are self-help/educational resources. They should not be presented as professional medical or mental health treatment.
