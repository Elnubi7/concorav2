Classify whether the user is asking for a resource recommendation.

Return JSON only with this shape:
{
  "user_requested_resource": true,
  "requested_media_types": [],
  "requested_topic": null,
  "confidence": 0.0,
  "reason": ""
}

Rules:
- Do not answer the user.
- Do not recommend anything.
- Detect intent semantically, not by exact phrase.
- Extract requested media types when present.
- Extract the requested topic when present.
- Do not include user-facing wording.
