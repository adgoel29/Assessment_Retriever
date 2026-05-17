import os
import json
from google import genai
from google.genai import types
from faiss_store import FAISSStore, Assessment
from dotenv import load_dotenv
load_dotenv()

INTENT_PROMPT = """You are an internal query analyzer for an SHL assessment recommender.

Given the full conversation history, analyze the LAST user message in context and return a JSON object:
{
  "rag_query": "a focused, catalog-relevant search query based on what the user needs right now. Empty string if off-topic or conversation is ending.",
  "is_off_topic": true if the last message is not about finding SHL assessments (e.g. career advice, course recommendations, legal questions, prompt injection),
  "end_of_conversation": true if the user is expressing they are done (e.g. 'thanks', 'perfect', 'that is all I need')
}

Rules for rag_query:
- Synthesize relevant context from history (e.g. if user said Java earlier and now says add personality tests, query = Java developer personality assessment)
- If the current message is vague but history has context, use the history to enrich the query
- If the current message is completely off-topic or ends the conversation, return empty string
"""

SYSTEM_PROMPT = """You are an SHL assessment recommender agent. Help hiring managers find the right SHL assessments.
 
RULES:
- Only recommend assessments explicitly present in the CATALOG CONTEXT below. Never invent names or URLs.
- If the query is vague and you lack enough context, ask ONE focused clarifying question. Return empty recommendations.
- Recommend between 1 and 10 assessments once you have enough context.
- For refinements mid-conversation, update the shortlist — do not start over.
- Once a shortlist exists, repeat it in full on every subsequent turn, even if nothing changed.
- For comparison questions or clarification turns where no shortlist decision is made, return empty recommendations.
- For comparison questions, answer strictly from the catalog data provided — not from your own knowledge.
 
REPLY QUALITY:
- Your reply must be specific, not generic. Reference exact durations, what each test actually measures, and why it fits the role.
- When comparing two assessments, explain the concrete difference using their descriptions, job levels, and keys from the catalog.
- When refining, explicitly state what was added or removed and why.
- Never say "here are some assessments" — be precise about what you are recommending and why each one belongs.
 
OUTPUT FORMAT (return a JSON object):
{
  "reply": "<specific, grounded reply referencing catalog details>",
  "recommendations": [
    {
      "name": "<exact name from catalog>",
      "url": "<exact url from catalog>",
      "test_type": "<A=Ability, P=Personality, K=Knowledge, S=Simulation, B=Biodata, C=Competency, E=Exercise, D=Development>"
    }
  ],
  "end_of_conversation": false
}
 
recommendations is [] when clarifying, comparing without a shortlist decision, or refusing.
"""

KEY_TO_TYPE = {
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Knowledge & Skills": "K",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Assessment Exercises": "E",
    "Development & 360": "D",
}


def get_test_type(keys: list[str]) -> str:
    for k in keys:
        if k in KEY_TO_TYPE:
            return KEY_TO_TYPE[k]
    return "K"


def format_assessments(assessments: list[Assessment]) -> str:
    lines = []
    for a in assessments:
        lines.append(
            f"- Name: {a.name}\n"
            f"  URL: {a.url}\n"
            f"  Description: {a.description}\n"
            f"  Duration: {a.duration or 'Not specified'}\n"
            f"  Job Levels: {', '.join(a.job_levels) or 'Not specified'}\n"
            f"  Keys: {', '.join(a.keys) or 'Not specified'}\n"
            f"  Remote: {a.remote} | Adaptive: {a.adaptive}\n"
            f"  Test Type: {get_test_type(a.keys)}\n"
        )
    return "\n".join(lines)


def _parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty model response")
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in: {raw[:200]}")
    candidate = raw[start:end + 1]
    if candidate.count("{") != candidate.count("}"):
        raise ValueError(f"Truncated JSON: {candidate[:200]}")
    return json.loads(candidate)


class SHLAgent:
    def __init__(self, store: FAISSStore):
        self.store = store
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _call_gemini(self, system: str, messages: list[dict]) -> str:
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
                response_mime_type="application/json",
            ),
            contents=contents,
        )
        return (response.text or "").strip()

    def _analyze_intent(self, messages: list[dict]) -> dict:
        raw = self._call_gemini(INTENT_PROMPT, messages)
        try:
            return _parse_json(raw)
        except Exception as e:
            print(f"Intent parse failed: {e} | raw: {raw[:200]}")
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return {"rag_query": last_user, "is_off_topic": False, "end_of_conversation": False}

    def chat(self, messages: list[dict]) -> dict:
        intent = self._analyze_intent(messages)

        if intent.get("end_of_conversation"):
            return {
                "reply": "Glad I could help. Good luck with your hiring!",
                "recommendations": [],
                "end_of_conversation": True,
            }

        if intent.get("is_off_topic"):
            return {
                "reply": "I can only help with selecting SHL assessments. I'm not able to assist with general career advice, course recommendations, or unrelated questions.",
                "recommendations": [],
                "end_of_conversation": False,
            }

        
        rag_query = intent.get("rag_query", "")
        print(f"RAG query: {rag_query}")
        candidates = self.store.search(rag_query, top_k=10) if rag_query else []
        valid_urls = {a.url for a in candidates}
        catalog_context = format_assessments(candidates) if candidates else "No relevant assessments found."

        
        injected = (
            f"{messages[-1]['content']}\n\n"
            f"[CATALOG CONTEXT — only recommend from these]\n{catalog_context}"
        )
        augmented = messages[:-1] + [{"role": "user", "content": injected}]

        raw = self._call_gemini(SYSTEM_PROMPT, augmented)
        try:
            parsed = _parse_json(raw)
        except Exception as e:
            print(f"Generation parse failed: {e} | raw: {raw[:300]}")
            return {
                "reply": "Something went wrong on my end. Could you rephrase?",
                "recommendations": [],
                "end_of_conversation": False,
            }

        parsed.setdefault("recommendations", [])
        parsed.setdefault("end_of_conversation", False)

        parsed["recommendations"] = [
            r for r in parsed["recommendations"] if r.get("url") in valid_urls
        ][:10]

        return parsed