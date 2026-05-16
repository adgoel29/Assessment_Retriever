import os
import json
from google import genai
from google.genai import types
from faiss_store import FAISSStore, Assessment
from dotenv import load_dotenv
load_dotenv()


SYSTEM_PROMPT = """You are an SHL assessment recommender agent. Your only job is to help hiring managers and recruiters find the right SHL assessments from the official SHL catalog.

RULES:
- Only recommend assessments that exist in the SHL catalog. Never invent assessment names or URLs.
- If the user's query is too vague (e.g. "I need an assessment"), ask ONE focused clarifying question before retrieving.
- Once you have enough context (role, seniority, or purpose), retrieve and recommend 1-10 assessments.
- If the user refines constraints mid-conversation, update the shortlist accordingly.
- If asked to compare two assessments, answer strictly from catalog data provided to you.
- Refuse politely if asked about general hiring advice, legal questions, or anything outside SHL assessments.
- Do not recommend on turn 1 if the query is vague.

OUTPUT FORMAT (always return valid JSON, nothing else):
{
  "reply": "<your conversational reply>",
  "recommendations": [
    {"name": "<assessment name>", "url": "<catalog url>", "test_type": "<single letter: A=Ability, P=Personality, K=Knowledge, S=Simulation, B=Biodata, C=Competency>"}
  ],
  "end_of_conversation": false
}

- recommendations must be an empty list [] when still clarifying or refusing.
- end_of_conversation is true only when the user confirms the shortlist is what they need.
- Return ONLY the JSON object. No markdown, no explanation outside the JSON.
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


def format_assessments_for_context(assessments: list[Assessment]) -> str:
    lines = []
    for a in assessments:
        lines.append(
            f"- Name: {a.name}\n"
            f"  URL: {a.url}\n"
            f"  Description: {a.description}\n"
            f"  Duration: {a.duration or 'N/A'}\n"
            f"  Job Levels: {', '.join(a.job_levels) or 'N/A'}\n"
            f"  Keys: {', '.join(a.keys) or 'N/A'}\n"
            f"  Remote: {a.remote} | Adaptive: {a.adaptive}\n"
        )
    return "\n".join(lines)


class SHLAgent:
    def __init__(self, store: FAISSStore):
        self.store = store
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _build_query_from_history(self, messages: list[dict]) -> str:
        """Extract a search query from the last few user messages."""
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        return " ".join(user_msgs)  # use last 3 user turns for context

    def _is_vague(self, messages: list[dict]) -> bool:
        """Heuristic: if only 1 user message and it's short, treat as vague."""
        user_msgs = [m for m in messages if m["role"] == "user"]
        if len(user_msgs) == 1 and len(user_msgs[0]["content"].split()) < 6:
            return True
        return False

    def chat(self, messages: list[dict]) -> dict:
        query = self._build_query_from_history(messages)
        # print(f"the query is {query}")
        candidates = self.store.search(query, top_k=10)
        # print(f"the search results are {candidates} \n\n")
        catalog_context = format_assessments_for_context(candidates)
        # print(f"the assessments are {catalog_context}\n\n")

        # Inject catalog context into the last user message
        injected_user_content = (
            f"{messages[-1]['content']}\n\n"
            f"[CATALOG CONTEXT - use only these assessments for recommendations]\n"
            f"{catalog_context}"
        )

       
        gemini_messages = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            gemini_messages.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        gemini_messages.append(types.Content(role="user", parts=[types.Part(text=injected_user_content)]))

        # print(f"\n\nthe inpit sent to gemini {gemini_messages}\n\n")

        try:
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
                contents=gemini_messages,
            )

            raw = (response.text or "").strip()

            

            if not raw:
                return {
                    "reply": "Empty model response.",
                    "recommendations": [],
                    "end_of_conversation": False,
                }

            start = raw.find("{")
            end = raw.rfind("}")

            if start != -1 and end != -1:
                raw = raw[start:end + 1]

            parsed = json.loads(raw)

        except Exception as e:
            print(f"\nERROR: {e}")

            parsed = {
                "reply": f"Error occurred: {str(e)}",
                "recommendations": [],
                "end_of_conversation": False,
            }

        
        parsed.setdefault("recommendations", [])
        parsed.setdefault("end_of_conversation", False)

        valid_urls = {a.url for a in candidates}
        parsed["recommendations"] = [
            r for r in parsed["recommendations"] if r.get("url") in valid_urls
        ]

        return parsed