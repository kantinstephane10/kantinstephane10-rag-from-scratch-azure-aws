"""OpenAI Responses API client. No local model transports or implicit provider fallbacks."""
import json
import os
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelError(Exception):
    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "outgoing": {
            "type": "array",
            "items": {"type": "object", "properties": {"recipient": {"type": "string"}, "message": {"type": "string"}},
                      "required": ["recipient", "message"], "additionalProperties": False},
        },
        "needs_human": {"type": "boolean"},
        "file_requests": {
            "type": "array", "items": {"type": "object", "properties": {
                "artifact_id": {"type": "string"},
                "page": {"type": ["integer", "null"]},
                "offset": {"type": "integer"},
                "query": {"type": "string"}},
                "required": ["artifact_id", "page", "offset", "query"],
                "additionalProperties": False},
        },
        "meeting_requests": {
            "type": "array", "items": {"type": "object",
                "properties": {"topic": {"type": "string"},
                               "attendees": {"type": "array", "items": {"type": "string"}}},
                "required": ["topic", "attendees"], "additionalProperties": False}},
    },
    "required": ["reply", "outgoing", "needs_human", "meeting_requests", "file_requests"],
    "additionalProperties": False,
}


def parse_response(data):
    if data.get("status") != "completed":
        raise ModelError("GPT response was not completed.", retryable=True)
    texts = [part.get("text", "") for item in data.get("output", [])
             if item.get("type") == "message" for part in item.get("content", [])
             if part.get("type") == "output_text"]
    if not texts or not "".join(texts).strip():
        raise ModelError("GPT response contained no output text.", retryable=True)
    try:
        obj = json.loads("".join(texts))
    except ValueError as exc:
        raise ModelError("GPT returned invalid structured JSON.") from exc
    if (not isinstance(obj, dict) or not isinstance(obj.get("reply"), str)
        or not isinstance(obj.get("needs_human"), bool)
        or not isinstance(obj.get("outgoing"), list)
        or not isinstance(obj.get("meeting_requests"), list)
        or not isinstance(obj.get("file_requests",[]),list)):
        raise ModelError("GPT response does not match the required fields.")
    for request in obj.get("file_requests",[]):
        if (not isinstance(request,dict) or not isinstance(request.get('artifact_id'),str)
            or not isinstance(request.get('offset'),int)
            or not isinstance(request.get('query'),str)
            or request.get('page') is not None and not isinstance(request['page'],int)):
            raise ModelError('GPT file request is malformed.')
    for msg in obj["outgoing"]:
        if not isinstance(msg, dict) or not isinstance(msg.get("recipient"), str) or not isinstance(msg.get("message"), str):
            raise ModelError("GPT outgoing message is malformed.")
    for req in obj["meeting_requests"]:
        if not isinstance(req, dict) or not isinstance(req.get("topic"),str) or not isinstance(req.get("attendees"),list) or any(not isinstance(a,str) for a in req["attendees"]):
            raise ModelError("GPT meeting request is malformed.")
    return obj


class GPTClient:
    def __init__(self, api_key=None, endpoint=None, timeout=60, transport=None, max_output_tokens=None):
        self.key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.endpoint = endpoint or "https://api.openai.com/v1/responses"
        self.timeout = timeout
        self.transport = transport or urlopen
        self.max_output_tokens = max_output_tokens

    def generate(self, role, model, instructions, context):
        if not self.key:
            raise ModelError("OPENAI_API_KEY is not configured. Queued work can resume after configuration.")
        body = {
            "model": model,
            "instructions": instructions,
            "input": json.dumps(context, ensure_ascii=False),
            "store": False,
            "text": {"format": {"type": "json_schema", "name": "agent_handoff", "schema": RESPONSE_SCHEMA, "strict": True}},
        }
        # Let the selected model/API determine output length instead of imposing a local cap.
        req = Request(self.endpoint, data=json.dumps(body).encode("utf-8"),
                      headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                      method="POST")
        try:
            with self.transport(req, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            transient = exc.code in {408, 409, 429, 500, 502, 503, 504}
            raise ModelError(f"OpenAI HTTP {exc.code}: check API key, model access, quota, and request configuration.", transient) from None
        except (socket.timeout, TimeoutError, URLError, OSError) as exc:
            raise ModelError(f"OpenAI connection failed: {type(exc).__name__}.", retryable=True) from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise ModelError("OpenAI returned an invalid JSON HTTP response.", retryable=True) from exc
        result = parse_response(payload)
        result["_usage"] = payload.get("usage") or {}
        return result


class DemoClient:
    """Deterministic offline demonstration: never presented as a live GPT result."""
    def generate(self, role, model, instructions, context):
        question = context.get("specific_question") or context.get("message") or context.get("topic") or ""
        if context.get("phase") in ("answer", "response"):
            message = (f"[OFFLINE DEMO — NOT GPT] {role} received a directed question from "
                       f"{context.get('question_asked_by', 'the meeting')}: '{question[:140]}'. "
                       "This is a test placeholder, not a researched answer or completed engineering task.")
        else:
            message = (f"[OFFLINE DEMO — NOT GPT] {role}: received '{question[:140]}'. "
                       "Proposed next step: validate requirements and produce reviewable evidence.")
        return {"reply": message, "outgoing": [], "needs_human": False, "meeting_requests": [], "file_requests": []}
