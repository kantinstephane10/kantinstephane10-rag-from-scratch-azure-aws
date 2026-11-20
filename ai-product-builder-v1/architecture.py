"""Exact layer/role structure transcribed from the supplied architecture PDF."""
from roles import ROLES

LAYERS = [
    ("L1", "Strategy & Product", ["business-analyst-1", "product-manager", "requirements-validator", "business-analyst-2", "cost-optimization-agent"]),
    ("L2", "Architecture & Orchestration", ["solution-architect", "orchestrator", "project-manager"]),
    ("L3", "Development Management", ["dev-manager"]),
    ("L4", "Specialized Engineering", ["rag-engineer", "agentic-ai-engineer", "llm-engineer", "memory-integration-engineer", "data-engineer", "prompt-engineer", "tool-integration-engineer", "performance-engineer"]),
    ("L5", "Development & Review", [f"dev-{i}" for i in range(1,9)] + [f"code-review-{i}" for i in range(1,9)]),
    ("L6", "Quality, Security & Evaluation", ["security-agent", "evaluation-agent", "red-team-agent", "senior-troubleshooter"]),
    ("L7", "Testing & Validation", ["tester-1", "tester-2", "lead-tester-1", "lead-tester-2", "lead-tester-3", "lead-tester-4"]),
    ("L8", "Approval & Delivery", ["code-approval-manager-1", "code-approval-manager-2", "delegated-approval-agent", "release-manager", "devops-mlops-agent"]),
    ("L9", "Operations & Improvement", ["monitoring-agent"]),
]
# Human approval is a non-agent gate: it is deliberately NOT a model-executable role.


def blueprint():
    layers=[{"id":number,"name":label,"roles":ids} for number,label,ids in LAYERS]
    seen={role for layer in layers for role in layer["roles"]}
    assert len(seen)==len(ROLES)==49, (len(seen),len(ROLES))
    assert all(sum(role in layer["roles"] for layer in layers)==1 for role in ROLES)
    return {"layers":layers,"roles":ROLES,
            "human_approval_gate":True,
            "communication":{"type":"durable-any-to-any","peer_count":49,"supports_replies":True,"max_relay_hops":4},
            "platform":["OpenAI GPT Router","Project Memory (SQLite)","Shared Agent Mailbox","Workflow Engine","Tool Registry (planned)","Source Control (planned)","Execution Sandbox (planned)","Artifact Registry (planned)","Security & Policy","Observability"]}
