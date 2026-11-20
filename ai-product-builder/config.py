"""GPT role profiles. Override individual model IDs via environment variables."""
import os

MODELS = {
    "architecture": "gpt-6.2-astra",
    "engineering": "gpt-6.2-sol",
    "business": "gpt-6.1-terra",
    "routine": "gpt-6-luna",
}
ARCHITECT = {"solution-architect", "senior-troubleshooter", "security-agent", "red-team-agent"}
BUSINESS = {"business-analyst-1", "business-analyst-2", "requirements-validator", "data-engineer", "prompt-engineer", "cost-optimization-agent"}
ROUTINE = {"monitoring-agent", "fact-checker"}


def model_for(role):
    if role in ARCHITECT:
        group = "architecture"
    elif role in BUSINESS:
        group = "business"
    elif role in ROUTINE:
        group = "routine"
    else:
        group = "engineering"
    return os.getenv(f"GPT_MODEL_{role.upper().replace('-', '_')}", os.getenv(f"GPT_MODEL_{group.upper()}", MODELS[group]))


def settings():
    return {
        "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "mode": "demo" if os.getenv("BUILDER_DEMO_MODE") == "1" else "openai",
        "models": {key: os.getenv(f"GPT_MODEL_{key.upper()}", val) for key, val in MODELS.items()},
        "host": os.getenv("BUILDER_HOST", "127.0.0.1"),
        "port": int(os.getenv("BUILDER_PORT", "8766")),
    }


def model_for_project(role, profile='standard'):
    """Project-level model profile with explicit environment overrides.

    Profiles do not manufacture API model IDs: if no override is configured,
    normal per-role routing stays in effect.
    """
    if profile in {'deep-review', 'cost-aware'}:
        override = os.getenv('GPT_MODEL_' + profile.upper().replace('-', '_'))
        if override:
            return override
    return model_for(role)
