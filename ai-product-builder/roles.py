"""AI job titles and scoped handoff instructions from the architecture diagram."""

ROLES = {
    "business-analyst-1": "Analyze business goals, users, existing workflows, and measurable outcomes.",
    "product-manager": "Set product scope and priorities. Define success metrics and acceptance criteria.",
    "requirements-validator": "Check requirements for conflicts, ambiguity, feasibility, and testability.",
    "business-analyst-2": "Independently review stakeholder assumptions and operational constraints.",
    "solution-architect": "Design components, interfaces, data flows, and trust boundaries. Deliver a proposed architecture diagram.",
    "orchestrator": "Design task routing, dependencies, checkpointing, retries, and escalation.",
    "project-manager": "Plan milestones, dependencies, risks, and ownership.",
    "dev-manager": "Sequence engineering work and define measurable completion criteria.",
    "rag-engineer": "Design ingestion, retrieval, grounding, citations, and abstention.",
    "agentic-ai-engineer": "Design bounded agent workflows, tool permissions, and safe failure handling.",
    "llm-engineer": "Specify OpenAI GPT model selection, inference settings, input-output contracts, latency budgets, and model evaluation. Distinguish measurements from proposed targets.",
    "memory-integration-engineer": "Specify memory storage, retention, access, and deletion.",
    "data-engineer": "Specify data schemas, validation, lineage, and sensitive data handling.",
    "prompt-engineer": "Draft role prompts, output contracts, and examples of expected behavior.",
    "tool-integration-engineer": "Specify tool interfaces, least privilege, auditability, and approval boundaries.",
    "performance-engineer": "Plan capacity, latency measurements, resource limits, and profiling.",
    "security-agent": "Review permissions, sensitive data exposure, trust boundaries, and mitigations.",
    "evaluation-agent": "Define quality metrics, representative fixtures, and acceptance thresholds.",
    "red-team-agent": "Plan defensive misuse and prompt-injection resilience checks with synthetic inputs. Focus on expected safe behavior and mitigations.",
    "cost-optimization-agent": "Act as Cost Estimator. Audit the supplied cost range and effort assumptions. Break effort down by delivery phase and distinguish implementation, recurring operation, and exclusions. Flag missing inputs and scope uncertainties. Use the calculated figures verbatim when available. Do not invent rates, currency, or totals.",
    "tester-1": "Plan unit, integration, and end-to-end checks for the product.",
    "tester-2": "Plan functional, retrieval, and agent behavior checks.",
    "lead-tester-1": "Review functional test coverage and acceptance criteria.",
    "lead-tester-2": "Review retrieval evaluation, citation accuracy, and groundedness.",
    "lead-tester-3": "Review workflow and tool behavior checks, including permissions and retries.",
    "lead-tester-4": "Review defensive security, performance, and scaling coverage.",
    "code-approval-manager-1": "Assess quality, security, and readiness evidence. Recommend a decision for human review.",
    "code-approval-manager-2": "Independently assess production readiness and recommend a decision for human review.",
    "delegated-approval-agent": "Represent the operator at internal checkpoints. Approve verified internal work, pause work, or schedule a focused meeting. Escalate external or production actions to the operator.",
    "release-manager": "Draft versioning, release, and rollback steps for human review.",
    "devops-mlops-agent": "Draft build, CI/CD, deployment, and operational verification steps.",
    "monitoring-agent": "Define health, latency, usage, drift, alert thresholds, and feedback signals.",
    "senior-troubleshooter": "Independently diagnose failed tasks, identify evidence-backed recovery steps, and specify how the original owner must verify the fix.",
}

ENGINEERING_ASSIGNMENTS = (
    "product interfaces and acceptance criteria",
    "workflow state and error recovery",
    "GPT model integration and response validation",
    "data ingestion and retrieval",
    "shared memory and retention",
    "tool interfaces and permissions",
    "performance and observability",
    "packaging and operational readiness",
)

DELIVERABLES = {
    "business-analyst-1": ("User journey", "Business outcome and measurement"),
    "business-analyst-2": ("Stakeholder exception", "Operational handoff"),
    "product-manager": ("Scope and priority decision", "Acceptance criteria and owner"),
    "requirements-validator": ("Requirement gap or conflict", "Falsifiable acceptance case"),
    "solution-architect": ("Component interfaces and data flow", "Trust boundary and design tradeoff"),
    "orchestrator": ("Routing and checkpoint decision", "Recovery and escalation rule"),
    "project-manager": ("Milestone and dependency sequence", "Delivery risk and accountable owner"),
    "dev-manager": ("Implementation assignment", "Completion and integration criteria"),
    "rag-engineer": ("Retrieval and authorization design", "Citation and abstention check"),
    "agentic-ai-engineer": ("Agent state transition and tool contract", "Failure and escalation behavior"),
    "llm-engineer": ("Model and response contract", "Inference evaluation and fallback"),
    "memory-integration-engineer": ("Memory schema and access", "Retention and deletion check"),
    "data-engineer": ("Source schema and lineage", "Ingestion validation and rejection path"),
    "prompt-engineer": ("Role prompt or output constraint", "Representative input and expected response"),
    "tool-integration-engineer": ("Tool input and output contract", "Permission and audit boundary"),
    "performance-engineer": ("Load scenario and latency measurement", "Capacity or timeout limit"),
    "security-agent": ("Trust boundary review", "Control evidence and residual risk"),
    "evaluation-agent": ("Evaluation fixture and scoring rule", "Evidence gap or threshold outcome"),
    "red-team-agent": ("Defensive resilience case", "Expected safe behavior and evidence"),
    "cost-optimization-agent": ("Effort and price reconciliation", "Cost driver, scope assumption, or exclusion"),
    "tester-1": ("Unit and integration evidence", "Observed failure or required coverage"),
    "tester-2": ("End-to-end scenario and input", "Expected and observed result"),
    "lead-tester-1": ("Acceptance coverage mapping", "Functional release blocker or evidence"),
    "lead-tester-2": ("Citation and groundedness review", "Retrieval regression evidence"),
    "lead-tester-3": ("Workflow safety and retry review", "State or tool failure evidence"),
    "lead-tester-4": ("Security and performance review", "Load or access control evidence"),
    "code-approval-manager-1": ("Code and security receipt review", "Independent quality decision"),
    "code-approval-manager-2": ("Release candidate evidence review", "Independent readiness recommendation"),
    "delegated-approval-agent": ("Checkpoint evidence and decision", "Scope of delegation and next action"),
    "release-manager": ("Release sequence and rollback trigger", "Rollback verification and owner"),
    "devops-mlops-agent": ("Build and deployment contract", "Operational verification and recovery"),
    "monitoring-agent": ("Signal, threshold, and alert owner", "Drift or incident response"),
    "senior-troubleshooter": ("Failure evidence and root-cause assessment", "Recovery steps and verification"),
}

for index, assignment in enumerate(ENGINEERING_ASSIGNMENTS, 1):
    ROLES[f"dev-{index}"] = f"Prepare a Codex implementation handoff for {assignment}, with file-level tasks and checks."
    ROLES[f"code-review-{index}"] = f"Independently review the supplied work for {assignment}. Cite evidence and identify missing verification."
    DELIVERABLES[f"dev-{index}"] = (f"Implementation for {assignment}", f"Files and acceptance check for {assignment}")
    DELIVERABLES[f"code-review-{index}"] = (f"Reviewed evidence for {assignment}", f"Findings and required fixes for {assignment}")


def run_role(name, brief, model, generate):
    instruction = (
        f"Your job title is {name}. {ROLES[name]} "
        f"Provide distinct sections for {DELIVERABLES[name][0]} and {DELIVERABLES[name][1]}. "
        "Include a next owner and verification criteria. "
        "Use at most 250 words. Label unknowns and assumptions."
    )
    content = generate(model, instruction, {
        "project_brief": brief,
        "model_selected_for_this_run": model,
        "verification_evidence": "No tests or benchmarks were run by this command.",
    })
    return f"# {name}\n\nGenerated by `{model}`. Review claims against the supplied evidence.\n\n{content}"
