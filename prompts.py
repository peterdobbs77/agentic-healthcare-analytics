from typing import Dict, Any, List, Optional
import json
from enum import Enum
from langchain.messages import HumanMessage  # type: ignore[import-not-found]
from langgraph.graph import MessagesState
from langgraph.types import Command

class AgentKey(Enum):
    FHIR_RESEARCHER = "fhir_researcher"
    MEDICAL_RESEARCHER = "medical_researcher"
    KG_GENERATOR = "kg_generator"
    ML_ENGINEER = "ml_engineer"
    ML_EVALUATOR = "ml_evaluator"
    CHART_GEN = "chart_generator"
    CHART_SUM = "chart_summarizer"
    SYNTHESIZER = "synthesizer"

# Custom State class with specific keys
class State(MessagesState):
    user_query: Optional[str] # The user's original query
    enabled_agents: Optional[List[AgentKey]] # Makes our multi-agent system modular on which agents to include
    # Current plan only: mapping from step number (as string) to step definition
    plan: Optional[Dict[str, Dict[str, Any]]] # Listing the steps in the plan needed to achieve the goal.
    current_step: int # Marking the current step in the plan.
    last_reason: Optional[str] # Explains the executor’s decision to help maintain continuity and provide traceability.
    # Replan attempts tracked per step number
    replan_flag: Optional[bool] # Set by the executor to indicate that the planner should revise the plan.
    replan_attempts: Optional[Dict[int, int]] # Replan attempts tracked per step number.
    retrain_flag: Optional[bool] # Set by the executor to indicate that the machine_learning_engineer should retrain the model.
    retrain_attempts: Optional[Dict[int, int]] # Retrain attempts tracked per step number.
    agent_query: Optional[str]

MAX_REPLANS = 3
MAX_RETRAINS = 3


def get_agent_descriptions() -> Dict[AgentKey, Dict[str, Any]]:
    """
    Return structured agent descriptions with capabilities and guidelines.
    Edit this function to change how the planner/executor reason about agents.
    """
    return {
        AgentKey.FHIR_RESEARCHER: {
            "name": "FHIR Researcher",
            "capability": "Query FHIR data, including patient records, procedures, financial claims",
            "use_when": "EMR data is required",
            "limitations": "Cannot access external data",
            "output_format": "Relevant FHIR Resources and code or tools to retrieve them"
        },
        AgentKey.MEDICAL_RESEARCHER: {
            "name": "Medical Researcher",
            "capability": "Query medical research databases for context on patient condition",
            "use_when": "Additional medical knowledge or context is required",
            "limitations": "Cannot access internal data or actual individual patient records",
            "output_format": "Detailed summary of findings with references to source materials"
        },
        AgentKey.KG_GENERATOR: {
            "name": "Knowledge Graph Generator",
            "capability": "Construct a Knowledge Graph for the provided data",
            "use_when": "Retrieved documents are structured as JSON with references to other documents",
            "limitations": "Cannot add details to graph not contained in provided data",
            "output_format": "Raw knowledge graph"
        },
        AgentKey.ML_ENGINEER: {
            "name": "Machine Learning Engineer",
            "capability": "Selects and implements machine learning models on source data",
            "use_when": "Machine learning tasks are required (e.g., Prediction, Segmentation, Classification, Anomaly Detection, and more)",
            "limitations": "Relies on established best practices for model selection",
            "output_format": "Return an explanation for model selection and include Python code for implementing the model"
        },
        AgentKey.ML_EVALUATOR: {
            "name": "Machine Learning Evaluator",
            "capability": "Evaluates the machine learning model selection, implementation, efficiency, and accuracy",
            "use_when": "After machine_learning_engineer has created a model",
            "limitations": "Requires Python code, a sample model inference, and (optionally) additional data for evaluation",
            "output_format": "Written summary and analysis of machine learning model"
        },
        AgentKey.CHART_GEN: {
            "name": "Chart Generator",
            "capability": "Build visualizations from structured data",
            "use_when": "User explicitly requests charts, graphs, plots, visualizations (keywords: chart, graph, plot, visualise, bar-chart, line-chart, histogram, etc.)",
            "limitations": "Requires structured data input from previous steps",
            "output_format": "Visual charts and graphs"
        },
        AgentKey.CHART_SUM: {
            "name": "Chart Summarizer",
            "capability": "Summarize and explain chart visualizations",
            "use_when": "After chart_generator has created a visualization",
            "limitations": "Requires a chart as input",
            "output_format": "Written summary and analysis of chart content",
        },
        AgentKey.SYNTHESIZER: {
            "name": "Synthesizer",
            "capability": "Write comprehensive prose summaries of findings",
            "use_when": "Final step when no visualization is requested - combines all previous research",
            "limitations": "Requires research data from previous steps",
            "output_format": "Coherent written summary incorporating all findings and relevant visualizations",
            "position_requirement": "Should be used as final step",
        },
    }

def _get_enabled_agents(state: State | None = None) -> List[AgentKey]:
    """Return enabled agents; if absent, use baseline/default.

    Supports both dict-style and attribute-style state objects.
    """
    baseline = list(get_agent_descriptions().keys())
    if not state:
        return baseline
    val = state.get("enabled_agents") if hasattr(state, "get") else getattr(state, "enabled_agents", None)
    
    if isinstance(val, list) and val:
        allowed = set(get_agent_descriptions().keys())
        filtered = [a for a in val if a in allowed]
        return filtered
    return baseline

def format_agent_list_for_planning(state: State | None = None) -> str:
    """
    Format agent descriptions for the planning prompt.
    """
    descriptions = get_agent_descriptions()
    enabled_list = _get_enabled_agents(state)
    agent_list = []
    
    for agent_key, details in descriptions.items():
        if agent_key not in enabled_list:
            continue
        agent_list.append(f"  • `{agent_key}` – {details['capability']}")
    
    return "\n".join(agent_list)

def format_agent_guidelines_for_planning(state: State | None = None) -> str:
    """
    Format agent usage guidelines for the planning prompt.
    """
    descriptions = get_agent_descriptions()
    enabled = set(_get_enabled_agents(state))
    guidelines = []

    for agent_key, details in descriptions.items():
        if agent_key not in enabled:
            continue
        guidelines.append(f"- Use `{agent_key}` when {details['use_when'].lower()}.")
    
    return "\n".join(guidelines)

def format_agent_guidelines_for_executor(state: State | None = None) -> str:
    """
    Format agent usage guidelines for the executor prompt.
    """
    descriptions = get_agent_descriptions()
    enabled = _get_enabled_agents(state)
    guidelines = []
    
    if AgentKey.FHIR_RESEARCHER in enabled:
        guidelines.append(f"- Use `\"{AgentKey.FHIR_RESEARCHER}\"` for {descriptions[AgentKey.FHIR_RESEARCHER]['use_when'].lower()}.")
    if AgentKey.MEDICAL_RESEARCHER in enabled:
        guidelines.append(f"- Use `\"{AgentKey.MEDICAL_RESEARCHER}\"` for {descriptions[AgentKey.MEDICAL_RESEARCHER]['use_when'].lower()}.")
    if AgentKey.ML_ENGINEER in enabled:
        guidelines.append(f"- Use `\"{AgentKey.ML_ENGINEER}\"` for {descriptions[AgentKey.ML_ENGINEER]['use_when'].lower()}.")
    
    return "\n".join(guidelines)

def plan_prompt(state: State) -> HumanMessage:
    """
    Build the prompt that instructs the LLM to return a high‑level plan.
    """
    replan_flag   = state.get("replan_flag", False)
    user_query    = state.get("user_query", state["messages"][0].content)
    prior_plan    = state.get("plan") or {}
    replan_reason = state.get("last_reason", "")
    
    # Get agent descriptions dynamically
    
    agent_list = format_agent_list_for_planning(state)
    agent_guidelines = format_agent_guidelines_for_planning(state)

    enabled_list = _get_enabled_agents(state)

    # Build planner agent enum based on enabled agents
    enabled_for_planner = [a for a in enabled_list]
    planner_agent_enum = " | ".join(enabled_for_planner) or f"{AgentKey.FHIR_RESEARCHER} | {AgentKey.CHART_GEN} | {AgentKey.SYNTHESIZER}"

    prompt = f"""
        You are the **Planner** in a multi‑agent system.  Break the user's request
        into a sequence of numbered steps (1, 2, 3, …).  **There is no hard limit on
        step count** as long as the plan is concise and each step has a clear goal.

        You may decompose the user's query into sub-queries, each of which is a
        separate step.  Break the query into the smallest possible sub-queries
        so that each sub-query is answerable with a single data source.
        For example, if the user's query is "What were the key
        action items in the last quarter, and what was a recent news story for 
        each of them?", you may break it into steps:

        1. Fetch the key action items in the last quarter.
        2. Fetch a recent news story for the first action item.
        3. Fetch a recent news story for the second action item.
        4. Fetch a recent news story for the last action item

        Here is a list of available agents you can call upon to execute the tasks in your plan. You may call only one agent per step.

        {agent_list}

        Return **ONLY** valid JSON (no markdown, no explanations) in this form:

        {{
        "1": {{
            "agent": "{planner_agent_enum}",
            "action": "string",
        }},
        "2": {{ ... }},
        "3": {{ ... }}
        }}

        Guidelines:
        {agent_guidelines}
        """

    if replan_flag:
        prompt += f"""
        The current plan needs revision because: {replan_reason}

        Current plan:
        {json.dumps(prior_plan, indent=2)}

        When replanning:
        - Focus on UNBLOCKING the workflow rather than perfecting it.
        - Only modify steps that are truly preventing progress.
        - Prefer simpler, more achievable alternatives over complex rewrites.
        """

    else:
        prompt += "\nGenerate a new plan from scratch."

    prompt += f'\nUser query: "{user_query}"'
    
    return HumanMessage(content=prompt)

def executor_prompt(state: State) -> HumanMessage:
    """
    Build the single‑turn JSON prompt that drives the executor LLM.
    """
    step = int(state.get("current_step", 0))
    latest_plan: Dict[str, Any] = state.get("plan") or {}
    plan_block: Dict[str, Any] = latest_plan.get(str(step), {})
    attempts       = (state.get("replan_attempts", {}) or {}).get(step, 0)
    
    # Get agent guidelines dynamically
    executor_guidelines = format_agent_guidelines_for_executor(state)
    plan_agent = plan_block.get("agent", "web_researcher")

    messages_tail = (state.get("messages") or [])[-4:]

    valid_agents = set(get_agent_descriptions().keys())

    executor_prompt = f"""
        You are the **executor** in a multi‑agent system with these agents:
        `{ '`, `'.join(sorted(set([a for a in _get_enabled_agents(state) if a in valid_agents] + ['planner']))) }`.

        **Tasks**
        1. Decide if the current plan needs revision.  → `"replan_flag": true|false`
        2. Decide which agent to run next.             → `"goto": "<agent_name>"`
        3. Give one‑sentence justification.            → `"reason": "<text>"`
        4. Write the exact question that the chosen agent should answer
                                                    → "query": "<text>"

        **Guidelines**
        {executor_guidelines}
        - After **{MAX_REPLANS}** failed replans for the same step, move on.
        - If you *just replanned* (replan_flag is true) let the assigned agent try before
        requesting another replan.
        - After **{MAX_RETRAINS}** retrainings, move on.
        - If you *just retrained* (retrain_flag is true) let the assigned agent try before
        requesting another retrain.

        Respond **only** with valid JSON (no additional text):

        {{
        "replan": <true|false>,
        "goto": "<{ '|'.join([a for a in _get_enabled_agents(state) if a in valid_agents] + ['planner']) }>",
        "reason": "<1 sentence>",
        "query": "<text>"
        }}

        **PRIORITIZE FORWARD PROGRESS:** Only replan if the current step is completely blocked.
        1. If any reasonable data was obtained that addresses the step's core goal, set `"replan": false` and proceed.
        2. Set `"replan": true` **only if** ALL of these conditions are met:
        • The step has produced zero useful information
        • The missing information cannot be approximated or obtained by remaining steps
        • `{attempts} < {MAX_REPLANS}`
        3. When `{attempts} == {MAX_REPLANS}`, always move forward (`"replan": false`).

        ### Decide `"goto"`
        - If `"replan": true` → `"goto": "planner"`.
        - If current step has made reasonable progress → move to next step's agent.
        - Otherwise execute the current step's assigned agent (`{plan_agent}`).

        ### Build `"query"`
        Write a clear, standalone instruction for the chosen agent. If the chosen agent 
        is `web_researcher` or `cortex_researcher`, the query should be a standalone question, 
        written in plain english, and answerable by the agent.

        Ensure that the query uses consistent language as the user's query.

        Context you can rely on
        - User query ..............: {state.get("user_query")}
        - Current step index ......: {step}
        - Current plan step .......: {plan_block}
        - Just‑replanned flag .....: {state.get("replan_flag")}
        - Previous messages .......: {messages_tail}

        Respond **only** with JSON, no extra text.
        """

    return HumanMessage(
        content=executor_prompt
    )

def agent_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )