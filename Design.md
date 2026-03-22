# Autonomous CLI Agent — Design Document

## 1. What This Is

A terminal-resident AI agent that operates in a **think → plan → act → observe → retry** loop. It reads your codebase, runs shell commands, writes and edits files, and self-corrects when things break — all from a single terminal session. No GUI, no browser, no hand-holding.

---

## 2. Core Loop (The Brain)

The entire agent is one recursive control loop:

```
┌─────────────────────────────────────────────────────┐
│                   USER PROMPT                       │
└──────────────────────┬──────────────────────────────┘
                       ▼
              ┌────────────────┐
              │     THINK      │  LLM decides intent + plan
              │  (reasoning)   │  "I need to read X, then fix Y"
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │      PLAN      │  Emit a structured action
              │  (tool call)   │  {tool: "shell", cmd: "pytest"}
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │      ACT       │  Execute the action in sandbox
              │  (execution)   │  Run command, write file, etc.
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │    OBSERVE     │  Capture stdout/stderr/exit code
              │  (feedback)    │  Feed result back into context
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │   EVALUATE     │  Did it work? Goal met?
              │  (check)       │  If no → back to THINK
              └───────┬────────┘
                      ▼
                 ┌──────────┐
                 │   DONE   │  Present result to user
                 └──────────┘
```

**Key constraint:** Cap the loop at N iterations (default: 10). If the agent can't solve it in N tries, surface what it tried and ask the user. Infinite loops are the #1 failure mode of naive agents.

---

## 3. Architecture

```
cli-agent/
├── main.py                  # Entry point, REPL / single-shot mode
├── agent/
│   ├── core.py              # The main agent loop (think-plan-act-observe)
│   ├── planner.py           # Prompt construction + LLM call
│   ├── context.py           # Context window manager (token budget)
│   └── memory.py            # Conversation history + summarization
├── tools/
│   ├── registry.py          # Tool registry (discover, validate, dispatch)
│   ├── shell.py             # Execute shell commands
│   ├── file_read.py         # Read files (with line ranges)
│   ├── file_write.py        # Write / create files
│   ├── file_edit.py         # Surgical str_replace style edits
│   ├── glob_search.py       # Find files by pattern
│   └── grep_search.py       # Search content across files
├── sandbox/
│   ├── executor.py          # Subprocess runner with timeout + resource limits
│   └── permissions.py       # Allowlist / blocklist for dangerous commands
├── llm/
│   ├── provider.py          # Abstract LLM interface
│   ├── ollama_provider.py   # Ollama (local — GLM4, Qwen, Llama, etc.)
│   ├── anthropic_provider.py# Claude API
│   └── openai_provider.py   # OpenAI-compatible API (optional)
├── config/
│   ├── settings.py          # Pydantic settings (env vars, defaults)
│   └── default.toml         # Default config file
└── utils/
    ├── tokens.py            # Token counting (tiktoken / approximate)
    ├── diff.py              # Unified diff generation for edits
    └── logger.py            # Structured logging (JSON lines)
```

---

## 4. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **Language** | Python 3.11+ | You already live here. Async support, rich ecosystem. |
| **LLM (local)** | Ollama — **Qwen 3.5 4B** (`qwen3.5:4b`) | 3.4GB Q4_K_M. Native tool calling + thinking support in Ollama. 256K context window. Apache 2.0. |
| **LLM (remote)** | Claude Sonnet via API | Fallback for complex multi-step tasks where 4B struggles. |
| **CLI framework** | `click` + `rich` | Click for arg parsing, Rich for pretty terminal output (spinners, syntax highlighting, panels). |
| **Config** | Pydantic Settings + TOML | Type-safe config, env var overrides, sensible defaults. |
| **Subprocess** | `asyncio.create_subprocess_exec` | Non-blocking command execution with timeout control. |
| **Token counting** | `tiktoken` (or character heuristic for local models) | Manage context window budget. |
| **Logging** | `structlog` | JSON-lines structured logs. Debug an agent's decisions after the fact. |

**No database. No server. No Docker.** It's a CLI tool that runs in your terminal. State lives in memory for the session and optionally dumps to a `.jsonl` log file.

---

## 5. Tool System Design

Every tool is a Python class that implements one interface:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolResult:
    output: str          # What the LLM sees
    success: bool        # Did the tool succeed?
    metadata: dict       # Extra info (exit_code, file_path, etc.)

class BaseTool:
    name: str            # "shell", "file_read", etc.
    description: str     # For the LLM's system prompt
    parameters: dict     # JSON Schema for the tool's arguments

    def validate(self, args: dict) -> bool:
        """Validate args before execution."""
        ...

    async def execute(self, args: dict) -> ToolResult:
        """Run the tool, return structured result."""
        ...
```

### 5.1 Tool Definitions

**shell** — Execute a command.
```
args: {command: str, timeout_seconds: int = 30, working_dir: str = "."}
returns: {stdout, stderr, exit_code}
```
- Runs via `asyncio.create_subprocess_exec` inside a subprocess.
- Hard timeout (kill after N seconds).
- Blocked commands: `rm -rf /`, `:(){ :|:& };:`, `mkfs`, `dd if=/dev/zero`, etc.
- Captures combined stdout+stderr up to 50KB. Truncates with `[...truncated, showing last 200 lines]`.

**file_read** — Read a file or directory listing.
```
args: {path: str, line_start: int = None, line_end: int = None}
returns: {content (with line numbers), total_lines, file_size}
```
- Returns numbered lines (so the LLM can reference them in edits).
- Caps at ~500 lines per read. If file is larger, force the LLM to use line ranges.

**file_write** — Create or overwrite a file.
```
args: {path: str, content: str, create_dirs: bool = True}
returns: {path, bytes_written}
```

**file_edit** — Surgical find-and-replace (like `str_replace`).
```
args: {path: str, old_str: str, new_str: str}
returns: {path, diff}
```
- `old_str` must appear exactly once in the file. If ambiguous, fail and tell the LLM.
- Returns a unified diff so the LLM can verify the change.

**glob_search** — Find files by pattern.
```
args: {pattern: str, root: str = "."}
returns: {matches: list[str]}
```
- Respects `.gitignore` by default. Skips `node_modules`, `.git`, `__pycache__`, `venv`.

**grep_search** — Search content across files.
```
args: {query: str, path: str = ".", include: str = None}
returns: {matches: list[{file, line_number, content}]}
```

### 5.2 Tool Registry

```python
class ToolRegistry:
    _tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool): ...
    def get(self, name: str) -> BaseTool: ...
    def schema_for_llm(self) -> list[dict]:
        """Return all tool schemas in OpenAI/Anthropic function-calling format."""
        ...
```

The registry auto-generates the tools section of the system prompt. When you add a new tool, the agent automatically knows about it.

---

## 6. Context Window Management

This is where most DIY agents fail. You will hit the context limit. Plan for it.

```
┌──────────────────── CONTEXT BUDGET ────────────────────┐
│                                                        │
│  [SYSTEM PROMPT]           ~1500 tokens (fixed)        │
│  [TOOL DEFINITIONS]        ~800 tokens (fixed)         │
│  [PROJECT CONTEXT]         ~2000 tokens (dynamic)      │
│  [CONVERSATION HISTORY]    remaining budget (sliding)   │
│  [CURRENT OBSERVATION]     last tool result             │
│                                                        │
│  TOTAL BUDGET: model_max_tokens - max_output_tokens    │
└────────────────────────────────────────────────────────┘
```

### Strategy: Sliding Window with Summarization

```python
class ContextManager:
    def __init__(self, max_tokens: int, reserve_output: int = 4096):
        self.budget = max_tokens - reserve_output
        self.system_prompt: str           # Fixed
        self.tool_schemas: str            # Fixed
        self.project_context: str         # Semi-fixed (tree, key files)
        self.history: list[Message]       # Sliding window
        self.summaries: list[str]         # Compressed old history

    def build_messages(self) -> list[dict]:
        """Assemble messages that fit within budget."""
        fixed_cost = count_tokens(self.system_prompt + self.tool_schemas + self.project_context)
        remaining = self.budget - fixed_cost

        # Take messages from the end (most recent first)
        # When history exceeds budget: summarize oldest N messages into one summary block
        ...
```

**Rules (tuned for 4B):**
1. Never send raw files over 200 lines. Always read with line ranges. (Tighter than larger models.)
2. Truncate tool output to 20KB. Last 100 lines for errors, first 50 for file reads.
3. When history grows past 50% of budget, summarize the oldest third into a single "so far" block.
4. Always keep the last 2 user messages + last 2 tool results intact (no summarization).
5. Effective context budget: target 8–16K tokens even though 256K is available. Quality > quantity at 4B.

---

## 7. LLM Provider Abstraction

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str | None              # Text response
    tool_calls: list[ToolCall] | None # Structured tool calls
    usage: dict                       # {prompt_tokens, completion_tokens}
    stop_reason: str                  # "end_turn", "tool_use", etc.

class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        ...

class OllamaProvider(LLMProvider):
    """
    POST to http://localhost:11434/api/chat

    Qwen 3.5 4B specifics:
    - Thinking mode is OFF by default for small models. Enable it:
      Option A: In the API call, set "think": true in options
      Option B: Create a Modelfile with PARAMETER think true
    - Tool calling format: Ollama native (compatible with OpenAI function-calling schema)
    - Force ONE tool call per turn (set "parallel_tool_calls": false or parse only the first)
    - At 4B, expect ~15-20% malformed tool call rate. Always validate JSON before executing.
    - 256K context window, but keep effective usage under 8-16K for speed + coherence.
    """
    ...

class AnthropicProvider(LLMProvider):
    """POST to https://api.anthropic.com/v1/messages"""
    ...
```

**Model routing logic** (important at 4B — not optional):
- Start with Qwen 3.5 4B for every request.
- Track: did the model produce a valid tool call JSON? If it fails JSON validation 2x in a row, escalate to Claude.
- If the task requires >3 retries (actual execution failures, not parse failures), escalate to Claude.
- Log which model handled what for later analysis.

---

## 8. Qwen 3.5 4B — Design Constraints

Running a 4B model as an agent brain means working within tight limits. These aren't suggestions — ignore them and the agent will feel broken.

**1. Enable thinking mode.**
Small Qwen 3.5 models ship with thinking off. Without it, the model jumps straight to a tool call without reasoning about what it should do. For an agent, that's fatal — it'll read the wrong file, run the wrong command, fix the wrong thing. Enable it via the Ollama API (`"think": true`) or a custom Modelfile (`PARAMETER think true`).

**2. One tool call per turn. Always.**
Don't let the model emit multiple tool calls in a single response. Parse only the first tool_call from the response. Multi-tool planning requires working memory that 4B doesn't reliably have.

**3. Max 6 tools.**
The 6 tools in this design (shell, file_read, file_write, file_edit, glob_search, grep_search) are the ceiling. If you need more capability, compose it — e.g., don't add a "run_tests" tool, just let the agent call `shell` with `pytest`. Every tool in the schema is a decision the model has to make, and at 4B, decision quality degrades fast past 6 options.

**4. System prompt under 800 tokens.**
4B models lose instruction-following quality as the system prompt grows. Cut every word that isn't a hard rule. No examples in the system prompt — if the model needs examples, put them in a few-shot format in the first user message instead.

**5. Tool output truncation is mandatory.**
At 4B, long context ≠ good context. Even though Qwen 3.5 supports 256K, keep effective context under 8–16K tokens. Truncate tool output aggressively (last 100 lines for errors, first 50 lines for file reads). The model's attention degrades on long inputs.

**6. Validate tool call JSON before execution.**
Budget for ~15–20% malformed tool calls. The validation layer should:
- Check that `tool_name` exists in the registry.
- Check that all required args are present and typed correctly.
- If invalid: append a short error ("Invalid tool call: missing 'path' argument") to context and re-prompt. Don't burn a retry cycle on this — it's a parse error, not a logic error.

**7. Keep conversation history short.**
Summarize aggressively. At 4B, the model starts contradicting its own earlier reasoning after ~6–8 turns. The summarize threshold in the context manager should be set to 50% (not 60% like larger models).

---

## 8. The System Prompt

This is 60% of your agent's quality. For a 4B model, **keep it under 800 tokens**. No fluff.

```
You are a CLI coding agent. You use tools to read files, edit code, and run commands.

RULES:
1. Read a file before editing it. Never guess contents.
2. Run code after changes to verify.
3. On error: read the error, fix root cause, re-run.
4. Use grep/glob to find files before assuming paths.
5. Never run destructive commands (rm -rf, drop, etc.) without asking.
6. One tool call per response. No parallel calls.
7. After 3 failed fixes, stop and ask the user.

WORKFLOW: Understand → Search → Read → Edit → Verify → Report.

TOOLS:
{auto-generated from tool registry}
```

That's it. Resist the urge to add more. Every extra sentence in the system prompt is a sentence the 4B model might forget mid-task.

---

## 9. Error Recovery (The Self-Fix Loop)

This is where "autonomous" actually matters:

```python
async def act_and_recover(self, action: ToolCall, max_retries: int = 3) -> ToolResult:
    for attempt in range(max_retries):
        result = await self.execute_tool(action)

        if result.success:
            return result

        # Feed failure back into context
        self.context.add_observation(
            f"[ATTEMPT {attempt + 1}/{max_retries}] Tool '{action.name}' failed:\n"
            f"{result.output}"
        )

        # Ask the LLM to diagnose and produce a new action
        recovery = await self.planner.get_next_action(
            self.context.build_messages()
        )

        if recovery.content and not recovery.tool_calls:
            # LLM gave up and wrote an explanation — surface to user
            return ToolResult(output=recovery.content, success=False, metadata={})

        action = recovery.tool_calls[0]

    return ToolResult(
        output=f"Failed after {max_retries} attempts. Last error:\n{result.output}",
        success=False,
        metadata={"exhausted_retries": True}
    )
```

**What makes this work:** The error output (stderr, traceback, exit code) goes directly into the LLM context as an observation. The LLM sees what went wrong and can course-correct — different fix, read more context, try a different approach entirely.

---

## 10. Safety & Sandboxing

| Risk | Mitigation |
|---|---|
| `rm -rf /` or destructive commands | Blocklist regex on commands. Require user confirmation for anything matching `rm`, `drop`, `truncate`, `kill`, `mkfs`. |
| Infinite loops | Hard cap on agent iterations (default 10). Hard timeout on subprocesses (default 30s). |
| Token blowout | Context manager enforces budget. Large outputs truncated. |
| Writes to wrong files | Log all file writes with full diffs. Support `--dry-run` mode that shows what would change without writing. |
| Runaway resource usage | `ulimit` on subprocess (max memory, max file size, max CPU time). |
| Prompt injection from file contents | Keep user instructions in system prompt. Treat file contents as untrusted data in user messages. |

**Confirmation mode** (default for sensitive ops):
```
Agent wants to run: rm -rf ./build/
[y/n/edit]>
```

---

## 11. Config Schema

```toml
# ~/.cli-agent/config.toml

[llm]
default_provider = "ollama"          # "ollama" | "anthropic" | "openai"
fallback_provider = "anthropic"      # Escalate on failure

[llm.ollama]
model = "qwen3.5:4b"
base_url = "http://localhost:11434"
temperature = 0.0
context_window = 262144              # 256K native, but keep effective use under 16K
enable_thinking = true               # OFF by default for small models — must enable
parallel_tool_calls = false          # Force sequential — critical for 4B reliability

[llm.anthropic]
model = "claude-sonnet-4-20250514"
# api_key via ANTHROPIC_API_KEY env var
temperature = 0.0
context_window = 200000

[agent]
max_iterations = 10                  # Max think-act cycles per request
max_retries_per_tool = 3             # Retries on tool failure
confirmation_required = ["rm", "drop", "kill", "truncate"]

[sandbox]
command_timeout_seconds = 30
max_output_bytes = 20480             # 20KB — tighter for 4B attention limits
blocked_commands = ["rm -rf /", "mkfs", "dd if=/dev/zero"]

[context]
max_file_lines = 200                 # Force line ranges above this (tighter for 4B)
summarize_threshold = 0.5            # Summarize when history hits 50% of budget
keep_recent_messages = 4             # Never summarize the last N messages
effective_context_target = 16384     # Aim for 16K tokens even though 256K available
```

---

## 12. Build Phases

### Phase 1 — Skeleton (Day 1-2)
- [ ] Project scaffolding, config loading, CLI entry point (`click`)
- [ ] LLM provider abstraction + Ollama provider (just `POST /api/chat`)
- [ ] Single tool: `shell` (execute commands, capture output)
- [ ] Basic agent loop: prompt → LLM → tool call → execute → show result
- [ ] **Milestone:** `agent "list all python files"` works

### Phase 2 — File Operations (Day 3-4)
- [ ] `file_read`, `file_write`, `file_edit` tools
- [ ] `glob_search`, `grep_search` tools
- [ ] Tool registry with auto-schema generation
- [ ] **Milestone:** `agent "read main.py and add error handling to the DB connection"` works

### Phase 3 — Context & Memory (Day 5-6)
- [ ] Token counting
- [ ] Context window manager with sliding window + summarization
- [ ] Multi-turn REPL mode (persistent session)
- [ ] **Milestone:** Agent can handle a 20-message conversation without blowing context

### Phase 4 — Self-Repair (Day 7-8)
- [ ] Error recovery loop (retry with feedback)
- [ ] Auto-run after edits (detect test runner, run it)
- [ ] Diff-based edit verification
- [ ] **Milestone:** `agent "fix the failing tests in tests/"` actually fixes them

### Phase 5 — Safety & Polish (Day 9-10)
- [ ] Command blocklist + confirmation prompts
- [ ] `--dry-run` mode
- [ ] Anthropic provider as fallback
- [ ] Structured JSON logging
- [ ] Rich terminal output (syntax-highlighted code, diff panels, spinners)
- [ ] **Milestone:** Ship it, use it daily

---

## 13. Critical Design Decisions

**1. Tool calls, not free-form parsing.**
Don't regex-parse the LLM's prose for commands. Use the model's native tool-calling / function-calling format. Ollama supports this for Qwen and Llama models. It returns structured JSON. Parse that. This alone eliminates 80% of "agent did something weird" bugs.

**2. Edit by replacement, not by rewrite.**
The `file_edit` tool does `str_replace` (find exact string → replace with new string), NOT "rewrite the whole file." This prevents the LLM from accidentally dropping code, losing imports, or hallucinating functions that don't exist.

**3. Observe before act.**
Hard-code into the system prompt: "ALWAYS read before edit." The #1 agent failure is editing a file the LLM has never seen, based on its assumption of what the file contains.

**4. Local-first, cloud-fallback.**
Start every request on Qwen 3.5 4B via Ollama. Escalate to Claude when the local model demonstrably fails — specifically: 2 consecutive malformed tool calls, or 3 execution failures on the same task. This keeps it fast, private, and cheap. At 4B, expect to escalate ~20–30% of complex multi-file tasks. That's fine — the goal is local for the 70% of tasks that are straightforward.

**5. Log everything.**
Every LLM call, every tool execution, every retry — write it to a `.jsonl` file. When the agent does something stupid (and it will), you need the full trace to debug the prompt, not guess at what happened.
