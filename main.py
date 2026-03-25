"""
main.py — CLI Agent entry point.

Two modes:
  - Single-shot: `cli-agent "do something"` — runs once and exits
  - REPL:        `cli-agent` with no args — enters interactive loop
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from config.settings import Settings, load_settings
from llm.gemini_provider import GeminiProvider
from utils.logger import get_logger, setup_logging

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Rich console with custom theme
# ---------------------------------------------------------------------------

THEME = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "model.gemini": "bold blue",
    "model.deepseek": "bold magenta",
    "model.claude": "bold yellow",
    "cost": "dim green",
    "prompt": "bold cyan",
})

console = Console(theme=THEME)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def show_banner(settings: Settings) -> None:
    """Display the welcome banner with project info."""
    banner = Text()
    banner.append("CLI Agent", style="bold cyan")
    banner.append(f"  v{__version__}\n", style="dim")
    banner.append("Autonomous coding agent with intelligent multi-model routing\n\n", style="dim")
    banner.append("Models: ", style="bold")
    banner.append(f"[T1] {settings.llm.gemini.model}", style="model.gemini")
    banner.append(" → ", style="dim")
    banner.append(f"[T2] {settings.llm.deepseek.model}", style="model.deepseek")
    banner.append(" → ", style="dim")
    banner.append(f"[T3] {settings.llm.anthropic.model}", style="model.claude")
    banner.append(f"\nBudget: ", style="bold")
    banner.append(f"${settings.cost.session_budget:.2f}/session", style="cost")
    banner.append(f"  Max iterations: {settings.agent.max_iterations}", style="dim")

    console.print(Panel(
        banner,
        border_style="cyan",
        title="[bold]🤖 CLI Agent[/bold]",
        subtitle="[dim]Type /help for commands, /exit to quit[/dim]",
        padding=(1, 2),
    ))


def build_gemini_provider(settings: Settings, model_override: str | None = None) -> GeminiProvider:
    """Create a Gemini provider instance from resolved settings."""
    gemini = settings.llm.gemini
    return GeminiProvider(
        model=model_override or gemini.model,
        api_key=gemini.api_key,
        base_url=gemini.base_url,
        temperature=gemini.temperature,
    )


def render_model_reply(provider: GeminiProvider, reply_text: str) -> None:
    """Render a model reply with a provider label."""
    label = f"[{provider.model}]"
    console.print(f"[model.gemini]{label}[/model.gemini] {reply_text or '[dim](empty response)[/dim]'}")


# ---------------------------------------------------------------------------
# REPL Commands
# ---------------------------------------------------------------------------

REPL_COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/exit": "Exit the agent",
    "/quit": "Exit the agent",
    "/cost": "Show current session cost",
    "/version": "Show version info",
    "/memory": "View/edit project memory",
    "/rewind": "Rewind to a previous step (e.g., /rewind 3)",
    "/explain": "Preview plan without executing",
}


def handle_repl_command(command: str, settings: Settings) -> bool:
    """
    Handle a REPL slash-command.

    Returns True if the REPL should continue, False if it should exit.
    """
    cmd = command.strip().lower().split()[0]
    args = command.strip().split()[1:]

    if cmd in ("/exit", "/quit"):
        console.print("\n[dim]Session ended.[/dim]")
        # TODO: Show session cost summary here
        return False

    elif cmd == "/help":
        console.print("\n[bold]Available commands:[/bold]")
        for name, desc in REPL_COMMANDS.items():
            console.print(f"  [prompt]{name:<12}[/prompt] {desc}")
        console.print()

    elif cmd == "/version":
        console.print(f"\n[info]CLI Agent v{__version__}[/info]")
        console.print(f"  Default model: {settings.llm.gemini.model}")
        console.print(f"  Python: {sys.version.split()[0]}\n")

    elif cmd == "/cost":
        # TODO: Integrate with CostTracker
        console.print("\n[cost]Session cost: $0.0000[/cost]")
        console.print("[dim]  Calls: gemini: 0, deepseek: 0, claude: 0[/dim]\n")

    elif cmd == "/memory":
        # TODO: Integrate with ProjectMemory
        console.print("\n[warning]Project memory not yet loaded.[/warning]\n")

    elif cmd == "/rewind":
        step = int(args[0]) if args else None
        if step is None:
            console.print("[error]Usage: /rewind <step_number>[/error]")
        else:
            # TODO: Integrate with SnapshotManager
            console.print(f"\n[warning]Rewind to step {step} — not yet implemented.[/warning]\n")

    elif cmd == "/explain":
        # TODO: Integrate with ExplainMode
        console.print("\n[warning]Explain mode — not yet implemented.[/warning]\n")

    else:
        console.print(f"[error]Unknown command: {cmd}[/error]")
        console.print("[dim]Type /help for available commands.[/dim]")

    return True


# ---------------------------------------------------------------------------
# REPL Loop
# ---------------------------------------------------------------------------

def run_repl(settings: Settings, provider: GeminiProvider) -> None:
    """Run the interactive REPL session."""
    log = get_logger("repl")
    show_banner(settings)

    while True:
        try:
            user_input = console.input("[prompt]❯ [/prompt]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            should_continue = handle_repl_command(user_input, settings)
            if not should_continue:
                break
            continue

        # Day 1 skeleton behavior: prompt -> Gemini -> print.
        log.info("User prompt received", prompt=user_input)
        try:
            with console.status(f"[model.gemini]Calling {provider.model}...[/model.gemini]"):
                response = provider.complete(user_input)
        except Exception as e:
            log.error("Gemini call failed", error=str(e))
            console.print(f"[error]Gemini request failed: {e}[/error]\n")
            continue

        render_model_reply(provider, response.text)
        if response.tool_calls:
            console.print(f"[dim]Tool calls suggested: {len(response.tool_calls)}[/dim]")
        console.print()


# ---------------------------------------------------------------------------
# Single-shot mode
# ---------------------------------------------------------------------------

def run_single_shot(prompt: str, provider: GeminiProvider) -> None:
    """Run the agent once on a single prompt, then exit."""
    log = get_logger("single_shot")
    log.info("Single-shot mode", prompt=prompt)

    console.print(f"\n[dim]Running:[/dim] {prompt}")
    try:
        with console.status(f"[model.gemini]Calling {provider.model}...[/model.gemini]"):
            response = provider.complete(prompt)
    except Exception as e:
        log.error("Gemini call failed", error=str(e))
        console.print(f"[error]Gemini request failed: {e}[/error]\n")
        raise SystemExit(1)

    render_model_reply(provider, response.text)
    if response.tool_calls:
        console.print(f"[dim]Tool calls suggested: {len(response.tool_calls)}[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("prompt", required=False, default=None)
@click.option(
    "--model", "-m",
    "model_override",
    default=None,
    help="Override Gemini model for this run (e.g. gemini-2.0-flash).",
)
@click.option(
    "--config", "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to custom config.toml file.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
@click.option(
    "--version",
    is_flag=True,
    default=False,
    help="Show version and exit.",
)
def cli(
    prompt: str | None,
    model_override: str | None,
    config_path: str | None,
    verbose: bool,
    version: bool,
) -> None:
    """
    🤖 CLI Agent — Autonomous coding agent with intelligent multi-model routing.

    Run with a PROMPT for single-shot mode, or with no arguments for interactive REPL.

    \b
    Examples:
      cli-agent "list all python files"
      cli-agent "fix the failing tests"
      cli-agent                          # Enter REPL mode
    """
    if version:
        console.print(f"CLI Agent v{__version__}")
        raise SystemExit(0)

    # Load configuration
    try:
        settings = load_settings(config_path)
    except Exception as e:
        console.print(f"[error]Failed to load config: {e}[/error]")
        raise SystemExit(1)

    # Initialize logging
    setup_logging(verbose=verbose)
    log = get_logger("main")
    log.debug("Config loaded", provider=settings.llm.default_provider)
    try:
        provider = build_gemini_provider(settings, model_override=model_override)
    except ValueError as e:
        console.print(f"[error]{e}[/error]")
        raise SystemExit(1)

    # Route to single-shot or REPL mode
    if prompt:
        run_single_shot(prompt, provider)
    else:
        run_repl(settings, provider)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
