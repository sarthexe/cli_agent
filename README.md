# CLI Agent

A terminal-based AI coding agent with a cost-aware multi-model architecture.

## Quick Start

1. Install dependencies:
   - `uv sync`
2. Set your OpenAI key:
   - `export OPENAI_API_KEY=your_key_here`
3. Run:
   - `python main.py`
   - or `python main.py "hello"`

## Day 1 Scope

- Click CLI entry point
- Config loading from `~/.cli-agent/config.toml`
- OpenAI provider integration
- Basic REPL (`input -> GPT-4.1-mini -> print`)
