"""
Section 01: The Agent Loop
"An agent is just while True + stop_reason"

    User Input --> [messages[]] --> LLM API --> finish_reason?
                                                /        \
                                            "stop"  "tool_calls"
                                              |           |
                                           Print      (next section)

Usage:
    cd claw0
    python en/s01_agent_loop.py

Required .env config:
    OPENAI_API_KEY=github_pat_xxxxx
    MODEL_ID=gpt-4o
    OPENAI_BASE_URL=https://models.inference.ai.azure.com
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "gpt-4o")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


# ---------------------------------------------------------------------------
# Core: The Agent Loop
# ---------------------------------------------------------------------------
# 1. Collect user input, append to messages
# 2. Call the API
# 3. Check stop_reason -- "end_turn" means print, "tool_use" means dispatch
#
# Here stop_reason is always "end_turn" (no tools yet).
# Next section adds tools; the loop structure stays the same.
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """Main agent loop -- conversational REPL."""

    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 01: The Agent Loop")
    print_info(f"  Model: {MODEL_ID}")
    print_info("  Type 'quit' or 'exit' to leave. Ctrl+C also works.")
    print_info("=" * 60)
    print()

    while True:
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}Goodbye.{RESET}")
            break

        messages.append({
            "role": "user",
            "content": user_input,
        })

        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                max_tokens=8096,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            )
        except Exception as exc:
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            messages.pop()
            continue

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        assistant_text = choice.message.content or ""

        # Check finish_reason to decide what happens next
        if finish_reason == "stop":
            print_assistant(assistant_text)
            messages.append({
                "role": "assistant",
                "content": assistant_text,
            })

        elif finish_reason == "tool_calls":
            print_info("[finish_reason=tool_calls] No tools in this section.")
            print_info("See s02_tool_use.py for tool support.")
            messages.append({
                "role": "assistant",
                "content": assistant_text,
            })

        else:
            print_info(f"[finish_reason={finish_reason}]")
            if assistant_text:
                print_assistant(assistant_text)
            messages.append({
                "role": "assistant",
                "content": assistant_text,
            })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{YELLOW}Error: OPENAI_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)

    agent_loop()


if __name__ == "__main__":
    main()
