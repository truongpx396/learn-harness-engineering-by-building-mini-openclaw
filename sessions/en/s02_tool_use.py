"""
Section 02: Tool Use
"Tools are data (schema dict) + a handler map. Model picks a name, you look it up."

The agent loop is unchanged from s01. The only additions:
  1. TOOLS array tells the model what tools exist (JSON schema)
  2. TOOL_HANDLERS dict maps tool names to Python functions
  3. When finish_reason == "tool_calls", dispatch and feed result back

    User --> LLM --> finish_reason == "tool_calls"?
                          |
                  TOOL_HANDLERS[name](**input)
                          |
                  tool result (role=tool) --> back to LLM
                          |
                   finish_reason == "stop"? --> Print

Tools:
    - bash        : Run shell commands
    - read_file   : Read file contents
    - write_file  : Write to a file
    - edit_file   : Exact string replacement in a file

Usage:
    cd claw0
    python en/s02_tool_use.py

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
import json
import subprocess
from pathlib import Path
from typing import Any

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

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "Use the tools to help the user with file operations and shell commands.\n"
    "Always read a file before editing it.\n"
    "When using edit_file, the old_string must match EXACTLY (including whitespace)."
)

MAX_TOOL_OUTPUT = 50000
WORKDIR = Path.cwd()

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def safe_path(raw: str) -> Path:
    """Resolve path, block traversal outside WORKDIR."""
    target = (WORKDIR / raw).resolve()
    if not str(target).startswith(str(WORKDIR)):
        raise ValueError(f"Path traversal blocked: {raw} resolves outside WORKDIR")
    return target


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_bash(command: str, timeout: int = 30) -> str:
    """Run a shell command and return its output."""
    dangerous = ["rm -rf /", "mkfs", "> /dev/sd", "dd if="]
    for pattern in dangerous:
        if pattern in command:
            return f"Error: Refused to run dangerous command containing '{pattern}'"

    print_tool("bash", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKDIR),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return truncate(output) if output else "[no output]"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"


def tool_read_file(file_path: str) -> str:
    """Read the contents of a file."""
    print_tool("read_file", file_path)
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error: Not a file: {file_path}"
        content = target.read_text(encoding="utf-8")
        return truncate(content)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    print_tool("write_file", file_path)
    try:
        target = safe_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} chars to {file_path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Exact string replacement. old_string must appear exactly once."""
    print_tool("edit_file", f"{file_path} (replace {len(old_string)} chars)")
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"

        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)

        if count == 0:
            return "Error: old_string not found in file. Make sure it matches exactly."
        if count > 1:
            return (
                f"Error: old_string found {count} times. "
                "It must be unique. Provide more surrounding context."
            )

        new_content = content.replace(old_string, new_string, 1)
        target.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {file_path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Tool schema + dispatch table
# ---------------------------------------------------------------------------
# TOOLS = tells the model what's available (JSON schema)
# TOOL_HANDLERS = tells our code what to call (name -> function)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command and return its output. "
                "Use for system commands, git, package managers, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds. Default 30.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory).",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file. Creates parent directories if needed. "
                "Overwrites existing content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory).",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact string in a file with a new string. "
                "The old_string must appear exactly once in the file. "
                "Always read the file first to get the exact text to replace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory).",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace. Must be unique.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
]

TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Look up handler by name, call it with the input kwargs."""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return handler(**tool_input)
    except TypeError as exc:
        return f"Error: Invalid arguments for {tool_name}: {exc}"
    except Exception as exc:
        return f"Error: {tool_name} failed: {exc}"


# ---------------------------------------------------------------------------
# Core: The Agent Loop (same while True as s01, plus tool dispatch)
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """Main agent loop -- REPL with tools."""

    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 02: Tool Use")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Workdir: {WORKDIR}")
    print_info(f"  Tools: {', '.join(TOOL_HANDLERS.keys())}")
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

        # Inner loop: model may chain multiple tool calls before stop
        while True:
            try:
                response = client.chat.completions.create(
                    model=MODEL_ID,
                    max_tokens=8096,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    tools=TOOLS,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                while messages and messages[-1]["role"] != "user":
                    messages.pop()
                if messages:
                    messages.pop()
                break

            choice = response.choices[0]
            finish_reason = choice.finish_reason
            message = choice.message

            if finish_reason == "stop":
                assistant_text = message.content or ""
                if assistant_text:
                    print_assistant(assistant_text)
                messages.append({
                    "role": "assistant",
                    "content": assistant_text,
                })
                break

            elif finish_reason == "tool_calls":
                # Add assistant message with tool_calls
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })

                # Dispatch each tool call and append result messages
                for tc in message.tool_calls:
                    tool_input = json.loads(tc.function.arguments)
                    result = process_tool_call(tc.function.name, tool_input)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            else:
                print_info(f"[finish_reason={finish_reason}]")
                assistant_text = message.content or ""
                if assistant_text:
                    print_assistant(assistant_text)
                messages.append({
                    "role": "assistant",
                    "content": assistant_text,
                })
                break


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
