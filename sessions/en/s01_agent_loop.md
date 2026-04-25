# Section 01: The Agent Loop

> An agent is just `while True` + `finish_reason`.

## Architecture

```
    User Input
        |
        v
    messages[] <-- append {role: "user", ...}
        |
        v
    client.chat.completions.create(model, messages)
        |
        v
    finish_reason?
      /           \
  "stop"      "tool_calls"
     |              |
   Print        (Section 02)
     |
     v
    messages[] <-- append {role: "assistant", ...}
     |
     +--- loop back, wait for next input
```

Everything else -- tools, sessions, routing, delivery -- layers on top
without changing this loop.

## Key Concepts

- **messages[]** is the only state. The LLM sees the full array every call.
- **finish_reason** is the single decision point after each API response.
- **`"stop"`** = "print the text." **`"tool_calls"`** = "execute, feed result back" (Section 02).
- The loop structure never changes. Later sections add features around it.

## Key Code Walkthrough

### 1. The complete agent loop

Three steps per turn: collect input, call API, branch on finish_reason.

```python
def agent_loop() -> None:
    messages: list[dict] = []

    while True:
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                max_tokens=8096,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            )
        except Exception as exc:
            print(f"API Error: {exc}")
            messages.pop()   # roll back so user can retry
            continue

        choice = response.choices[0]
        if choice.finish_reason == "stop":
            assistant_text = choice.message.content or ""
            print_assistant(assistant_text)

            messages.append({
                "role": "assistant",
                "content": assistant_text,
            })
```

### 2. The finish_reason branch

Even in Section 01, the code stubs out `tool_calls`. No tools exist yet,
but the scaffolding means Section 02 requires zero changes to the outer loop.

```python
        elif choice.finish_reason == "tool_calls":
            print_info("[finish_reason=tool_calls] No tools in this section.")
            messages.append({"role": "assistant", "content": choice.message.content})
```

| finish_reason    | Meaning                      | Action             |
|------------------|------------------------------|--------------------||
| `"stop"`         | Model finished its reply     | Print, loop        |
| `"tool_calls"`   | Model wants to call a tool   | Execute, feed back |
| `"length"`       | Reply cut off by token limit | Print partial text |

## Try It

```sh
# Make sure .env has your key
echo 'OPENAI_API_KEY=your-key-here' > .env
echo 'MODEL_ID=gpt-4o' >> .env
echo 'OPENAI_BASE_URL=https://models.inference.ai.azure.com' >> .env

# Run the agent
python en/s01_agent_loop.py

# Talk to it -- multi-turn works because messages[] accumulates
# You > What is the capital of France?
# You > And what is its population?
# (The model remembers "France" from the previous turn.)
```

## How OpenClaw Does It

| Aspect         | claw0 (this file)              | OpenClaw production                   |
|----------------|--------------------------------|---------------------------------------|
| Loop location  | `agent_loop()` in one file     | `AgentLoop` class in `src/agent/`     |
| Messages       | Plain `list[dict]` in memory   | JSONL-persisted SessionStore          |
| finish_reason  | Same branching logic           | Same logic + streaming support        |
| Error handling | Pop last message, continue     | Retry with backoff + context guard    |
| System prompt  | Hardcoded string               | 8-layer dynamic assembly (Section 06) |
