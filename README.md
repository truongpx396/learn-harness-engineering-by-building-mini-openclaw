
# 🤖 Learn Harness Engineering by Building a Mini Openclaw

##  Table of Contents

- [🔀 Origin & Modifications](#-origin--modifications)
- [🤔 What is this?](#-what-is-this)
- [🏗️ Architecture](#️-architecture)
- [🔗 Section Dependencies](#-section-dependencies)
- [⚡ Quick Start](#-quick-start)
- [🗺️ Learning Path](#️-learning-path)
- [📋 Section Details](#-section-details)
- [📁 Repository Structure](#-repository-structure)
- [📦 Prerequisites](#-prerequisites)
- [🧩 Dependencies](#-dependencies)
- [🔗 Related Projects](#-related-projects)
- [👥 About](#-about)
- [📄 License](#-license)

---

## 🔀 Origin & Modifications

This repository is a fork of [shareAI-lab/claw0](https://github.com/shareAI-lab/claw0)

Changes made in this fork:
- 🔄 **SDK migration**: Migrated from the Anthropic SDK to the [OpenAI SDK](https://github.com/openai/openai-python), making all sections compatible with any OpenAI-compatible endpoint.
- 🖥️ **Local model support**: Added setup guides for running fully offline with [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com), and [GPT4All](https://www.nomic.ai/gpt4all) — no cloud API required.
- ⚙️ **`.env`-based configuration**: Introduced `OPENAI_BASE_URL` and `MODEL_ID` environment variables so you can point any section at a different provider or local server without touching the code.

All credit for the original curriculum, architecture, and teaching approach goes to [shareAI-lab](https://github.com/shareAI-lab).

---

🚀 **From Zero to One: Build an AI Agent Gateway**

> 10 progressive sections -- every section is a single, runnable Python file.
> code + docs co-located.

---

## 🤔 What is this?

Most agent tutorials stop at "call an API once." This repository starts from that while loop and takes you all the way to a production-grade gateway.

Build a minimal AI agent gateway from scratch, section by section. 10 sections, 10 core concepts, ~7,000 lines of Python. Each section introduces exactly one new idea while keeping all prior code intact. After all 10, you can read OpenClaw's production codebase with confidence.

```sh
s01: Agent Loop           -- The foundation: while + finish_reason
s02: Tool Use             -- Let the model call tools: dispatch table
s03: Sessions & Context   -- Persist conversations, handle overflow
s04: Channels             -- Telegram + Feishu: real channel pipelines
s05: Gateway & Routing    -- 5-tier binding, session isolation
s06: Intelligence         -- Soul, memory, skills, prompt assembly
s07: Heartbeat & Cron     -- Proactive agent + scheduled tasks
s08: Delivery             -- Reliable message queue with backoff
s09: Resilience           -- 3-layer retry onion + auth profile rotation
s10: Concurrency          -- Named lanes serialize the chaos
```

## 🏗️ Architecture

```
+--- agent layers ---+
|                                                     |
|  s10: Concurrency  (named lanes, generation track)  |
|  s09: Resilience   (auth rotation, overflow compact)|
|  s08: Delivery     (write-ahead queue, backoff)     |
|  s07: Heartbeat    (lane lock, cron scheduler)      |
|  s06: Intelligence (8-layer prompt, hybrid memory)  |
|  s05: Gateway      (WebSocket, 5-tier routing)      |
|  s04: Channels     (Telegram pipeline, Feishu hook) |
|  s03: Sessions     (JSONL persistence, 3-stage retry)|
|  s02: Tools        (dispatch table, 4 tools)        |
|  s01: Agent Loop   (while True + finish_reason)     |
|                                                     |
+-----------------------------------------------------+
```

## 🔗 Section Dependencies

```
s01 --> s02 --> s03 --> s04 --> s05
                 |               |
                 v               v
                s06 ----------> s07 --> s08
                 |               |
                 v               v
                s09 ----------> s10
```

- s01-s02: Foundation (no dependencies)
- s03: Builds on s02 (adds persistence to the tool loop)
- s04: Builds on s03 (channels produce InboundMessages for sessions)
- s05: Builds on s04 (routes channel messages to agents)
- s06: Builds on s03 (uses sessions for context, adds prompt layers)
- s07: Builds on s06 (heartbeat uses soul/memory for prompt)
- s08: Builds on s07 (heartbeat output flows through delivery queue)
- s09: Builds on s03+s06 (reuses ContextGuard for overflow, model config)
- s10: Builds on s07 (replaces single Lock with named lane system)

## ⚡ Quick Start

```sh
# 1. Clone and enter
git clone https://github.com/truongpx396/learn-harness-engineering-by-building-mini-openclaw && cd learn-harness-engineering-by-building-a-mini-openclaw

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, MODEL_ID, and OPENAI_BASE_URL

# 4. Run any section (pick your language)
python sessions/en/s01_agent_loop.py    # English
```

## 🗺️ Learning Path

Each section adds exactly one new concept. All prior code stays intact:

```
Phase 1: FOUNDATION     Phase 2: CONNECTIVITY     Phase 3: BRAIN        Phase 4: AUTONOMY       Phase 5: PRODUCTION
+----------------+      +-------------------+     +-----------------+   +-----------------+   +-----------------+
| s01: Loop      |      | s03: Sessions     |     | s06: Intelligence|  | s07: Heartbeat  |   | s09: Resilience |
| s02: Tools     | ---> | s04: Channels     | --> |   soul, memory, | ->|   & Cron        |-->|   & Concurrency |
|                |      | s05: Gateway      |     |   skills, prompt |  | s08: Delivery   |   | s10: Lanes      |
+----------------+      +-------------------+     +-----------------+   +-----------------+   +-----------------+
 while + dispatch        persist + route            personality + recall  proactive + reliable  retry + serialize
```

## 📋 Section Details

| # | Section | Core Concept | Lines |
|---|---------|-------------|-------|
| 01 | Agent Loop | `while True` + `finish_reason` -- that's an agent | ~175 |
| 02 | Tool Use | Tools = schema dict + handler map. Model picks a name, you look it up | ~445 |
| 03 | Sessions | JSONL: append on write, replay on read. Too big? Summarize old parts | ~890 |
| 04 | Channels | Every platform differs, but they all produce the same `InboundMessage` | ~780 |
| 05 | Gateway | Binding table maps (channel, peer) to agent. Most specific wins | ~625 |
| 06 | Intelligence | System prompt = files on disk. Swap files, change personality | ~750 |
| 07 | Heartbeat & Cron | Timer thread: "should I run?" + queue work alongside user messages | ~660 |
| 08 | Delivery | Write to disk first, then send. Crashes can't lose messages | ~870 |
| 09 | Resilience | 3-layer retry onion: auth rotation, overflow compaction, tool-use loop | ~1130 |
| 10 | Concurrency | Named lanes with FIFO queues, generation tracking, Future-based results | ~900 |

## 📁 Repository Structure

```
learn-harness-engineering-by-building-a-mini-openclaw/
  README.md              English README
  .env.example           Configuration template
  requirements.txt       Python dependencies
  sessions/              All teaching sessions (code + docs)
    en/                  English
      s01_agent_loop.py  s01_agent_loop.md
      s02_tool_use.py    s02_tool_use.md
      ...                (10 .py + 10 .md)
  workspace/             Shared workspace samples
    SOUL.md  IDENTITY.md  TOOLS.md  USER.md
    HEARTBEAT.md  BOOTSTRAP.md  AGENTS.md  MEMORY.md
    CRON.json
    skills/example-skill/SKILL.md
```


## 📦 Prerequisites

- Python 3.11+
- An OpenAI-compatible API key (e.g. GitHub Models, Azure OpenAI, or any provider)

### 💻 Running Locally (no cloud API required)

All agents speak the OpenAI chat-completions protocol. Any local server that exposes a compatible endpoint works out of the box — no GPU required, CPU-only inference is supported by all three options below.

---

#### Option A — 🖥️ LM Studio

[LM Studio](https://lmstudio.ai) provides a GUI for downloading and serving models.

**1. Install & load a model**
- Download and install [LM Studio](https://lmstudio.ai).
- In the **Discover** tab, search for a small instruction-tuned model.
  Good CPU-friendly choices: `Qwen2.5-7B-Instruct`, `Mistral-7B-Instruct`, `Phi-3-mini`.
- Click **Download** next to your chosen model.

**2. Start the local server**
- Open the **Developer** tab (`</>` icon in the left sidebar).
- Select your model from the dropdown and click **Start Server**.
- LM Studio listens at `http://localhost:1234/v1`. Copy the model identifier shown (e.g. `lmstudio-community/Qwen2.5-7B-Instruct-GGUF`).

**3. Configure `.env`**
```sh
OPENAI_API_KEY=lm-studio        # any non-empty string works
OPENAI_BASE_URL=http://localhost:1234/v1
MODEL_ID=lmstudio-community/Qwen2.5-7B-Instruct-GGUF
```

---

#### Option B — 🦙 Ollama

[Ollama](https://ollama.com) is a lightweight CLI that manages and serves models with a single command.

**1. Install Ollama**
```sh
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download the installer from https://ollama.com/download
```

**2. Pull a model & start the server**
```sh
ollama pull qwen2.5:7b          # or: mistral, phi3, llama3.2, gemma2:2b …
ollama serve                    # starts at http://localhost:11434
```
> If you ran `ollama pull` without `ollama serve`, the server is already running in the background — no extra step needed.

**3. Configure `.env`**
```sh
OPENAI_API_KEY=ollama           # any non-empty string works
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_ID=qwen2.5:7b            # must match the name you pulled
```

---

#### Option C — 🌐 GPT4All

[GPT4All](https://www.nomic.ai/gpt4all) offers a desktop app with a built-in API server mode.

**1. Install GPT4All**
- Download and install the desktop app from [nomic.ai/gpt4all](https://www.nomic.ai/gpt4all).

**2. Download a model & enable the API server**
- Go to **Models** → browse and download a model (e.g. `Mistral 7B Instruct`).
- Open **Settings → API Server**, toggle **Enable API Server** on.
- The server starts at `http://localhost:4891/v1`.

**3. Configure `.env`**
```sh
OPENAI_API_KEY=gpt4all          # any non-empty string works
OPENAI_BASE_URL=http://localhost:4891/v1
MODEL_ID=Mistral 7B Instruct   # must match the model name shown in the app
```

---

**4. Run (same for all options)**
```sh
python sessions/en/s01_agent_loop.py
```

> **Tips for CPU inference**
> - **Under 8 GB RAM:** use 1.5B–3B models — e.g. `Qwen2.5-1.5B-Instruct`, `Llama-3.2-1B-Instruct`.
> - **8 GB–16 GB RAM:** use 4-bit quantized 7B–8B models — e.g. `Llama-3.1-8B-Instruct (Q4)`, `Mistral-7B-Instruct (Q4)`.
> - **16 GB+ RAM:** standard 7B–13B models work well without extra quantization.
> - Keep context length at 4096 or lower in your server settings to reduce RAM pressure.
> - The agents already cap `max_tokens` at 8096, so small models won't be overwhelmed.

## 🧩 Dependencies

```
openai>=1.0.0
python-dotenv>=1.0.0
websockets>=12.0
croniter>=2.0.0
python-telegram-bot>=21.0
httpx>=0.27.0
```

## 🔗 Related Projects

- **[learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)** -- A companion teaching repo that builds an agent **framework** (nano Claude Code) from scratch in 12 progressive sessions. Where learn-harness-engineering-by-building-a-mini-openclaw focuses on gateway routing, channels, and proactive behavior, learn-claude-code dives deep into the agent's internal design: structured planning (TodoManager + nag), context compression (3-layer compact), file-based task persistence with dependency graphs, team coordination (JSONL mailboxes, shutdown/plan-approval FSM), autonomous self-organization, and git worktree isolation for parallel execution. If you want to understand how a production-grade unit agent works inside, start there.

## 👥 About
<img width="260" src="https://github.com/user-attachments/assets/fe8b852b-97da-4061-a467-9694906b5edf" /><br>

Scan with Wechat to fellow us,  
or fellow on X: [shareAI-Lab](https://x.com/baicai003)  

## 📄 License

MIT
