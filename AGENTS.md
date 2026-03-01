# AGENTS.md

See `.rust-skills/AGENTS.md` for Rust development guidelines.
See https://ethskills.com/ for Ethereum and L2 development guidelines.
See https://docs.uniswap.org/assets/files/llms-0535f49abd170e69dc72fdc37b81dff2.txt for uniswap development guidelines.

**Context:** Local dev environment. User is pre-authenticated via `claude login` and `gcloud auth`.

**Rules:**
- Run all commands in the user's local terminal — no sandboxed environments.
- Do not ask for API keys.
- DO NOT MODIFY CONTRACTS

---

## Tool Selection

| Task | Tool | Pattern |
| :--- | :--- | :--- |
| Complex reasoning, security audits, refactoring | `claude` | `claude -p "PROMPT" [FILES] --model claude-opus-4-6` |

---

## Claude CLI (`@anthropic-ai/claude-code`)

**Always pass `-p`** to run non-interactively (prevents TUI from hanging the session).

### Flags

| Flag | Purpose |
| :--- | :--- |
| `-p "..."` | Non-interactive prompt (required) |
| `--model claude-opus-4-6` | Use Opus 4.6 (latest, most capable) |
| `--output-format json` | Machine-readable output for piping/parsing |

### Examples

don't run more than one claud invokation, if claude hangs for a very long time, ask me what to do, but i can see from my billing console if claude is responding.  with claude -p --output-format json, you usually get a final JSON blob, not continuous token streaming. Wait atleast 10 min. And never kill claude on your own, ask me first.

```bash
# Single file review
claude -p "Review for logic errors, race conditions, and type safety." src/main.ts --model claude-opus-4-6

# Multi-file refactor
claude -p "Refactor to improve modularity and reduce duplication." src/lib/api.ts src/hooks/useApi.ts --model claude-opus-4-6

# Pipe content in
cat src/lib/*.ts | claude -p "Identify shared patterns and suggest abstractions." --model claude-opus-4-6

# JSON output for downstream processing
claude -p "List all exported functions with their signatures." src/index.ts --model claude-opus-4-6 --output-format json
```

### Prompt Guidelines

- Be specific about what to analyze (logic errors, types, performance, security).
- State the desired output format if needed (diff, list, explanation).
- Include relevant constraints (e.g., "preserve the public API", "no breaking changes").


## Project Overview

A prediction market portfolio balancing bot. 

We focus initially on L1 Markets, L2 and originality are TODO.

For L1 Markets, we rebalance a multi-scalar market normalized by one. There are 100 outcomes split across 2 markets, where there are 98 tradeable outcomes. Due to gas constraints, the outcomes were not deployed in the same transaction, but are connected. The first market (0x3220a208aaf4d2ceecde5a2e21ec0c9145f40ba6), has 67 outcomes, where one outcome is a "other repositories not present on the market created on november 26, 2025" (0x3220a208aaf4d2ceecde5a2e21ec0c9145f40ba6), and that outcome is the base, collateral of a conditional market which extends to a further 33 outcomes, one of which is the "invalid result" outcome (0xd1ebfedd9a6480e0407552eb521270c6ddc837b0). These are implementation details, but all pools for the effective 98 tradable outcomes are SUSDS (stable coin) / outcome pairs, so we can neglect the fact that these are technically from two different, but connected markets.

## Architecture

See architecture.md and other relevant .md docs.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


## Coding & Documentation Workflow

1.  **Plan First**: Always propose a detailed implementation plan before generating any code.
2.  **Selective Context**: Read only the documentation files strictly relevant to the current task to avoid context bloat.
3.  **Documentation is Code**: After coding, you must create or update a permanent, well-named markdown file that documents the feature or logic.
4.  **Verification Loop**: Explicitly verify that the written code matches the documentation and vice versa before considering the task complete.

## Build & Run Commands

```bash
cargo build              # Debug build
cargo build --release    # Optimized release build
cargo run                # Run WebSocket-based binary (main.rs)
cargo check              # Quick syntax check
cargo clippy             # Lint checks
cargo fmt                # Format code
```

## Testing

```bash
cargo test               # Run all tests
cargo test test_prepare  # Run specific test
cargo test -- --nocapture  # Run with output visible
```

## Architecture

**Two entry points:**
- `src/main.rs` (binary): WebSocket connection to Optimism RPC (`wss://optimism.drpc.org`), queries Market contract using Alloy's `sol!` macro
- `src/lib.rs` (library): HTTP requests to Seer PM API for L1 market data

**Key patterns:**
- Alloy `sol!` macro for compile-time contract interface definitions
- `ProviderBuilder` pattern for blockchain provider creation
- Tokio async runtime with `#[tokio::main]` and `#[tokio::test]`
- `Box<dyn Error>` for flexible error propagation

**External dependencies:**
- Optimism RPC: `wss://optimism.drpc.org`
- Seer PM API: `https://deep.seer.pm/.netlify/functions/get-l1-markets-data`
# Using Gemini CLI for Large Codebase Analysis

  Only use the gemini-3.1-pro-preview model!

Don't run more than one gemini invokation, if gemini hangs for a very long time, ask me what to do, but i can see from my billing console if gemini is responding. Wait atleast 10 min. And never kill gemini on your own, ask me first.

  When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive
  context window. Use gemini -p to leverage Google Gemini's large context capacity.

  ## File and Directory Inclusion Syntax

  Use the @ syntax to include files and directories in your Gemini prompts. You may need to use "/Users/shotaro/.nvm/versions/node/v20.19.2/bin/gemini". The paths should be relative to WHERE you run the
   gemini command:

 --model=gemini-3.1-pro-preview

  ### Examples:

  Single file analysis:
  ```bash
  gemini -p "@src/main.py Explain this file's purpose and structure"

  Multiple files:
  gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"

  Entire directory:
  gemini -p "@src/ Summarize the architecture of this codebase"