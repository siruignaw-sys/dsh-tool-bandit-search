# dsh-tool-bandit-search

A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) plugin that replaces the standard `web_search` tool with a `search` tool that **learns which search strategy to use** through a contextual multi-armed bandit, instead of relying on a single hardcoded approach.

## Why

Every web search has a tradeoff: a fast, narrow query gets you an answer quickly, but a broader, multi-angle query gets you better coverage at the cost of latency. Hardcoding one strategy means always overpaying for simple questions or always underdelivering on complex ones. This plugin lets the tool discover, from real usage, which strategy tends to pay off — and keeps adapting as conditions change.

## How it works

The `search` tool has two internal strategies ("arms"):

- **`quick`** — a single search query, capped at 5 results. Fast, good for simple factual lookups.
- **`thorough`** — three query variants (the original plus two reframed angles) run in parallel and merged/deduplicated, capped at 10 results. Slower, better for open-ended or multi-perspective questions.

On every call, the plugin uses **Thompson sampling** to pick an arm: each arm has a Beta(α, β) distribution representing its estimated reward, the plugin samples from both distributions, and whichever sample is higher gets used. This naturally balances *exploration* (trying the less-proven arm occasionally) against *exploitation* (favoring the arm that's performed better so far).

After the call, a **continuous reward in [0, 1]** is computed from two components, weighted equally:

- **Quality** — how many results came back, relative to that arm's own maximum (so a 5-of-5 "quick" result is scored the same as a 10-of-10 "thorough" result — neither arm is structurally favored by its own result cap).
- **Speed** — how fast the call completed, calibrated against realistic search latency.

That reward updates the chosen arm's Beta distribution (`α += reward`, `β += 1 − reward`), so the bandit's beliefs shift a little after every single call — no separate training phase, no manual tuning.

The model never sees the two arms directly. It just calls `search(query)`; the plugin decides internally which strategy to run.

## Example output

```
[bandit-search] arm=quick reward=1.000 durationMs=4393 resultCount=5 stats={"quick":{"alpha":2,"beta":1},"thorough":{"alpha":1,"beta":1}}
[bandit-search] arm=thorough reward=0.854 durationMs=8481 resultCount=10 stats={"quick":{"alpha":2,"beta":1},"thorough":{"alpha":1.85,"beta":1.15}}
[bandit-search] arm=quick reward=0.000 durationMs=5777 resultCount=0 stats={"quick":{"alpha":2,"beta":2},"thorough":{"alpha":1.85,"beta":1.15}}
```

Each log line shows which arm was picked, the reward it earned, and the running Beta parameters for both arms — you can watch the bandit's confidence shift in real time as it accumulates evidence.

## Install

```sh
dsh plugin --profile web add github:siruignaw-sys/dsh-tool-bandit-search
```

For local development against a cloned/edited copy instead:

```sh
dsh plugin --profile web add link:/absolute/path/to/dsh-tool-bandit-search
```

Either way, restart the Web UI (a fresh `pnpm dsh web` / `dsh web`, not just a new chat) after installing — bundle installs only take effect on the next boot, and the plugin's system-prompt instruction steering the model toward `search` over the built-in `web_search` tool only applies to sessions started after that.

## Requirements

Runs on top of dsh's native `ctx.web` search service — no separate API key needed beyond whatever search provider your dsh profile already has configured (e.g. `dsh-web-search-deepseek`).

## Known limitations

- **Bandit state is in-memory** and resets on every restart. Persisting it via `ctx.storage` (which dsh already exposes) is a natural next step.
- **Reward is a heuristic**, not a measure of actual answer quality — it captures result count and latency, not whether the results were *relevant* or *correct*. A stronger version might score reward against whether the model's final answer actually used the returned sources.
- **The model can still issue multiple `search` calls per turn** even when `thorough` is already broadening internally — the plugin optimizes strategy *per call*, not the model's own multi-call behavior.
- Built and tested against dsh's developer preview; the plugin/tool APIs may change before a stable release.

## License

MIT
