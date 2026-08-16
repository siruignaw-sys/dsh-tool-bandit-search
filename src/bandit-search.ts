import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'bandit-search-tool'
export const inject = ['tools', 'web', 'systemPrompt']

// --- Thompson Sampling bandit state (in-memory; resets on restart) ---
type ArmName = 'quick' | 'thorough'

interface ArmStats {
  alpha: number
  beta: number
}

const arms: Record<ArmName, ArmStats> = {
  quick: { alpha: 1, beta: 1 },
  thorough: { alpha: 1, beta: 1 },
}

// Each arm's own ceiling — quality is measured against what the arm itself
// promises, not a shared constant that favors whichever arm asks for more.
const ARM_MAX_RESULTS: Record<ArmName, number> = {
  quick: 5,
  thorough: 10,
}

function normalSample(): number {
  const u1 = Math.random()
  const u2 = Math.random()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

function gammaSample(shape: number): number {
  if (shape < 1) {
    const u = Math.random()
    return gammaSample(1 + shape) * Math.pow(u, 1 / shape)
  }
  const d = shape - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  while (true) {
    let x: number, v: number
    do {
      x = normalSample()
      v = 1 + c * x
    } while (v <= 0)
    v = v * v * v
    const u = Math.random()
    if (u < 1 - 0.0331 * x ** 4) return d * v
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v
  }
}

function sampleBeta(alpha: number, beta: number): number {
  const x = gammaSample(alpha)
  const y = gammaSample(beta)
  return x / (x + y)
}

function pickArm(): ArmName {
  const sQuick = sampleBeta(arms.quick.alpha, arms.quick.beta)
  const sThorough = sampleBeta(arms.thorough.alpha, arms.thorough.beta)
  return sQuick >= sThorough ? 'quick' : 'thorough'
}

// Continuous reward in [0, 1]: rewards both good coverage (relative to what
// THIS arm itself promises) and speed (calibrated to real observed latency,
// ~5-15s per call), so the bandit learns a genuine tradeoff rather than a
// result-cap artifact.
function computeReward(arm: ArmName, sourceCount: number, durationMs: number): number {
  if (sourceCount === 0) return 0 // hard floor: no results is always bad

  const qualityPart = Math.min(sourceCount / ARM_MAX_RESULTS[arm], 1)
  const speedPart = Math.min(6000 / durationMs, 1) // reward calls near/under 6s

  return 0.5 * qualityPart + 0.5 * speedPart
}

function recordResult(arm: ArmName, reward: number) {
  arms[arm].alpha += reward
  arms[arm].beta += (1 - reward)
}

interface Source {
  url: string
  title?: string
  snippet?: string
}

function broadenQueries(query: string): string[] {
  // Naive but genuine broadening: a couple of angle-shifted variants
  // alongside the original, rather than just re-asking for more results.
  return [
    query,
    `${query} analysis`,
    `${query} latest`,
  ]
}

export function apply(ctx: Context) {
  ctx.systemPrompt.section({
    name: 'tool:search-bandit',
    order: 105,
    text: 'Always use the `search` tool for web searches instead of `web_search`. `search` is a smarter tool that internally chooses the best search strategy for you.',
  })

  ctx.tools.register(defineTool({
    name: 'search',
    description: 'Search the web for information on a query. Internally chooses between a fast single-query strategy and a thorough multi-angle strategy using a learning bandit.',
    parameters: {
      query: { type: 'string', required: true, description: 'The search query' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args: { query: string }, exec) {
      const arm = pickArm()
      const start = Date.now()
      let sources: Source[] = []
      let content: string | undefined
      let reward = 0

      try {
        if (arm === 'quick') {
          const result = await ctx.web.search({ query: args.query, maxResults: ARM_MAX_RESULTS.quick }, exec.signal)
          sources = [...result.sources]
          content = result.content
        } else {
          const queries = broadenQueries(args.query)
          const results = await Promise.all(
            queries.map(q => ctx.web.search({ query: q, maxResults: 5 }, exec.signal)),
          )
          const seen = new Set<string>()
          for (const r of results) {
            for (const s of r.sources) {
              if (seen.has(s.url)) continue
              seen.add(s.url)
              sources.push(s)
            }
          }
          content = results.find(r => r.content)?.content
          sources = sources.slice(0, ARM_MAX_RESULTS.thorough)
        }
      } finally {
        const durationMs = Date.now() - start
        reward = computeReward(arm, sources.length, durationMs)
        recordResult(arm, reward)
        console.log(`[bandit-search] arm=${arm} reward=${reward.toFixed(3)} durationMs=${durationMs} resultCount=${sources.length} stats=${JSON.stringify(arms)}`)
      }

      const lines = sources.map(s => `- ${s.title ?? s.url}: ${s.url}${s.snippet ? `\n  ${s.snippet}` : ''}`)
      return `[strategy used: ${arm}]\n${(content ? `${content}\n\n` : '') + lines.join('\n')}`
    },
  }))
}
