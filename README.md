# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/aufaakmalbunaya/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/aufaakmalbunaya/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One Python 3.13 standard-library server that serves the OpenAI Chat
Completions API and the Anthropic Messages API over ten simulated free-tier
model providers. For each request it picks the highest-rated free model that
meets the request's requirements, speaks that provider's own wire (OpenAI,
Anthropic or Gemini), fails over to the next-best free model when a provider
refuses, and reports the catalog it read.

## How it works

- The trial hands the server a brief (`$MAKEFUN_PROMPT_FILE`) with the
  provider directory, the quality table and the workload manifest. The server
  probes every provider's catalog route in that provider's own wire and builds
  one catalog from the answers; `python3 main.py --list` prints that same
  document, byte for byte.
- Routing ranks every free model that meets the request (tools, vision, JSON
  mode, context length) by the quality table's rating. A JSON-mode request
  never goes to an Anthropic-wire provider, whose wire cannot carry it. A
  provider whose last two attempts both failed is cooled to the end of the
  list until it answers again.
- Failover is immediate: a 402, 429 or 5xx (or an unreachable provider) moves
  the request to the next-best free model. No same-provider retry: the retry's
  extra call costs the tie-break while its earned gain never moves the
  integer score.
- Wire translation preserves exactly what the provider echoes back: the
  system prompt, user text, tool names, `max_tokens`, stop sequences, image
  hash and JSON mode, plus the provider's usage counts and stop reason,
  mapped between the OpenAI, Anthropic and Gemini shapes in JSON and in
  server-sent events.
- The server registers its port and catalog with the env
  (`POST $MAKEFUN_ENV_URL/register`), serves the workload, polls
  `GET $MAKEFUN_ENV_URL/work` and exits 0 when the env says the workload is
  done. Deterministic: no wall clock and no unseeded randomness in anything
  it sends, fixed timestamps in every reply it builds itself.

## Run it

```sh
python3 main.py            # serve a trial (needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE)
python3 main.py --list     # print the catalog document and exit 0
```

Local practice scores with the kit's env and scorer: 99 on all three
practice trials (quality 99, served 100, wire 100, failover 100,
catalog 100); quality 99 is the ceiling there, set by the providers'
per-trial budgets.

---

<sub>Entered on [make.fun](https://make.fun/?ref=69qprj9z&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
