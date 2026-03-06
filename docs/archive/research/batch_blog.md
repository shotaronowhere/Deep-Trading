From Solo Player to Multiplayer: Batch Clearing for Prediction Market Portfolios
How we're extending a single-agent prediction market rebalancer to handle N agents trading simultaneously — without anyone getting front-run.

The Starting Point: One Agent, One AMM
Imagine you believe certain outcomes in a prediction market are underpriced. You have a budget, you have beliefs, and you have AMM pools where you can trade. Your goal is simple: buy the outcomes you think are cheap, and sell the ones you think are expensive.

The math boils down to a clean constrained optimization. You want to maximize your expected portfolio value subject to a budget constraint. The complication is that AMM pools have price impact — buying pushes the price up, selling pushes it down. So your cost function is nonlinear, and you can't just buy at the current spot price.

Our existing system solves this with what we call the waterfall algorithm. The idea is intuitive: pour your capital into the most profitable outcome first. As you buy, the price rises and profitability drops. Eventually it drops to the level of the second-most-profitable outcome — at which point you split capital between both, maintaining equal marginal profitability. Then three outcomes, then four. Your capital cascades down like water filling containers of different heights until you run out of money or run out of profitable outcomes.

This isn't a heuristic. It's the exact KKT optimality condition for the constrained problem — at the optimum, the marginal expected return per unit of cost must be equal across all active positions. If it weren't, you could improve by shifting capital from the low-return position to the high-return one.

We even have closed-form solutions. For direct buys within a single Uniswap V3 tick range, the target profitability level π solves to (A/B)² - 1, where A and B are sums involving pool liquidity and predictions. No iteration needed.

This works. It runs in 3.6ms for 98 outcomes. Ship it.

The Problem: What Happens with Multiple Agents?
Now suppose there are N agents, each with their own beliefs and budget. They all want to rebalance against the same AMM pools. What do you do?

The naive approach: run them sequentially. Agent 1 trades, then agent 2 sees the new prices, then agent 3, and so on.

This is bad for three reasons:

It's path-dependent. Different orderings produce different outcomes. If agent 1 goes first and buys outcome X, agent 2 faces a higher price for X. Reverse the order and agent 1 gets the worse deal.

It's unfair. Whoever goes first gets better prices. This isn't some theoretical concern — it's literally the MEV problem that has consumed Ethereum's attention for years.

It's wasteful. Here's the killer: suppose agent A wants to buy outcome 1, and agent B wants to sell outcome 1. If they trade sequentially, A buys from the AMM (paying impact and fees), then B sells to the AMM (receiving less due to impact and fees). Two sets of fees, two rounds of price impact, and the AMM served as an unnecessary intermediary for what was fundamentally a direct trade between A and B.

The Solution: Batch Clearing
Instead of executing trades sequentially, we collect all agents' desired positions, find the optimal joint allocation, and execute only the net residual against the AMMs.

If A wants to buy 10 of outcome 1 and B wants to sell 5 of outcome 1, the AMM only needs to provide 5 — the other 5 transfer directly from B to A. This is what CoW Protocol calls a "Coincidence of Wants." The matched volume avoids all AMM fees and impact.

But our system goes further than simple per-token netting, because prediction markets have a special feature: you can mint complete sets. Pay 1 stablecoin, get one token of every outcome. This unlocks what I'd call "bundle-space netting."

Here's an example. Agent A wants outcome 1. Agent B wants outcome 2. Neither is selling anything. In a per-token netting world, there's nothing to match — both are pure buys. But with minting: the system mints complete sets, gives token 1 to A, token 2 to B, and sells the remaining 96 tokens. If A and B's combined demand is large enough, this "mint and distribute" approach can be substantially cheaper than two separate AMM buys, because you're only paying impact on the sell side for tokens nobody wants, not the buy side for tokens everybody wants.

The batch optimizer finds the right balance automatically.

The Formulation: A Convex Program
The batch clearing is expressed as a single optimization:

Maximize total expected value (summed across all agents, weighted by their beliefs) minus the total cost of interacting with the AMMs and the mint facility.

The constraints are flow balance — for each outcome, what the agents collectively end up holding minus what they started with must equal what was traded on the AMM plus what was minted.

The beauty is that netting is structural. There's no separate "netting phase." The flow balance constraint captures it automatically. If agents' desires partially cancel, the net AMM flow Δ is smaller, and the convex AMM cost C(Δ) is lower. The optimizer finds the maximum-netting solution because minimizing convex cost is equivalent to maximizing netting.

The mint is a single scalar variable M. M > 0 means minting complete sets (buying from the protocol). M < 0 means redeeming (selling back to the protocol). At equilibrium, if the sum of all clearing prices exceeds 1, minting is profitable and M increases until that sum equals exactly 1. If the sum is below 1, redeeming is profitable. This replaces the single-agent's per-outcome route selection (direct vs. mint) with a single continuous decision variable. Much cleaner.

But Wait — What About Individual Budgets?
The pooled problem (maximize total welfare, everyone shares one budget) is convex. Easy to solve. But we need each agent to respect their own budget.

Per-agent budget constraints create a subtle problem. An agent's cost depends on clearing prices, but clearing prices depend on everyone's allocation, which depends on budgets. This creates a bilinear coupling — price times quantity — that makes the combined problem non-convex as written.

The resolution is a standard economic trick: Lagrangian decomposition. Fix the prices, let each agent independently choose their best allocation at those prices, then update the prices based on whether there's excess demand or supply. This is just... a market. We're finding market-clearing prices.

The Algorithm: Active-Set Newton
Here's where it gets interesting. Because agents have linear utility (they just want to maximize expected value — no risk aversion), their demand is bang-bang. Each agent concentrates their entire budget on a single outcome — whichever gives the best ratio of prediction to price. There's no diversification with linear utility.

This makes the demand function discontinuous. Small price changes can cause an agent to jump from buying outcome 3 to buying outcome 7. Traditional smooth optimization methods hate this.

Our solution: active-set Newton. The "active set" is the mapping from each agent to their chosen outcome. Given a fixed active set (everyone has committed to which outcome they're buying), the remaining system is smooth — we just need to find prices where supply equals demand. That's a K×K linear system (K = 98 outcomes), solvable in one Newton step.

The algorithm:

Start with current AMM prices
Each agent picks their best outcome at those prices
Solve for market-clearing prices given those choices
Check if anyone wants to switch outcomes at the new prices
If yes, update the active set and go to 3
If no, we're done
In practice this converges in 5-15 iterations. Each iteration is O(K²) — a 98×98 linear system. For 100 agents and 98 outcomes, total runtime is estimated around 20ms.

When N = 1, this reduces exactly to the existing waterfall. The single agent's "equalize marginal profitability" condition is exactly the competitive equilibrium price for a one-player market.

Why This Is Well-Defined: The Eisenberg-Gale Connection
A natural worry: does this equilibrium even exist? Is it unique? Could Newton be chasing a ghost?

The Eisenberg-Gale theorem, from 1959, answers this for us. It says: the competitive equilibrium for agents with linear utilities and budgets is the unique solution to a convex program — specifically, maximizing the sum of B^a × log(π^a · x^a).

Now, this looks like agents have log utility, but they don't. The log is solver machinery — it's a mathematical device that selects the unique competitive equilibrium. Each agent still has linear utility. The equilibrium gives bang-bang allocations, exactly as we want.

What this gives us for free:

Existence: the convex program has a solution (concave objective, compact feasible set)
Price uniqueness: the dual variables (clearing prices) are unique
The Newton target is well-defined: we're not searching for something that might not exist
We don't implement Eisenberg-Gale as our solver — Newton is faster and simpler. But the theory tells us Newton's target exists and is unique.

Settlement and Fairness
After solving, every agent faces the same clearing price λ_i for each outcome. This is uniform pricing — no ordering effects, no front-running advantage.

Settlement is straightforward: each agent pays Σ λ_i × (new_holdings - old_holdings). The coordinator executes only the net flows on the AMMs and mints. Internal transfers between agents are implicit in the payment settlement.

There's an interesting subtlety with inframarginal surplus. Since AMM cost is convex, the marginal price (what the last unit costs) exceeds the average price (what all units cost on average). Agents collectively pay at the marginal price, but AMM execution only costs the average. The difference is surplus. We recommend burning it or putting it in an insurance fund — distributing it back to agents creates a circular dependency where surplus changes budgets which changes the optimum which changes the surplus.

The mechanism is fair in the economic sense: no agent envies another's trade at the clearing prices (if they did, they would have chosen that trade themselves). It's not strategyproof — a sufficiently large agent can shade their reported beliefs to move prices. But the manipulation gain is bounded by B²/L (budget squared over liquidity), which is negligible for agents small relative to pool liquidity. And strategy-proofness is impossible anyway for any mechanism that's simultaneously efficient, budget-balanced, and individually rational (this is a standard impossibility result in mechanism design).

What Makes This Different from CoW Protocol?
CoW Protocol solves a similar problem for token swaps on Ethereum. The key difference: minting.

In a prediction market, you can create new tokens from thin air (mint complete sets for 1 stablecoin). This is a production facility, not just an exchange. The batch optimizer jointly decides how much to mint, how much to trade on AMMs, and how to distribute internally. This "bundle-space netting" — where agents wanting different outcomes can be served from the same mint — is unique to prediction markets and is a genuine improvement over per-token matching.

The mint variable M replaces discrete route selection (direct buy vs. mint-and-sell-others) with a single continuous optimization variable. The optimizer figures out the right mix automatically. No heuristic needed.

The Fallback Hierarchy
Active-set Newton is the primary solver, but because bang-bang demand can sometimes cause the active set to oscillate (agent A flips between outcome 3 and 7, changing prices, causing agent B to flip, changing prices back...), we have fallbacks:

Homotopy warm-start: before running exact Newton, solve a smoothed version where demand is softmax instead of argmax. Reduce the temperature over 2-3 steps. This often finds the right active set before Newton even starts.

Interior-point on the Eisenberg-Gale program: if Newton cycles, solve the EG convex program directly. Slower (variables scale as N×K) but guaranteed polynomial convergence.

Dampened tâtonnement: the simplest fallback. Adjust prices proportionally to excess demand. Not exact, but with many heterogeneous agents, aggregate demand smooths out the individual discontinuities and convergence is reliable.

In practice, we expect Newton with warm start from AMM prices to handle everything. The fallbacks are insurance.

Implementation: Incremental, Not Big-Bang
The implementation plan builds on the existing single-agent waterfall without replacing it:

For N = 1, delegate to the existing rebalance function (3.6ms, exact, battle-tested)
For N > 1, use the batch Newton solver
Build incrementally: active set computation → KKT solver → Newton loop → mint complementarity → settlement → cycling fallback
Each step has a verification criterion: the N=1 case must match the existing waterfall output exactly
The data structures are simple. An AgentState holds beliefs, budget, and current holdings. The BatchResult returns clearing prices, net AMM flows, per-agent allocations, mint amount, payments, and executable actions. The whole thing targets under 100ms for 100 agents.

Looking Forward
The immediate plan is cooperative agents — a fund with N sub-strategies sharing the same capital. But the formulation generalizes naturally to competitive settings:

Solver competition: multiple solvers compete to produce the best batch, on-chain verification selects the winner (the CoW Protocol model)
Belief privacy: the Newton solver only needs each agent's demand at current prices, not their full belief vector — compatible with encrypted computation
Risk preferences: replacing linear utility with concave utility eliminates bang-bang demand entirely, making all solvers converge smoothly
The single-agent waterfall was a good starting point. The batch formulation is where it gets interesting — it turns a portfolio rebalancer into a micro-exchange, with all the mechanism design questions that entails. The math is clean, the algorithm is fast, and the theory tells us the equilibrium we're computing exists and is unique. Now we just need to build it.