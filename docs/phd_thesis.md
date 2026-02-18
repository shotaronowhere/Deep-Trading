\section*{Convex Network Flows}
by
Theo Diamandis
B.S. Electrical Engineering, Stanford University. 2019
M.S. Electrical Engineering, Stanford University, 2019

Submitted to the Department of Electrical Engineering and Computer Science in partial fulfillment of the requirements for the degree of

DOCTOR OF PHILOSOPHY
at the
MASSACHUSETTS INSTITUTE OF TECHNOLOGY
September 2024
(C) 2024 Theo Diamandis. This work is licensed under a CC BY 4.0 license.

The author hereby grants to MIT a nonexclusive, worldwide, irrevocable, royalty-free license to exercise any and all rights under copyright, including to reproduce, preserve, distribute and publicly display copies of the thesis, or release the thesis under an open-access license.

\begin{tabular}{|l|l|}
\hline Authored by: & Theo Diamandis \\
\hline & Department of Electrical Engineering and Computer Science August 6, 2024 \\
\hline \multirow[t]{3}{*}{Certified by:} & Alan Edelman \\
\hline & Professor of Applied Mathematics \\
\hline & Thesis Supervisor \\
\hline \multirow[t]{3}{*}{Accepted by:} & Leslie A. Kolodziejski \\
\hline & Professor of Electrical Engineering and Computer Science \\
\hline & Chair, Department Committee on Graduate Students \\
\hline
\end{tabular}

\title{
Convex Network Flows
}
by
Theo Diamandis
Submitted to the Department of Electrical Engineering and Computer Science on August 6, 2024 in partial fulfillment of the requirements for the degree of

DOCTOR OF PHILOSOPHY

\begin{abstract}
This thesis introduces a new framework for flow problems over hypergraphs. Our problem formulation, which we call the convex flow problem, only assumes that the constraints on the flows over each edge are in some convex set. The objective is to maximize a sum of concave utility functions - one for the net flow at every node and one for each edge flowsubject to these constraints. This framework not only includes many classic problems in network optimization, such as max flow, min-cost flow, and multi-commodity flows, but also generalizes these problems to allow, for example, concave edge gain functions. As a result, our framework includes applications spanning a number of fields: optimal power flow over lossy networks, routing and resource allocation in ad-hoc wireless networks, Arrow-Debreu Nash bargaining, and order routing through financial exchanges, among others. This problem has a number of interesting properties, including a 'calculus' of flow sets, an equivalent conic form, and a natural generalization of many classic network flow results.

We develop an efficient algorithm for solving the convex flow problem by constructing a particular dual problem that decomposes over the edges of the hypergraph. This dual problem has a number of useful interpretations and admits a straightforward specification: the dual function and its gradient can be evaluated using only simple subroutines which often have closed-form solutions. These subroutines suggest a clean, easy-to-use problem interface, which we provide in the open-source software package ConvexFlows.jl, written in the Julia programming language. We discuss implementation considerations, including how to handle important special cases, and we provide a simple interface for specifying convex flow problems. We show that our solver is significantly faster than the state-of-the-art commercial optimization solver Mosek, even for small problems sizes with limited parallelization.

Finally, we consider the nonconvex flow problem with fixed costs on the edges, i.e., where there is some fixed cost to send any nonzero flow over an edge. We show that this problem has almost integral solutions by a Shapley-Folkman argument, and we provide a simple modification of our original algorithm for this nonconvex problem. We conclude by discussing a number of interesting avenues for future work.
\end{abstract}

Thesis supervisor: Alan Edelman
Title: Professor of Applied Mathematics

To my parents

\section*{Acknowledgments}

First, I would like to give a huge thank you to my advisor, Alan Edelman. Alan picked me up as a wandering second-year student, trying to reorient my research direction. In fact, before Alan took me into his group, I was so close to dropping out that I had lined up a job. I'm so glad I joined Alan's group instead! Since joining the Julia Lab, Alan has given me the freedom to explore my interests across a number of different fields (and even geographies), while being a constant source of advice. I'm grateful to Alan not only for answering all my esoteric linear algebra questions with both the relevant mathematics and Julia code, but also for teaching me that, in research, it's important to tell a good story. I hope that I've managed to do that here.

I'd also like to thank my committee members, Pablo Parrilo and Patrick Jaillet. I'd especially like to thank Pablo for supporting me in creating and teaching a convex optimization course during IAP, which was one of the most rewarding experiences I've had at MIT, and to thank Patrick for entertaining my hare-brained ideas that are totally unrelated to my thesis. Both also provided valuable feedback on work that has gone into this thesis, and I've enjoyed the whiteboard sessions we have had during my time at MIT.

Research is (thankfully!) not a solo journey. I'm grateful to have been part of the Julia Lab. Chris Rackauckas and Alan have assembled a incredible group of people. (I also have to thank Chris for Julia-pilling me early on in grad school; taking Chris' class had a profound influence on my grad school trajectory.) I think few labs have experts in everything from random matrix theory to differential equation solvers to the intricacies of compilers. This diversity makes the lab such a fun place to do a PhD. I'll also always look back fondly on the stretch of 2021 that Flemming Holtorf, Alex LeNail, and I had the entire office (and most of the floor) to ourselves. I couldn't think of better people to share it with.

I've been lucky to work with many incredible collaborators over the past few years. I'd like to thank a number who have been particularly influential in my research journey. Madeleine Udell and Zach Frangella introduced me to the fun world of randomized numerical linear algebra. Bartolomeo Stellato inspired and helped me begin to write performant optimization solvers. (Though I still have a ways to go!) I've enjoyed working with both Mallesh Pai and Ciamac Moallemi on real-world applications of optimization, and I also appreciate both of their mentorship. I've had the pleasure to work on many projects with my friends from 'SCP': Guillermo Angeris, Tarun Chitra, Kshitij Kulkarni, and Alex Evans. You all have dramatically expanded my research aperture - from new academic areas, including photonics and game theory, to novel real-world applications including decentralized finance. You all have been so much fun to work with, and I look forward to many more projects together in the future. I'd like to also thank Tarun for making my time in New York so fun and
academically enriching. Every time we hang out I leave with new ideas. Finally, I want to extend a special thank you to Guillermo. Who would've thought when we started working together during Covid (There are few others I'd want to spend an entire day with on Zoom.) that, only a few years later, we'd have over ten papers together? (And counting!) Thanks for being such a great collaborator, conference buddy, and friend. I look forward to doing much more math over many more glasses of wine together in the future.

I'm lucky to have many friends who have been an integral part of my PhD journey and made the last five years so special. There are far too many names to list, but I wanted to give a shout-out to a few. I'm so grateful to have met a great EECS crew during the first semester of grad school: Axel Feldmann, Logan Engstrom, Evan Hernandez and AJ Miller. From our first-year trivia nights at the Thirsty Ear to our continued regular dinners, I can't imagine a better crew to have survived a pandemic and a PhD with. I'd especially like to thank Logan for keeping me sane (and fed) during summer 2020 and Axel for always being game to discuss the finer points of Matt Levine's latest column. Through my entire PhD, I've enjoyed regular runs with Axel and with Sam Goldman; I always felt better-and more energized-after a long conversation with y'all. Tom Dean and David Barton, I'm really glad we overlapped in Cambridge. I've appreciated your mentorship since undergrad and really enjoyed our many dinners and glasses of wine together. Chandler Squires, thanks for always being down for a last-minute hike. Dylan Wootton, I already miss our regular Life Alive breakfasts and our late night, meandering conversations. Shannon West, I'm so grateful we ended up in the same city again and loved hanging out so much during our time as next door neighbors. I'm also grateful for my book club crew: Katie Reinders, Liz Yao, and Ryan Cohen. I miss our regular discussions. Who would've thought I would end up learning so much biology during my EECS PhD? Over the last few years, I've also loved having the opportunity to spend so much time in with my friends in New York City. I'm especially grateful to Max Franzblau and Adam Slakter for being sounding boards through my PhD and frequently giving me a couch to crash on. I'm so glad we've stayed close practically since birth, and thank you both for showing me the city. Omissions from this list are not due to lack of importance, so if you aren't listed here, please bug me for dinner or a drink soon!

Finally, and most importantly, I'd like to thank my family and especially my parents. From a young age, you supported me wholeheartedly in pursuing my interests-from my childhood infatuation with the engineering behind roller coasters to my middle school self's desire to go to early-morning math team practices-despite how strange these might have seemed to two attorneys. Without any exaggeration, I would not be anywhere close to where I am today without your support. Spending six months at home with y'all during my PhD and sharing some of the experience with you was a silver lining to the pandemic. You all have ridden the ups and downs of the PhD with me and continue to be a much-needed source of advice and support. (Although the days of getting help with my homework are long gone.) You've made everything possible. I love you.

\section*{Contents}
Title page ..... 1
Abstract ..... 2
Acknowledgments ..... 4
List of Figures ..... 9
1 Introduction ..... 12
1.1 Related work ..... 13
1.2 Outline ..... 14
1.3 Origins of this thesis ..... 15
2 The convex flow problem ..... 18
2.1 Downward closure and monotonicity ..... 21
2.2 A calculus of flows ..... 22
2.2.1 Composition rules ..... 23
2.3 The conic problem ..... 26
2.3.1 Basic properties ..... 26
2.3.2 Reduction ..... 27
3 Applications ..... 31
3.1 Maximum flow and friends ..... 31
3.1.1 Maximum flow ..... 33
3.1.2 Minimum cost flow ..... 34
3.1.3 Concave edge gains ..... 35
3.1.4 Multi-commodity flows ..... 36
3.2 Optimal power flow ..... 37
3.3 Routing and resource allocation in wireless networks ..... 38
3.4 Market equilibrium and Nash bargaining ..... 40
3.5 Routing orders through financial exchanges ..... 43
4 The dual problem and flow prices ..... 47
4.1 Dual decomposition ..... 47
4.2 The dual problem ..... 50
4.2.1 Cycle condition ..... 52
4.2.2 Downward closure and monotonicity ..... 53
4.3 Special cases ..... 54
4.3.1 Zero edge utilities ..... 54
4.3.2 Circulation problem ..... 55
4.4 Conic dual and self-duality ..... 56
5 Solving the dual problem. ..... 59
5.1 Two-node edges ..... 60
5.1.1 Gain functions ..... 61
5.1.2 Properties ..... 62
5.1.3 Bounded edges ..... 63
5.2 Restoring primal feasibility ..... 64
5.3 Numerical examples ..... 66
5.3.1 Optimal power flow ..... 66
5.3.2 Routing orders through financial exchanges ..... 69
6 Solver: ConvexFlows.j1 ..... 72
6.1 Interface ..... 72
6.1.1 The first subproblem ..... 72
6.1.2 The arbitrage subproblem ..... 73
6.2 Algorithmic modifications ..... 74
6.3 Simple examples ..... 74
6.3.1 Optimal power flow. ..... 74
6.3.2 Trading with constant function market makers ..... 76
6.3.3 Market clearing ..... 76
6.4 Example: multi-period optimal power flow ..... 78
7 Fixed edge fees ..... 83
7.1 Integrality constraint ..... 84
7.2 Convex hull ..... 84
7.3 Convex relaxation ..... 86
7.4 Tightness of the relaxation ..... 87
7.5 Fixed cost dual problem ..... 90
8 Conclusion ..... 91
A Extended monotropic programming ..... 92
B Additional details for the numerical experiments ..... 94
B. 1 Optimal power flow ..... 94
B. 2 Routing orders through financial exchanges ..... 95
C An Efficient Algorithm for Optimal Routing Through Constant Function Market Makers ..... 99
C. 1 Introduction ..... 99
C. 2 Optimal routing ..... 100
C.2.1 Constant function market makers ..... 101
C. 3 An efficient algorithm ..... 103
C.3.1 Dual decomposition ..... 103
C.3.2 The dual problem ..... 10.5
C.3.3 Solving the dual problem ..... 10.5
C. 4 Swap markets ..... 106
C.4.1 General swap markets ..... 106
C. 5 Closed form solutions ..... 109
C. 6 Implementation ..... 110
C.6.1 Markets ..... 110
C.6.2 Utility functions. ..... 112
C. 7 Numerical results ..... 112
C. 8 Conclusion ..... 114
D The Geometry of Constant Function Market Makers ..... 115
D. 1 Introduction ..... 115
D.1.1 What is a CFMM? ..... 116
D. 2 Fee-free constant function market makers ..... 117
D.2.1 Reachable set ..... 118
D.2.2 Composition rules ..... 120
D.2.3 Liquidity cone and canonical trading function ..... 122
D.2.4 Dual cone and portfolio value function ..... 128
D.2.5 Connection to prediction markets ..... 134
D.2.6 Liquidity provision ..... 137
D. 3 Single trade ..... 139
D.3.1 Trading set ..... 139
D.3.2 Trading cone and dual ..... 141
D.3.3 Routing problem ..... 146
D.3.4 Path independence ..... 147
D. 4 Conclusion ..... 149
D.5 Appendix ..... 151
D.5.1 Primer on conic duality ..... 151
D.5.2 Curve ..... 153
D.5.3 Proof of concavity of Uniswap v3 ..... 153
D.5.4 Proof of consistency ..... 154
References ..... 155

\section*{List of Figures}
2.1 A hypergraph with 4 nodes and 3 edges (left) and its corresponding bipartite graph representation (right). ..... 19
2.2 A set of allowable flows $T$ (left) and its downward closure $\tilde{T}$ (right) for a two-node directed edge that for input $w$ outputs $h(w)=w /(1+w)$ units of flow. ..... 22
2.3 Multiplying the set $T$ (left) with the nonnegative matrix, followed by taking the downward closure (middle), results in another set of allowable flows (right). ..... 24
2.4 We take the Minkowski sum of the sets of allowable flows $T$ and $\tilde{T}$ for two directed edges with the form of that in figure 2.2 (left and middle) to obtain a new set of allowable flows $T+\tilde{T}$ that corresponds to an undirected edge (right). ..... 24
2.5 Each tick of the orderbook (left) corresponds to a linear edge with coeffi- cient corresponding to the exchange rate (middle). These linear edges can be combined into an aggregate edge defining the entire orderbook (right). ..... 25
3.1 A simple directed (hyper)edge $e_{i}$ connecting nodes $v_{1}$ and $v_{2}$ (left) and the corresponding set of allowable flows $T_{i}$ (right). ..... 32
3.2 An example convex nondecreasing cost function $c(w)=w^{2}$ for $w \geq 0$ (left) and its corresponding concave, nondecreasing edge utility function $V$ (right). ..... 35
3.3 An example concave edge gain function $\gamma(w)=\sqrt{w}$ (left) and the correspond- ing allowable flows (right). ..... 36
3.4 The power loss function (left), the corresponding power output (middle), and the corresponding set of allowable flows (right). ..... 37
3.5 Wireless ad-hoc network. The (outgoing) hyperedge associated with user $u$ is shown in blue, and the corresponding set of outgoing neighbors $O_{u}$ contains user $v$ and the two base stations, $b_{1}$ and $b_{2}$. ..... 39
3.6 Network representation of the linear Fisher market model, where we aim to maximize the utility of the net flows $y$. Colored edges represent the $n_{b}+1$ hyperedges connecting each buyer to all the goods, so each of these edges is incident to $n_{g}+1$ vertices. Flows on the right indicate that there is at most one unit of each good to be divided among the buyers. ..... 42
3.7 Left: the trading set for Uniswap (without fees) for $R=(1,1)$ (light gray) and $R=(2,2)$ (dark gray). Right: the trading set for Uniswap v3. ..... 45
4.1 At optimality, the local prices $\eta_{i}^{\star}$ define a supporting hyperplane of the set of allowable flows $T_{i}$ at an optimal flow $x_{i}^{\star}$. ..... 52
5.1 Sample network for $n=100$ nodes. ..... 67
5.2 Convergence of ConvexFlows with $n=100$ using L-BFGS-B (left) and BFGS (right). On the left, the objective is compared to a high-precision solution from Mosek. The primal residual measures the net flow constraint violation, with $\left\{x_{i}\right\}$ from (4.3c) and $y$ from (4.3a). Note the linear convergence of $\mathrm{L}-\mathrm{BFGS}$ and the superlinear convergence of BFGS . ..... 68
5.3 Comparison of ConvexFlows with Mosek. Lines indicate the median time over 10 trials, and the shaded region indicates the 25th to 75th quantile range. Dots indicate the maximum time over the 10 trials. ..... 69
5.4 Convergence of ConvexFlows on an example with $n=100$ assets and $m=$ 2500 markets. ..... 70
5.5 Comparison of ConvexFlows and Mosek for $m$ varying from 100 to 100,000 and $n=2 \sqrt{m}$. Lines indicate the median time over 10 trials, and the shaded region indicates the 25th to 75th quantile range. Dots indicate the maximum time over the 10 trials. ..... 71
6.1 Graph representation of a power network with three nodes over time. Each solid line corresponds to a transmission line edge, and each dashed line corre- sponds to a storage edge. ..... 78
6.2 Power generated (top), power used by the first node (middle) and by the second node, which has a battery (bottom). ..... 82
7.1 The set $Q_{i}$ (left) and its convex hull $\boldsymbol{\operatorname { c o n v }}\left(Q_{i}\right)$ (right). ..... 85
7.2 A visual representation of the Shapley-Folkman lemma for the $1 / 2$-norm ball. As we take the Minkowski sum of the set with itself, it becomes closer and closer to its convex hull. ..... 88
C. 1 Solve time of Mosek vs. CFMMRouter.jl (left) and the resulting objective values for the arbitrage problem, with the dashed line indicating the relative increase in objective provided by our method (right). ..... 113
C. 2 Average price of market sold ETH in routed vs. single-pool (left) and routed vs. single-pool surplus liquidation value (right). ..... 113
D. 1 The set of reachable reserves for Uniswap (left) and Uniswap v3 (right). ..... 119
D. 2 Adding two Uniswap v3 bounded liquidity pools (left, middle) gives us another CFMM (right). ..... 120
D. 3 Another interpretation of the canonical trading function (D.6): we scale along the line segment defined by $\left(R_{1}, R_{2}\right)$ to $(0,0)$, with scale factor $1 / \lambda$, increasing $\lambda$ until we hit the reachable set boundary. ..... 124
D. 4 Left: the liquidity cone for Uniswap, with the level set defined by the trading function $\varphi(R)=\sqrt{R_{1} R_{2}}=1$ shown. Right: each $\lambda$-level set of the surface looks like the boundary of the set of reachable reserves (see figure D.1). The trading function $\varphi$ is highlighted. ..... 127
D. 5 Left: the trading set for Uniswap (without fees) for $R=(1,1)$ (light gray) and $R=(2,2)$ (dark gray). Right: the trading set for Uniswap v3. ..... 140
D. 6 The trading set for Uniswap with fees (notice that the set is kinked at 0 ) and the corresponding no-trade cone. ..... 145
D. 7 The trading set $T\left(R^{\prime}\right)$ for Uniswap (left) and the corresponding reachable set $S$ (right). ..... 149
D. 8 Equivalence between CFMM representations. Arrows represent easy transfor- mations between objects that were introduced in this work. ..... 150

\section*{Chapter 1}

\section*{Introduction}

Network problems are abundant. Most models assume linearity. Many engineering applications require convexity.

Theorists and practitioners both apply network flow models to describe, analyze, and solve problems from many domains-from routing trucks to routing bits. For linear flows, an extensive academic literature developed the associated theory, algorithms, and applications. (See, e.g., [AMO88], [Wil19], and references therein.) However, these linear models often fail to describe real networked systems. For example, in electrical systems, the power lost increases as more power is transmitted; in communications systems, the message failure rate increases as more messages are transmitted; and, in financial systems, the price of an asset increases as more of that asset is purchased. In each of these cases, the marginal flow out of an edge decreases as the flow into this edge increases; in other words, the output of the edge is a concave function of its input. Additionally, there are many applications where the flows through multiple edges connected to a single node are nonlinearly related. For example, in a wireless network, a transmitter has a power constraint across all of its links. Alternatively, in economics, utilities may be superadditive when goods are complements. The fact that classical network flow models cannot incorporate these well-studied applications suggests that there is a natural generalization that can.

This thesis introduces the convex flow problem, a generalization of the network flow problem that significantly expands its modeling power. We introduce two key ideas which, taken together, allow our framework to model many additional problems present in the literature. First, instead of a graph, we consider flows over a hypergraph: a graph where an edge may connect more than two vertices. Second, we consider the allowable flows for each edge (which may contain more than two vertices) to be a general convex set. This setup includes, as special cases, the linear relationship studied in most network flow problems and the concave, monotonic increasing edge input-output functions studied in [Shi06; Vég14]. Our framework also encompasses a number of other problems in networked physical and economic systems previously studied in the literature. In many cases, it offers immediate generalizations or more succinct formulations. We outline examples from a number of fields, including power systems, wireless networks, Fisher markets, and financial asset networks.

The convex flow problem is a convex optimization problem which can, in practice, be efficiently solved. Our framework preserves the overall network structure present in the problem and provides several interesting insights. These insights, in turn, allow us to de-
velop an efficient algorithm for solving the convex flow problem. We show that the dual problem decomposes over the network's edges, which leads to a highly-parallelizable algorithm that can be decentralized. Importantly, this algorithm has a clean problem interface: we only need access to (1) a Fenchel conjugate-like function of the objective terms and (2) the solution to a simple subproblem for each edge. These subproblem evaluations can be parallelized and have efficiently-computable (and often closed form) solutions in many applications. As a result, our algorithm enjoys better scaling and order-of-magnitude faster solve times than commercial solvers like Mosek. Furthermore, the framework naturally extends to the nonconvex problem of convex network flows with fixed edge costs.

\subsection*{1.1 Related work}

The classic linear network flow problem has been studied extensively and we refer the reader to [AMO88] and [Wil19] for a thorough treatment. In the classic setting, edges connecting more than two vertices can be modeled by simply augmenting the graph with additional nodes and two-node edges. While nonlinear cost functions have also been extensively explored in the literature (e.g., see [Ber98] and references therein), nonlinear edge flows-when the flow out of an edge is a nonlinear function of the flow into it-has received considerably less attention despite its increased modeling capability.

Nonlinear edge flows. Extending the network flow problem to include nonlinear edge flows was first considered by Truemper [Tru78]. Although Truemper formulated a max flow and min cost flow problem for nonlinear edge gain functions, his algorithm to solve these problems used piecewise linear approximations to these functions. Work in the subsequent decades mainly focused on the linear case - when the flow out of an edge is a linear function of the flow into that edge-possibly with a convex cost function in the objective. (See, for example, [Ber98] and references therein.) More recently, Shigeno [Shi06] and Végh [Vég14] considered the maximum flow problem where the flow leaving an edge is a concave function of the flow entering that edge and proposed theoretically efficient algorithms tailored to this case. This problem is a special case of the convex flow problem we introduce in this work. The nonlinear network flow problem has also appeared in a number of applications, which we refer to in the relevant sections.

Dual decomposition methods for network flows. The use of dual decomposition methods for network flow problems has a long and rich history, dating back to Kuhn's 'Hungarian method' for the assignment problem [Kuh55]. The optimization community has explored these methods extensively for network optimization problems (e.g., see [Ber98, §6, §9]) and, more generally, for convex optimization problems with coupling constraints (e.g., see [Ber16, §7]). These methods have also been applied to many network problems in practice. For example, they have facilitated the analysis and design of networking protocols, such as those used for TCP congestion control [Chi+07]. These protocols are, in essence, distributed, decentralized algorithms for solving some global optimization problem.

Extended monotropic programming. Perhaps most related to our framework is the extended monotropic programming problem, introduced by Bertsekas [Ber08], of which our convex flow problem is a special case. Both the convex flow problem and the extended monotropic programming problem generalize Rockafellar's monotropic programming problem [Roc84]. The strong duality result of [Ber08], therefore, applies to our convex flow problem as well, and we make this connection explicit in appendix A. Although the convex flow problem we introduce is a special case of the extended monotropic programming problem, our work differs from that of Bertsekas along a number of dimensions. First, we construct a different dual optimization problem which has a number of nice properties. Second, this dual leads to a different algorithm than the one developed in [Ber08] and [Ber15, §4], and our dual admits an easier-to-implement interface with simpler 'subproblems'. Finally, while the application to multi-commodity flows is mentioned in [Ber08], we show that our framework encompasses a number of other problems in networked physical and economic systems previously studied in the literature, and we numerically illustrate the benefit of our approach.

\subsection*{1.2 Outline}

The remainder of this thesis is organized as follows.
Chapter 2 introduces a general framework for optimizing convex flows over hypergraph structures, where each edge may connect for than two vertices. We call this problem the convex flow problem. We show that this framework admits a helpful 'calculus' of edge flows and derive an equivalent conic problem.

Chapter 3 shows that this framework encompasses a number of problems previously studied in the literature and, in many cases, offers immediate generalizations. We discuss the classic network flow problems and their generalizations in §3.1. We then discuss applications from many fields: the optimal power flow problem in §3.2, routing through ad-hoc wireless networks in §3.3, clearing Fisher markets in §3.4, and constructing orders in financial asset networks in §3.5.

Chapter 4 derives a particular dual problem for the convex flow problem, which we show has many useful interpretations and decomposes nicely over the edges of the network. We examine the special case of zero edge flow utilities, as this case is common in practice. In addition, we derive the dual of the conic problem, which we show is almost self-dual.

Chapter 5 proposes an efficient algorithm to solve the convex flow problem and discusses implementation details, including the common special case of two-node edges. We also provide a method to recover a primal solution from the dual when the problem is not strictly convex. Finally, we use our method to solve a number of numerical examples and compare run times to the preeminent commercial convex optimization solver, Mosek.

Chapter 6 introduces the Julia package ConvexFlows.jl, which implements the solver for the convex flow problem and an easy-to-use interface. This interface allows non-sophisticated users to easily specify instances of the convex flow problem.

Chapter 7 extends the convex flow problem to include fixed costs on the edges. Although this problem is nonconvex, we show that its convex relaxation can be solved efficiently using our algorithm and that this relaxation has an almost-integral solution.

Chapter 8 concludes this thesis and suggests fruitful directions for future work.

\subsection*{1.3 Origins of this thesis}

We'd all like to pretend the frameworks we create-especially when they neatly encapsulate many existing problems-are a result of divine inspiration. The truth, in this case, is much more pedestrian. During my 2022 internship, I was asked to come up with an implement an efficient algorithm for the problem of finding an optimal trade given a utility function and a number of constant function market makers. (These are a particular type of automated market maker which have become widely implemented for public blockchain-based exchanges.) This optimization problem had already been formulated in [Ang+22a], but one quickly ran into difficulties attempting to solve any real-world example with off-the-shelf software. The problem was ill-conditioned but still required high accuracy, and interior point solvers like the commercial solver Mosek simply didn't scale to useful problem sizes.

I implemented a dual decomposition algorithm for my internship project, which we released as the open-source Julia package CFMMRouter.jl. The package became fairly popular, quickly passing 200 stars on GitHub. In addition, in the course of deriving the dual algorithm, we improved the original problem formulation, allowing for easy incorporation of much more complicated markets. This improvement led to a more general formulation, subsequently written up after I returned to MIT in [Dia + 23] and then presented at Financial Cryptography the next year. Through this process of altering our formulation and method to handle more and more complicated markets, I began to suspect that we may have inadvertently solved a more general problem. Indeed, the problem of finding the optimal trade was exactly a generalization of the classic maximum flow problem, where we now allowed for edges connecting more than two vertices and for concave edge input-output functions.

Eventually, this suspicion prompted me to scour the network optimization literature, looking for the max flow generalization onto which we stumbled. Surprisingly, while the network flows literature is vast, the literature that examines the case of nonlinear edge gains is quite sparse. However, I did manage to find many other applications that more or less looked like the max flow problem that we solved for order routing in constant function market makers. After further generalizing our problem formulation to include many of these examples already in the literature, we wrote up the framework in [DAE24a] and called the associated problem the convex flow problem. Notably, this problem formulation included edge flow utilities, which extended our previous work and, as a result, generalized not only the max flow problem but also the min cost flow problem.

Unfortunately, the interface to use the solver written for [DAE24a] would baffle anyone without graduate-level knowledge of convex analysis. In the constant function market maker case, the interface was fairly easy; the building blocks were well-defined and corresponded to known operations. In the general case, this was not so. As a result, we developed a more userfriendly interface in the Julia package ConvexFlows.jl, which was written up in [DA24] and in the package documentation. We hoped that this package made our framework more accessible for unsophisticated users, i.e., those without graduate-level convex optimization courses. (I'll admit, 'unsophisticated' may be a little harsh of a term given this bar.)

Simultaneously, I was exploring the fundamental theoretical properties of constant function market makers, which we eventually wrote up in $[\mathrm{Ang}+23]$. This work asks the question 'what, at its core, is a constant function market maker?'. The answer is that these automated market makers are simply convex sets with particular properties. It turns out that the sets of allowable edge flows in [DAE24a] can be required to have the exact same properties without affecting the solution of the convex flow problem. Instantly, we had ported many of the results from constant function market makers to the more general convex network flows setting. These results will appear in [DAE24b], in addition to a novel conic formulation that is almost self-dual (similar to Bertsekas' extended monotropic programming problem [Ber08]) and also heavily inspired by insights gleaned from our work on constant function market makers.

In the course of working with practitioners on the original problem of financial order routing, another problem arose: most real markets had some type of fixed fee. Eventually, via the conic formulation, we had the necessary theoretical tools to tackle this problem, and we describe in [DAE24b] how to fit the convex flow problem with fixed fees cleanly into our existing framework and algorithm.

Over the past three years, this back and forth between practice and theory, and between specific applications and general problem structures has been exceptionally fruitful for me. I look forward to continuing to 'touch grass' in the future while attempting to generalize application-specific insights into more broadly-applicable results. I hope this thesis gives a little taste of this interplay.

For completeness, I include the full papers on the algorithm for routing orders through constant function market makers $[\mathrm{Dia}+23]$ and the related theory for constant function market makers $[\mathrm{Ang}+23]$ in appendices C and D respectively.

Published work and preprints. This thesis is based on the following papers. Note that we presented [Ang+ 23] at Economics and Computation 2024 but chose to submit it to a journal.
- Theo Diamandis, Max Resnick, Tarun Chitra, and Guillermo Angeris. "An efficient algorithm for optimal routing through constant function market makers". In: International Conference on Financial Cryptography and Data Security. Springer. 2023, pp. 128-145
- Guillermo Angeris, Tarun Chitra, Theo Diamandis, Alex Evans, and Kshitij Kulkarni. "The geometry of constant function market makers". In: arXiv preprint arXiv:2308.08066 (2023)
- Theo Diamandis, Guillermo Angeris, and Alan Edelman. "Convex Network Flows". In: arXiv preprint arXiv:2404.00765 (2024)
- Theo Diamandis and Guillermo Angeris. "Solving the Convex Flow Problem". In: to be presented at the Conference on Decision and Control 2024 (2024). url: theodiamandis. com/pdfs/papers/routing-algorithm.pdf
- Theo Diamandis, Guillermo Angeris, and Alan Edelman. "The Geometry of Convex Network Flows". In: in preparation (2024)
- Kshitij Kulkarni, Theo Diamandis, and Tarun Chitra. "Routing MEV in Constant Function Market Makers". In: International Conference on Web and Internet Economics. Springer. 2023, pp. 456-473

\section*{Chapter 2}

\section*{The convex flow problem}

In this section, we introduce the convex flow problem, which generalizes a number of classic optimization problems in graph theory, including the maximum flow problem, the minimum cost flow problem, the multi-commodity flow problem, and the monotropic programming problem, among others. Our generalization builds on two key ideas: first, instead of a graph, we consider a hypergraph, where each edge can connect more than two nodes, and, second, we represent the set of allowable flows for each edge as a convex set. These two ideas together allow us to model many practical applications which have nonlinear relationships between flows.

Hypergraphs. We consider a hypergraph with $n$ nodes and $m$ hyperedges. Each hyperedge (which we will refer to simply as an 'edge' from here on out) connects some subset of the $n$ nodes. This hypergraph may also be represented as a bipartite graph with $n+m$ vertices, where the first independent set contains $n$ vertices, each corresponding to one of the $n$ nodes in the hypergraph, and the second independent set contains the remaining $m$ vertices, corresponding to the $m$ edges in the hypergraph. An edge in the bipartite graph exists between vertex $i$, in the first independent set, and vertex $j$, in the second independent set, if, and only if, in the corresponding hypergraph, node $i$ is incident to (hyper)edge $j$. Figure 2.1 illustrates these two representations. From the bipartite graph representation, we can easily see that the labeling of 'nodes' and 'edges' in the hypergraph is arbitrary, and we will sometimes switch these labels based on convention in the applications. While this section presents the bipartite graph representation as a useful perspective for readers, it is not used in what follows.

Flows. On each of the edges in the graph, $i=1, \ldots, m$, we denote the flow across edge $i$ by a vector $x_{i} \in \mathbf{R}^{n_{i}}$, where $n_{i} \geq 2$ is the number of nodes incident to edge $i$. Each of these edges $i$ also has an associated closed, convex set $T_{i} \subseteq \mathbf{R}^{n_{i}}$, which we call the allowable flows over edge $i$, such that only flows $x_{i} \in T_{i}$ are feasible. We will require that $0 \in T_{i}$, i.e., we have the option to not use an edge, and that $T_{i}$ is bounded from above: there exists some $b_{i}$ such that
$$
\begin{equation*}
x \leq b_{i} \mathbf{1} \quad \text { for all } \quad x \in T_{i} \tag{2.1}
\end{equation*}
$$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-019.jpg?height=502&width=1449&top_left_y=240&top_left_x=388}
\captionsetup{labelformat=empty}
\caption{Figure 2.1: A hypergraph with 4 nodes and 3 edges (left) and its corresponding bipartite graph representation (right).}
\end{figure}

If this were not the case, some edge could, with bounded input flow, output an infinite amount of flow to at least one of its incident nodes. (Note that we do not need this assumption for most of what follows; we will explicitly state when it is needed.) By convention, we will use positive numbers to denote flow out of an edge (equivalently, into a node) and negative numbers to denote flow into an edge (equivalently, out of a node). For example, in a standard graph, every edge connects exactly two vertices, so $n_{i}=2$. If 1 unit of flow travels from the first node to the second node through an edge, the flow vector across that edge is
$$
x_{i}=\left[\begin{array}{c}
-1 \\
1
\end{array}\right] .
$$

If this edge is bidirectional (i.e., if flow can travel in either direction), lossless, and has some upper bound $b_{i}>0$ on the flow (sometimes called the 'capacity'), then its allowable flows are
$$
T_{i}=\left\{z \in \mathbf{R}^{2} \mid z \leq b_{i} \mathbf{1} \text { and } z_{1}+z_{2}=0\right\} .
$$

While this formalism may feel overly cumbersome when dealing with standard graphs, it will be useful for working with hypergraphs.

Local and global indexing. We denote the number of nodes incident to edge $i$ by $n_{i}$. This set of 'local' incident nodes is a subset of the 'global' set of $n$ nodes in the hypergraph. To connect the local node indices to the global node indices, we introduce matrices $A_{i} \in \mathbf{R}^{n \times n_{i}}$. In particular, we define $\left(A_{i}\right)_{j k}=1$ if node $j$ in the global index corresponds to node $k$ in the local index, and $\left(A_{i}\right)_{j k}=0$, otherwise. For example, consider a hypergraph with 3 nodes. If edge $i$ connects nodes 2 and 3 , then
$$
A_{i}=\left[\begin{array}{ll}
0 & 0 \\
1 & 0 \\
0 & 1
\end{array}\right]=\left[\begin{array}{cc}
\mid & \mid \\
e_{2} & e_{3} \\
\mid & \mid
\end{array}\right] .
$$

Written another way, if the $k$ th node in the edge corresponds to global node index $j$, then the $k$ th column of $A_{i}$, is the $j$ th unit basis vector, $e_{j}$. The matrix $A_{i}$ has the general form
$$
A_{i}=\left[\begin{array}{lll}
a_{1} & \ldots & a_{n_{i}} \tag{2.2}
\end{array}\right],
$$
where each $a_{k} \in \mathbf{R}^{n}$ is a distinct unit basis vector. Note that the ordering of nodes in the local indices need not be the same as the global ordering.

Net flows. By summing the flow in each edge, after mapping these flows to the global indices, we obtain the net flow vector
$$
y=\sum_{i=1}^{m} A_{i} x_{i} .
$$

We can interpret $y$ as the netted flow across the hypergraph. If $y_{j}>0$, then node $j$ ends up with flow coming into it. (These nodes are often called sinks.) Similarly, if $y_{j}<0$, then node $j$ must provide some flow to the network. (These nodes are often called sources.) Note that a node $j$ with $y_{j}=0$ may still have flow passing through it; zero net flow only means that this node is neither a source nor a sink.

Utilities. Now that we have defined the individual edge flows $x_{i}$ and the net flow vector $y$, we introduce utility functions for each. First, we denote the network utility by $U: \mathbf{R}^{n} \rightarrow \mathbf{R} \cup\{-\infty\}$, which maps the net flow vector $y$ to a utility value, $U(y)$. Infinite values denote constraints: any flow with $U(y)=-\infty$ is unacceptable. We also introduce a utility function for each edge, $V_{i}: \mathbf{R}^{n_{i}} \rightarrow \mathbf{R} \cup\{-\infty\}$, which maps the flow $x_{i}$ on edge $i$ to a utility, $V_{i}\left(x_{i}\right)$. We require that both $U$ and the $V_{i}$ are concave, nondecreasing functions. This restriction is not as strong as it may seem; we may also minimize convex nondecreasing cost functions with this framework. For example, we can minimize the convex nondecreasing cost function
$$
c(z)=\left\|(z)_{+}\right\|^{2}
$$
by maximizing the concave nondecreasing utility function
$$
U(y)=-\left\|(-y)_{+}\right\|^{2}
$$
and changing the sign convention accordingly.
Convex flow problem. The convex flow problem seeks to maximize the sum of the network utility and the individual edge utilities, subject to the constraints on the allowable flows:
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i}  \tag{2.3}\\
& x_{i} \in T_{i}, \quad i=1, \ldots, m
\end{array}
$$

Here, the variables are the edge flows $x_{i} \in \mathbf{R}^{n_{i}}$, for $i=1, \ldots, m$, and the net flows $y \in \mathbf{R}^{n}$. Each of these edges can be thought of as a subsystem with its own local utility function $V_{i}$. The individual edge flows $x_{i}$ are local variables, specific to the $i$ th subsystem. The overall system, on the other hand, has a utility that is a function of the net flows $y$, the global variable. As we will see in what follows, this structure naturally appears in many applications and lends itself nicely to parallelizable algorithms. Note that, because the objective is nondecreasing in all of its variables, a solution $\left\{x_{i}^{\star}\right\}$ to problem (2.3) will almost
always have $x_{i}^{\star}$ at the boundary of the feasible flow set $T_{i}$. If an $x_{i}^{\star}$ were in the interior, we could increase its entries without decreasing the objective value until we hit the boundary of the corresponding $T_{i}$, assuming some basic conditions on $T_{i}$ (e.g., $T_{i}$ does not contain a strictly positive ray). We will further explore this notion in §4.2.2.

\subsection*{2.1 Downward closure and monotonicity}

We say that a set $T \subseteq \mathbf{R}^{n}$ is downward closed if, for any $x \in T$ and $x^{\prime} \leq x$, we have $x^{\prime} \in T$. In other words, if a flow is feasible, then any smaller flow is also feasible. If $x^{\prime} \geq x$, we say that the flow $x^{\prime}$ dominates the flow $x$, since, under any nonnegative utility function, the flow $x^{\prime}$ is always at least as 'good' as $x$. In [DAE24a], the authors assumed that the functions $U$ and $\left\{V_{i}\right\}$ in the convex flow problem are nondecreasing. This assumption is, in fact, equivalent to the sets $\left\{T_{i}\right\}$ being downward closed in the following sense: if the sets $\left\{T_{i}\right\}$ are downward closed, then the functions $U$ and $\left\{V_{i}\right\}$ can be replaced with their nondecreasing concave envelopes without affecting the optimal objective value. Similarly, if the functions $U$ and $\left\{V_{i}\right\}$ are nondecreasing, then the sets $\left\{T_{i}\right\}$ can be replaced by their downward closures, i.e.,
$$
\tilde{T}_{i}=T_{i}-\mathbf{R}_{+}^{n}
$$
without affecting the objective value. This downward closedness property has a number of immediate and useful implications, and it will be important in the rest of this paper.

Example. As a simple example, consider a directed edge $i$ with maximum input capacity 1 that, when $w$ units of flow enter the edge, outputs $h(w)$ units of Flow. The corresponding set of allowable flows is
$$
T_{i}=\left\{z \in \mathbf{R}^{2} \mid-1 \leq z_{1} \leq 0 \text { and } z_{2} \leq h\left(-z_{1}\right)\right\}
$$

This set is easily verified to be closed and convex, as it is the intersection of two halfspaces and the hypograph of a concave function. The set is also bounded from above by the $\mathbf{1}$ vector, in the sense we assume in $\S 2$. Figure 2.2 shows a $T_{i}$ and its downward closure $\tilde{T}_{i}$. The downward closure clearly satisfies the same properties: it is closed, convex, and bounded.

Proof. Consider a finite solution $\left(y^{\star},\left\{x_{i}^{\star}\right\}\right)$ to (2.3). If the objective functions are nondecreasing, then $U\left(x^{\prime}\right) \leq U(x)$ for any $x^{\prime} \leq x$, and similarly for the functions $\left\{V_{i}\right\}$. If this solution is not at the boundary, i.e., if there exists some $x_{i}^{\star}$ in the relative interior of the corresponding $T_{i}$, then we can find a nonnegative direction $d \in \mathbf{R}^{n_{i}}$ such that $x_{i}^{\star}+t d \in T_{i}$ for some $t>0$. Since the objective functions are nondecreasing, this new point will have a objective value equal to the original solution. As a result, there exists a solution at the boundary, and we can replace the sets $\left\{T_{i}\right\}$ with their 'downward extension',
$$
\tilde{T}_{i}=T_{i}-\mathbf{R}_{+}^{n}
$$
without affecting the solution.
Conversely, if the sets are downward closed and the optimal value is finite, then, by the downward closure of the $T_{i}$, there does not exist a nonnegative direction $d$ such that, for

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-022.jpg?height=587&width=1647&top_left_y=236&top_left_x=240}
\captionsetup{labelformat=empty}
\caption{Figure 2.2: A set of allowable flows $T$ (left) and its downward closure $\tilde{T}$ (right) for a two-node directed edge that for input $w$ outputs $h(w)=w /(1+w)$ units of flow.}
\end{figure}
some $t>0, x_{i}^{\star}+t d \in T_{i}$ and the objective value is larger. (Otherwise, we could find a new point dominated by $x_{i}^{\star}$, i.e., in the downward closure of $T_{i}$, with a higher objective value.) Equivalently, all subgradients at this solution must be nonnegative: $\partial_{U}\left(y^{\star}\right) \subseteq \mathbf{R}_{+}^{n}$ and $\partial_{V_{i}}\left(x_{i}^{\star}\right) \subseteq \mathbf{R}_{+}^{n_{i}}$ for $i=1, \ldots, m$. This fact immediately suggests that there exists a solution on the boundary and we can replace the objective functions with their monotonic concave envelopes without affecting the solution. We will give an alternate proof of this equivalence in §4.2.2, after we have derived a dual problem for (2.3).

Redefining allowable flows. Due to this result, we can redefine a set of allowable flows $T$ as any set satisfying the following properties:
1. The set $T$ is closed and convex.
2. The set $T$ is downward closed: if $x \in T$ and $x^{\prime} \leq x$, then $x^{\prime} \in T$.
3. The set $T$ contains the zero element, $0 \in T$.

The three conditions imposed on the set of allowable flows have a natural interpretation. Convexity means that as more flow enters an edge, the marginal output does not increase. Downward closure means that positive flow (i.e., flow out of an edge) can be dissipated. This property often has a nice interpretation. In power systems, it means that we can dissipate power by, for example, by adding a resistive load. In financial markets, it means that we have the option to 'overpay' for an asset. The last condition simply means that we have the option to not use an edge.

\subsection*{2.2 A calculus of flows}

In this section, we discuss a number of properties that directly follow from the downward closure condition of the sets $\left\{T_{i}\right\}$ and their implications. Despite their equivalence, these
results lead us to assume downward closure of the sets $\left\{T_{i}\right\}$ rather than assuming a nondecreasing objective function, which is more common. Much of this section generalizes the authors' previous work in the context of automated market makers $[\mathrm{Ang}+23, \S 2]$. In the remainder of this section, we will drop subscripts for convenience.

Definition and interpretation. Recall that a set of allowable flows $T$ can be any set satisfying the following properties:
1. The set $T$ is closed and convex.
2. The set $T$ is downward closed: if $x \in T$ and $x^{\prime} \leq x$, then $x^{\prime} \in T$.
3. The set $T$ contains the zero vector: $0 \in T$.

The three conditions imposed on the set of allowable flows have a natural interpretation. Convexity means that as more flow enters an edge, the marginal output does not increase. Downward closure means that positive flow (i.e., flow out of an edge) can be dissipated. This property often has a nice interpretation. In power systems, it means that we can dissipate power by, for example, by adding a resistive load. In financial markets, it means that we have the option to 'overpay' for an asset. Finally, the last condition means that we need not use an edge. This assumption is not fundamental; we can always translate a set $T$ and absorb the translation into the utility functions. The assumption, however, simplifies some of the proofs.

\subsection*{2.2.1 Composition rules}

As a result of the downward closure condition, sets of allowable flows satisfy certain composition rules. Many of these rules follow directly from the calculus of convex sets [BV04, §2.3], for example, the intersection of two sets of allowable flows yields another set of allowable flows. We discuss a few important composition rules that will be useful in the rest of this paper below.

Nonnegative matrix multiplication. Multiplication of a set of allowable flows by a nonnegative matrix $A \in \mathbf{R}^{p \times k}$ with $(A)=\{0\}$, followed by taking the downward closure, results in another set of allowable flows:
$$
A T-\mathbf{R}_{+}^{p}=\left\{x \mid x \leq A x^{\prime} \text { for some } x^{\prime} \in T\right\} .
$$

This resulting set is downward closed by definition, and also closed and convex. Convexity follows from the fact that convexity is preserved under linear transforms [BV04, §2.3.2] and under downward closure. Closedness of the set follows from [Roc70, Theorem 9.1], as we require $A$ to be injective. This set has a nice interpretation: given some $x \in T$, each element of the vector $A x$ is a weighted 'meta-flow' with weights given by the rows of $A$.

Lifting. A a special case of nonnegative matrix multiplication, the lifting of a set of allowable flows into a larger space is also a set of allowable flows. Specifically, let $A$ be a selector matrix (as defined in (2.2)). Then the set $A T-\mathbf{R}_{+}^{k}$ is a set of allowable flows in $\mathbf{R}^{n}$. This set describes an edge that connects all vertices but only allows flow between a subset of them. We show an example of 'lifting' an edge in figure 2.3.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-024.jpg?height=394&width=471&top_left_y=545&top_left_x=335}
\captionsetup{labelformat=empty}
\caption{Figure 2.3: Multiplying the set $T$ (left) with the nonnegative matrix, followed by taking the downward closure (middle), results in another set of allowable flows (right).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-024.jpg?height=306&width=856&top_left_y=584&top_left_x=917}
\captionsetup{labelformat=empty}
\caption{Figure 2.3: Multiplying the set $T$ (left) with the nonnegative matrix, followed by taking the downward closure (middle), results in another set of allowable flows (right).}
\end{figure}

Set addition. Finally, under an additional boundedness assumption (see (2.1)), the Minkowski sum of allowable flow sets $T$ and $\tilde{T}$,
$$
T+\tilde{T}=\{x+\tilde{x} \mid x \in T, \tilde{x} \in \tilde{T}\},
$$
is also a set of allowable flows. Note that, for this composition rule, we need a boundedness condition on the sets $T$ and $\tilde{T}$ to ensure that the resulting set is closed. Specifically, we say that the set $T$ is bounded from above if there exists some $b$ such that $x \leq b \mathbf{1}$ for all $x \in T$. This condition means that a bounded input flow cannot produce infinite output flow. We can interpret this combined set as an aggregate edge that can use either of the two original edges. We provide an example in figure 2.4. Note that even if $T \cap \mathbf{R}_{+}^{n}=\{0\}$ and $\tilde{T} \cap \mathbf{R}_{+}^{n}=\{0\}$, the Minkowski sum $T+\tilde{T}$ may not satisfy this property. For example, consider the $T$ with corresponding edge gain function $h(w)=\sqrt{w}$.
![](https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-024.jpg?height=422&width=504&top_left_y=1849&top_left_x=242)
![](https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-024.jpg?height=422&width=503&top_left_y=1849&top_left_x=812)

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-024.jpg?height=420&width=502&top_left_y=1851&top_left_x=1381}
\captionsetup{labelformat=empty}
\caption{Figure 2.4: We take the Minkowski sum of the sets of allowable flows $T$ and $\tilde{T}$ for two directed edges with the form of that in figure 2.2 (left and middle) to obtain a new set of allowable flows $T+\tilde{T}$ that corresponds to an undirected edge (right).}
\end{figure}

Aggregate edges. Using the previous two rules, we can combine edges with possibly nonoverlapping incident vertices. Importantly, we can view the net flow vector $y$ in (2.3) as the flow over an 'aggregate edge' that connects all vertices with associated allowable flows
$$
T=\sum_{i=1}^{m} A T_{i}-\mathbf{R}_{+}^{n_{i}} .
$$

Thus, when the edge utility functions are equal to zero, the convex network flow problem (2.3) is equivalent to the following problem over one large aggregate edge:
$$
\begin{array}{ll}
\operatorname{maximize} & U(y) \\
\text { subject to } & y \in T .
\end{array}
$$

While this particular rewriting is not immediately useful, combining or splitting certain trading sets, for example those with the same incident vertices, can sometimes help us compute certain subproblems more quickly.

Example. Often, a directed edge between two nodes has a gain function defined in a piecewise manner. For example, consider a financial market between two assets given by an order book: sellers list the amount of an asset they are willing to sell for the other at a given price. We can view each 'tick' as an individual linear edge, which, when combined, define an aggregate edge corresponding to the entire orderbook. We provide a simple example in figure 2.5.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-025.jpg?height=793&width=1589&top_left_y=1433&top_left_x=244}
\captionsetup{labelformat=empty}
\caption{Figure 2.5: Each tick of the orderbook (left) corresponds to a linear edge with coefficient corresponding to the exchange rate (middle). These linear edges can be combined into an aggregate edge defining the entire orderbook (right).}
\end{figure}

\subsection*{2.3 The conic problem}

In this section, we will introduce what looks like a restriction of problem (2.3), which we will call the conic network flow problem, defined as
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i}  \tag{2.4}\\
& x_{i} \in K_{i}, \quad i=1, \ldots, m
\end{array}
$$

This problem set up is identical to that of (2.3), except that the sets $K_{i} \subseteq \mathbf{R}^{n_{i}}$, instead of being downward closed convex sets, are downward closed convex cones. A set $K_{i}$ is called a cone if it satisfies the following property: if $x \in K_{i}$, then, for any $\alpha \geq 0$, we must have that $\alpha x \in K_{i}$. We call any downward closed convex cone $K_{i}$ an allowable flow cone.

Certainly, every conic flow problem (2.4) is an instance of a convex network flow problem (2.3) as every downward closed convex cone is also, by definition, a downward closed convex set. In this section, we will show that the converse is also true: every instance of a convex network flow problem can be turned into an instance of a conic network flow problem. In this sense, problem (2.3) and problem (2.4) are equivalent. We will use the conic problem (2.4) for the remainder of this paper to give a number of important theoretical properties, extensions, and a duality result, all of which easily translate to the original (2.3), but are much simpler in the conic formulation.

\subsection*{2.3.1 Basic properties}

All of the composition rules presented in §2.2.1 for the allowable flow sets also hold for the allowable flow cones. More specifically, given two allowable flow cones (i.e., cones that are downward closed) summation, intersection, nonnegative scaling, and nonnegative matrix multiplication all yield another allowable flow cone.

Cone is nonpositive. One immediate consequence of the fact that $K \subseteq \mathbf{R}^{n}$ is both a cone and downwards closed is that either $K=\mathbf{R}^{n}$ or $K$ contains no strictly positive vectors; that is,
$$
K \cap \mathbf{R}_{++}^{n}=\varnothing
$$

To see this, let $x \in K$ be any element of $K$ that has only strictly positive entries $x>0$. Then for every $n$ vector $y \in \mathbf{R}^{n}$, there exists some $\alpha \geq 0$ such that $y \leq \alpha x$. Since $\alpha x$ is in $K$, as it is a cone, and $K$ is downward closed, then $y \in K$, as required.

Polar cone. As is standard in convex optimization, given a cone $K \subseteq \mathbf{R}^{n}$ there exists a polar cone, defined
$$
\begin{equation*}
K^{\circ}=\left\{y \in \mathbf{R}^{n} \mid y^{T} x \leq 0 \text { for all } x \in K\right\} . \tag{2.5}
\end{equation*}
$$

This cone $K^{\circ}$ is always a closed convex cone (even when $K$ is not). If $K$ is also a closed convex cone, then we have the following duality result $\left(-K^{\circ}\right)^{*}=K$; in other words, the dual of the negative polar cone (called the dual cone) is the original. If, in addition, the cone $K$
is a downward closed cone with any strictly negative element (i.e., there is some $x \in K$ with $x<0$ ), then we must have that
$$
K^{\circ} \cap-\mathbf{R}_{+}^{n}=\{0\}
$$
(A sufficient condition for this to hold is, $e . g$. , if the cone $K$ has nonempty interior, which is almost always the case in practice.)

\subsection*{2.3.2 Reduction}

It is clear that the conic problem (2.4) is a special case of the original problem (2.3). In this subsection, we will show that we can reduce any instance of the original problem to an instance of the conic problem.

High level outline. We begin with an instance of (2.3), which we write again for convenience:
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i} \\
& x_{i} \in T_{i}, \quad i=1, \ldots, m
\end{array}
$$

As in (2.3) we have some nondecreasing convex network utility function $U: \mathbf{R}^{n} \rightarrow \mathbf{R} \cup\{-\infty\}$, edge utility functions $V_{i}: \mathbf{R}^{n_{i}} \rightarrow \mathbf{R} \cup\{-\infty\}$, selector matrices $A_{i} \in \mathbf{R}^{n \times n_{i}}$, and downward closed sets $T_{i} \subseteq \mathbf{R}^{n_{i}}$. The variables are the edge flows $x_{i} \in \mathbf{R}^{n_{i}}$ and the network flows $y \in \mathbf{R}^{n}$. The goal will be to construct some (simple) nondecreasing edge utility functions $\tilde{V}_{i}: \mathbf{R}^{n_{i}+1} \rightarrow \mathbf{R} \cup\{-\infty\}$, selector matrices $\tilde{A} \in \mathbf{R}^{n \times\left(n_{i}+1\right)}$, and downward closed convex cones $\tilde{K}_{i} \subseteq \mathbf{R}^{n_{i}+1}$, such that any solution to the corresponding conic problem (2.4) over these new functions, matrices, and sets,
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} \tilde{V}_{i}\left(\tilde{x}_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} \tilde{A}_{i} \tilde{x}_{i} \\
& \tilde{x}_{i} \in \tilde{K}_{i}, \quad i=1, \ldots, m
\end{array}
$$
can be (easily) converted to a solution for the original problem (2.3). We do this process in two steps. First, we define a basic cone $\tilde{K}_{i}$ associated with each $T_{i}$, which is essentially the perspective transformation of $T_{i}$, done in such a way as to ensure that $\tilde{K}_{i}$ is downward closed. We then show that any solution over this cone, with an additional constraint, always corresponds to a solution to the original set. Finally, we add this constraint to the objective as an extra term in the edge cost $\tilde{V}_{i}$.

\section*{Flow cone}

We define the flow cone corresponding to a set $T_{i} \subseteq \mathbf{R}^{n_{i}}$ as
$$
\begin{equation*}
\tilde{K}_{i}=\mathbf{c l}\left\{(x,-\lambda) \in \mathbf{R}^{n_{i}} \times \mathbf{R} \mid x / \lambda \in T_{i}, \lambda>0\right\} \tag{2.6}
\end{equation*}
$$
where $\mathbf{c l}$ denotes the closure of a set. This definition is just the perspective transformation of the set $T_{i}$, with a sign change in the last argument. This set is closed (by definition) and convex (see [BV04, §2.3.3]). It is not hard to show that the set is also downward closed, which we do next.

Recovering the set. We make use of the following (perhaps obvious) observation repeatedly. The set $T_{i}$ can be easily recovered from the cone $\tilde{K}_{i}$ by restricting the last coordinate to be equal to -1 ; that is,
$$
\begin{equation*}
T_{i}=\left\{x \in \mathbf{R}^{n_{i}} \mid(x,-1) \in \tilde{K}_{i}\right\} \tag{2.7}
\end{equation*}
$$
which follows by definition.

Boundary. From the flow cone, we may define a homogenous, nondecreasing, convex function
$$
\varphi(x)=\min \{\lambda \geq 0 \mid(x,-\lambda) \in K\} .
$$

Equivalently, we may write $\varphi$ as the Minkowski functional
$$
\varphi(x)=\inf \{\lambda>0 \mid x / \lambda \in T\},
$$

This function's one-level set, $\varphi(x)=1$ parameterizes the boundary of $T$ and is sometimes useful in practical applications (see, for example [Dia+23]).

Scaling property. An important property of a set of allowable flows is that, for any flow set $T_{i}$, if $x \in T_{i}$, then $\alpha x \in T_{i}$ for any $0 \leq \alpha \leq 1$. The proof is nearly immediate by the convexity of $T_{i}$ and the fact that $0 \in T_{i}$ by noting that
$$
\begin{equation*}
\alpha x=\alpha x+(1-\alpha) 0 \in T_{i} . \tag{2.8}
\end{equation*}
$$

Downward closure. We will now prove that the flow cone $K_{i}$ is downward closed. In other words, we will show that, for any $(x, \lambda) \in K_{i}$ and $\left(x^{\prime}, \lambda^{\prime}\right)$ with $x^{\prime} \leq x$ and $\lambda^{\prime} \leq \lambda$, then $\left(x^{\prime}, \lambda^{\prime}\right) \in K_{i}$. We will first show that $(x, \lambda) \in K_{i}$ implies that $\left(x, \lambda^{\prime}\right) \in K_{i}$. To see this, note that, if $(x, \lambda) \in K_{i}$, then there exists a sequence $\left\{\left(x_{k}, \lambda_{k}\right)\right\}$ such that $x_{k} /\left(-\lambda_{k}\right) \in T_{i}$ and $\lambda_{k}<0$ for each $k$, while the sequences converge in that $x_{k} \rightarrow x$ and $\lambda_{k} \rightarrow \lambda$ by the definition (2.6). If $\lambda=\lambda^{\prime}$ then obviously $\left(x, \lambda^{\prime}\right) \in K_{i}$, so assume that $\lambda^{\prime}<\lambda$. In this case, there exists some $k^{\prime}$ such that, $\lambda_{k} \geq \lambda^{\prime}$ for all $k \geq k^{\prime}$. But, since
$$
\frac{x_{k}}{-\lambda_{k}} \in T_{i},
$$
for all $k \geq k^{\prime}$ by definition, then, since $0 \leq \lambda_{k} / \lambda^{\prime} \leq 1$ we have that, using the scaling property (2.8):
$$
\frac{\lambda_{k}}{\lambda^{\prime}} \frac{x_{k}}{-\lambda_{k}} \in T_{i},
$$
so $x_{k} /\left(-\lambda^{\prime}\right) \in T_{i}$ for each $k \geq k^{\prime}$. Since $x_{k} \rightarrow x$ and $T_{i}$ is closed, then $x /\left(-\lambda^{\prime}\right) \in T_{i}$, so, by definition $\left(x, \lambda^{\prime}\right) \in K_{i}$, as required. The final question, if $x^{\prime} \leq x$ then $\left(x^{\prime}, \lambda^{\prime}\right) \in K_{i}$ is easy: if $\lambda^{\prime}<0$ then $x /\left(-\lambda^{\prime}\right) \in T_{i}$ implies that $x^{\prime} /\left(-\lambda^{\prime}\right) \in T_{i}$ by the downward closure of $T_{i}$, so $\left(x^{\prime}, \lambda^{\prime}\right) \in K_{i}$. If $\lambda^{\prime}=0$ then we know that $\lambda=0$. Set $\delta=x^{\prime}-x$ and note that $\delta \leq 0$ since $x^{\prime} \leq x$. This means that, for the sequence $\lambda_{k}<0$ with $\lambda_{k} \rightarrow 0$, we have
$$
\frac{x_{k}+\delta}{-\lambda_{k}} \in T_{i},
$$
since $x_{k} /\left(-\lambda_{k}\right) \in T_{i}$ and $\left(x_{k}+\delta\right) /\left(-\lambda_{k}\right) \leq x_{k} /\left(-\lambda_{k}\right)$ because $\delta \leq 0$ and $T_{i}$ is downward closed. But, since $x_{k} \rightarrow x$, then $x_{k}+\delta \rightarrow x+\delta=x^{\prime}$. By the definition of (2.6), then we have that $(x+\delta, 0) \in K_{i}$, so $\left(x^{\prime}, \lambda^{\prime}\right)=(x+\delta, 0) \in K_{i}$ as required.

Dominating points. We can rewrite the above observation in the following (slightly more useful) way: given any $\lambda^{\prime}$ such that $-1 \leq \lambda^{\prime} \leq 0$ then
$$
\begin{equation*}
\left(x, \lambda^{\prime}\right) \in \tilde{K}_{i} \quad \text { implies } \quad(x,-1) \in \tilde{K}_{i} . \tag{2.9}
\end{equation*}
$$

\section*{Rewriting via the flow cone}

We start with the original problem (2.3). Using the cone defined previously equation (2.7), we rewrite the original problem using the flow cone as
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i} \\
& \left(x_{i}, \lambda_{i}\right) \in \tilde{K}_{i}, \quad i=1, \ldots, m \\
& \lambda_{i}=-1
\end{array}
$$

Since $U$ and $V_{i}$ are both nondecreasing in the $x_{i}$, the relaxation
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i}  \tag{2.10}\\
& \left(x_{i}, \lambda_{i}\right) \in \tilde{K}_{i}, \quad i=1, \ldots, m \\
& \lambda_{i} \geq-1, \quad i=1, \ldots, m
\end{array}
$$
is exact by the 'dominating points' result (2.9): any solution with ( $x_{i}, \lambda_{i}$ ) may be replaced with a solution $\left(x_{i},-1\right)$ which is also feasible. Indeed, in many cases, such as when the set $T_{i}$ is locally strictly concave around 0 , one can show that if $\lambda>-1$, there exists a strictly dominating point $x_{i}^{\prime}$ such that $x_{i}^{\prime}>x_{i}$ and $\left(x_{i}^{\prime},-1\right) \in \tilde{K}_{i}$, so choosing $\lambda_{i}>-1$ is never optimal.

\section*{Final rewriting}

Finally, we take the conic relaxation given in (2.10), which, due to the constraint $\lambda_{i} \geq-1$, is not quite a conic flow problem (2.4), and replace the matrices $A_{i}$, edge cost functions $V_{i}$, and variables $\left(x_{i}, \lambda_{i}\right)$ to get a problem of the required form.

Constraint. The first part is easy: let $I(z)=0$ if $z \geq-1$ and $+\infty$ otherwise be the nonnegative indicator function for a scalar. Note that $I$ is nonincreasing so $-I$ is nondecreasing, which means we can rewrite (2.10) by pulling the constraint into the objective
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right)-I\left(\lambda_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i}  \tag{2.11}\\
& \left(x_{i}, \lambda_{i}\right) \in \tilde{K}_{i}, \quad i=1, \ldots, m
\end{array}
$$

We then define
$$
\tilde{V}\left(x_{i}, \lambda_{i}\right)=V_{i}\left(x_{i}\right)-I\left(\lambda_{i}\right)
$$
for each edge $i=1, \ldots, m$, which we note is a nondecreasing function in each of its arguments, to get
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} \tilde{V}_{i}\left(x_{i}, \lambda_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i} \\
& \left(x_{i}, \lambda_{i}\right) \in \tilde{K}_{i}, \quad i=1, \ldots, m
\end{array}
$$

This is very close to the form of (2.4) except for the linear constraint.

Matrices and relabeling. Finally, we define the matrix
$$
\tilde{A}_{i}=\left[\begin{array}{ll}
A_{i} & 0
\end{array}\right]
$$
which is just the matrix $A_{i}$ with an additional all-zeros column. Setting $\tilde{x}_{i}=\left(x_{i}, \lambda_{i}\right)$ gives the final result:
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} \tilde{V}_{i}\left(\tilde{x}_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} \tilde{A}_{i} \tilde{x}_{i}  \tag{2.12}\\
& \tilde{x}_{i} \in \tilde{K}_{i}, \quad i=1, \ldots, m
\end{array}
$$

This problem is exactly of the form of the conic flow problem (2.4), as required.

Discussion. As a side note, instead of introducing an all-zeros column to $A_{i}$, we can create a matrix $\tilde{A}_{i} \in \mathbf{R}^{(n+1) \times\left(n_{i}+1\right)}$, which has one additional row and column, and set
$$
\tilde{A}_{i}=\left[\begin{array}{cc}
A_{i} & 0 \\
0 & 1
\end{array}\right]
$$

We can then define a new network utility function $\tilde{U}(y, \lambda)=U(y)$, which simply ignores the last entry of the new net flow vector.

\section*{Chapter 3}

\section*{Applications}

In this section, we give a number of applications of the convex flow problem (2.3). We first show that many classic optimization problems in graph theory are special cases of this problem. Then, we show that the convex flow problem models problems in a variety of fields including power systems, communications, economics, and finance, among others. We start with simple special cases and gradually build up to those that are firmly outside the traditional network flows literature.

\subsection*{3.1 Maximum flow and friends}

In this section, we show that many classic network flow problems are special cases of problem (2.3). We begin with a standard setup that will be used for the rest of this section.

Edge flows. We consider a directed graph with $m$ edges and $n$ nodes, which we assume to be connected. Recall that we denote the flow over edge $i$ by the vector $x_{i} \in \mathbf{R}^{2}$. We assume that edge $i$ 's flow has upper bound $b_{i} \geq 0$, so the set of allowable flows is
$$
\begin{equation*}
T_{i}=\left\{z \in \mathbf{R}^{2} \mid 0 \leq z_{2} \leq b_{i} \text { and } z_{1}+z_{2}=0\right\} \tag{3.1}
\end{equation*}
$$

We show this edge in figure 3.1. With this framework, it is easy to see how gain factors or other transformations can be easily incorporated into the problem. For example, we can instead require that $\alpha z_{1}+z_{2}=0$ where $\alpha>0$ is some gain or loss factor. Note that if the graph is instead undirected, with each set of two directed edges replaced by one undirected edge, the allowable flows for each pair of directed edges can be combined into the set
$$
T_{i}=\left\{z \in \mathbf{R}^{2} \mid z \leq b_{i} \mathbf{1} \text { and } z_{1}+z_{2}=0\right\}
$$
which is the Minkowski sum of the two allowable flows in the directed case, one for each direction. For what follows, we only consider directed graphs, but the extension to the undirected case is straightforward.

Net flow. To connect these edge flows to the net flow we use the matrices $A_{i} \in\{0,1\}^{n \times 2}$ for each edge $i=1, \ldots, m$ such that, if edge $i$ connects node $j$ to node $k$ (assuming the

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-032.jpg?height=544&width=1395&top_left_y=236&top_left_x=365}
\captionsetup{labelformat=empty}
\caption{Figure 3.1: A simple directed (hyper)edge $e_{i}$ connecting nodes $v_{1}$ and $v_{2}$ (left) and the corresponding set of allowable flows $T_{i}$ (right).}
\end{figure}
direction of the edge is from node $j$ to node $k$ ), then we have
$$
A_{i}=\left[\begin{array}{cc}
\mid & \mid  \tag{3.2}\\
e_{j} & e_{k} \\
\mid & \mid
\end{array}\right] .
$$

Using these matrices, we write the net flow through the network as the sum of the edge flows:
$$
y=\sum_{i=1}^{m} A_{i} x_{i}
$$

Conservation laws. One important consequence of the definition of the allowable flows $T_{i}$ is that there is a corresponding local conservation law: for any allowable flow $x_{i} \in T_{i}$, we have that
$$
\mathbf{1}^{T} x_{i}=\left(x_{i}\right)_{1}+\left(x_{i}\right)_{2}=0
$$
by definition of the set $T_{i}$. Since the $A_{i}$ matrices are simply selector matrices, we therefore have that $\mathbf{1}^{T} A_{i} x_{i}=0$ whenever $x_{i} \in T_{i}$, which means that we can turn the local conservation law above into a global conservation law:
$$
\begin{equation*}
\mathbf{1}^{T} y=\sum_{i=1}^{m} \mathbf{1}^{T} A_{i} x_{i}=0 \tag{3.3}
\end{equation*}
$$
where $y$ is the corresponding net flow, for any set of allowable flows $x_{i} \in T_{i}$, for $i=1, \ldots, m$. We will use this fact to show that feasible flows are, indeed, flows through the network in the 'usual' sense. Conversely, we can find conservation laws for a given convex flow problem, which we discuss next.

Finding conservation laws. We define the set of conservation laws for an instance of the convex flow problem as
$$
C=\left\{c \in \mathbf{R}^{n} \mid c^{T}\left(\sum_{i=1}^{m} A_{i} x_{i}\right) \geq 0 \text { for all } x_{i} \in T_{i}\right\} .
$$

Our goal is to find vectors in this set. Define the dual cone for a set $S$ as
$$
K(S)=\left\{y \mid y^{T} x \geq 0 \text { for all } x \in S\right\}
$$

A sufficient condition for a vector $c$ to be in $C$ is that
$$
c \in \bigcap_{i=1}^{m} K\left(A_{i} T_{i}\right) .
$$
(Of course, in general, this condition is not necessary.) We can then find conservation laws by solving the convex optimization problem
$$
\begin{array}{ll}
\text { find } & c \\
\text { subject to } & c^{T} z=1 \\
& c \in \bigcap_{i=1}^{m} K\left(A_{i} T_{i}\right),
\end{array}
$$
where $z$ is some nonzero vector, for example, sampled from the standard normal distribution.

\subsection*{3.1.1 Maximum flow}

Given a directed graph, the maximum flow problem seeks to find the maximum amount of flow that can be sent from a designated source node to a designated sink node. The problem can model many different situations, including transportation network routing, matching, and resource allocation, among others. It dates back to the work of Harris and Ross [HR55] to model Soviet railway networks in a report written for the US Air Force and declassified in 1999, at the request of Schrijver [Sch02]. While well-known to be a linear program at the time [FF56] (and therefore solvable with the simplex method), specialized methods were quickly developed [FF57]. The maximum flow problem has been extensively studied by the operations research and computer science communities since then.

Flow conservation. Relabeling the graph such that the source node is node 1 and the sink node is node $n$, we write the net flow conservation constraints as the set
$$
\begin{equation*}
S=\left\{y \in \mathbf{R}^{n} \mid y_{1}+y_{n} \geq 0, \quad y_{j} \geq 0 \quad \text { for all } j \neq 1, n\right\} \tag{3.4}
\end{equation*}
$$

Note that this set $S$ is convex as it is the intersection of halfspaces (each of which is convex), and its corresponding indicator function, written
$$
I_{S}(y)= \begin{cases}0 & y \in S \\ +\infty & \text { otherwise }\end{cases}
$$
is therefore also convex. This indicator function is nonincreasing in that, if $y^{\prime} \geq y$ then $I_{S}\left(y^{\prime}\right) \leq I_{S}(y)$ by definition of the set $S$. Thus, its negation, $-I_{S}$, is nondecreasing and concave.

Problem formulation. The network utility function in the maximum flow problem is to maximize the flow into the terminal node while respecting the flow conversation constraints:
$$
U(y)=y_{n}-I_{S}(y)
$$

From the previous discussion, this utility function is concave and nondecreasing. We set the edge utility functions to be zero, $V_{i}=0$ for all $i=1, \ldots, m$, to recover the maximum flow problem (see, for example, [Ber98, Example 1.3]) in our framework:
$$
\begin{array}{ll}
\operatorname{maximize} & y_{n}-I_{S}(y) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i}  \tag{3.5}\\
& x_{i} \in T_{i}, \quad i=1, \ldots, m
\end{array}
$$
where the sets $\left\{T_{i}\right\}$ are the feasible flow sets (3.1) and may be either directed or undirected.
Problem properties. Any feasible point (i.e., one that satisfies the constraints and has finite objective value) is a flow in that the net flow through node $j$ is zero $\left(y_{j}=0\right)$ for any node that is not the source, $j=1$, or the sink, $j=n$. To see this, note that, for any $j \neq 1, n$, if $y \in S$, then
$$
\mathbf{1}^{T} y \geq y_{1}+y_{n}+y_{j} \geq y_{j} \geq 0
$$

The first and third inequalities follow from the fact that $y \in S$ means that $y_{j} \geq 0$ for all $j \neq 1, n$; the second follows from the fact that $y_{1}+y_{n} \geq 0$ from the definition of $S$ as well. From the conservation law (3.3), we know that $\mathbf{1}^{T} y=0$, so $y_{j}=0$ for every node $j$ that is not the source nor the sink. Therefore, $y$ is a flow in the 'usual' sense. A similar proof shows that $-y_{1}=y_{n}$; i.e., the amount provided by the source is the amount dissipated by the sink. Since we are maximizing the total amount dissipated $y_{n}$, subject to the provided capacity and flow constraints, the problem above corresponds exactly to the standard maximum flow problem.

\subsection*{3.1.2 Minimum cost flow}

The minimum cost flow problem seeks to find the cheapest way to route a given amount of flow between specified source and sink nodes. We consider the same setup as above, but with two modifications: first, we fix the value of the flow from node 1 to node $n$ to be at least some value $v \geq 0$; and second, we introduce a convex, nondecreasing cost function for each edge $i$, denoted $c_{i}: \mathbf{R}_{+} \rightarrow \mathbf{R}_{+}$, which maps the flow on this edge to a cost. We modify the flow conservation constraints to be (cf., (3.4))
$$
\tilde{S}=\left\{y \mid y_{n} \geq v, \quad y_{1}+y_{n} \geq 0, \quad y_{j} \geq 0 \quad \text { for all } j \neq 1, n\right\}
$$

Much like the previous, the negative indicator of this set, $-I_{\tilde{S}}$, is a concave, nondecreasing function. We take the edge flow utility function $V_{i}$ to be
$$
V_{i}\left(x_{i}\right)=-c_{i}\left(-\left(x_{i}\right)_{1}\right)
$$
which is a concave nondecreasing function of $x_{i}$. (Recall that $\left(x_{i}\right)_{1} \leq 0$. We provide an example in figure 3.2.) Modifying the network utility function to be the indicator over this new set $\tilde{S}$,
$$
U(y)=-I_{\tilde{S}}(y),
$$
we recover the minimum cost flow problem in our framework:
$$
\begin{array}{ll}
\text { maximize } & -I_{\tilde{S}}(y)+\sum_{i=1}^{m}-c_{i}\left(-\left(x_{i}\right)_{1}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i} \\
& x_{i} \in T_{i}, \quad i=1, \ldots, m
\end{array}
$$

Here, as before, the sets $\left\{T_{i}\right\}$ are the directed feasible flow sets defined in (3.1).

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-035.jpg?height=456&width=529&top_left_y=872&top_left_x=423}
\captionsetup{labelformat=empty}
\caption{Figure 3.2: An example convex nondecreasing cost function $c(w)=w^{2}$ for $w \geq 0$ (left) and its corresponding concave, nondecreasing edge utility function $V$ (right).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-035.jpg?height=433&width=527&top_left_y=893&top_left_x=1147}
\captionsetup{labelformat=empty}
\caption{Figure 3.2: An example convex nondecreasing cost function $c(w)=w^{2}$ for $w \geq 0$ (left) and its corresponding concave, nondecreasing edge utility function $V$ (right).}
\end{figure}

\subsection*{3.1.3 Concave edge gains}

We can generalize the maximum flow problem and the minimum cost flow problem to include concave, nondecreasing edge input-output functions, as in [Shi06; Vég14], by modifying the sets of feasible flows. We denote the edge input-output functions by $\gamma_{i}: \mathbf{R}_{+} \rightarrow \mathbf{R}_{+}$. (For convenience, negative arguments to $\gamma_{i}$ are equal to negative infinity.) If $w$ units of flow enter edge $i$, then $\gamma_{i}(w)$ units of flow leave edge $i$. In this case, we can write the set of allowable flows for each edge to be
$$
T_{i}=\left\{z \in \mathbf{R}^{2} \mid z_{2} \leq \gamma_{i}\left(-z_{1}\right)\right\} .
$$

We provide an example in figure 3.3. The inequality has the following interpretation: the magnitude of the flow out of edge $i$, given by $z_{2} \geq 0$, can be any value not exceeding $\gamma_{i}\left(-z_{1}\right)$; however, we can 'destroy' flow. From the problem properties presented in section 2, there exists a solution such that this inequality is tight, since the utility function $U$ is nondecreasing. In other words, we can find a set of feasible flows $\left\{x_{i}\right\}$ such that
$$
\left(x_{i}\right)_{2}=\gamma_{i}\left(-\left(x_{i}\right)_{1}\right),
$$
for all edges $i=1, \ldots, m$.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-036.jpg?height=462&width=1277&top_left_y=232&top_left_x=416}
\captionsetup{labelformat=empty}
\caption{Figure 3.3: An example concave edge gain function $\gamma(w)=\sqrt{w}$ (left) and the corresponding allowable flows (right).}
\end{figure}

\subsection*{3.1.4 Multi-commodity flows}

Thus far, all the flows have been of the same type. Here, we show that the multi-commodity flow problem, which seeks to route $K$ different commodities through a network in an optimal way, is also a special case of the convex flow problem. We denote the flow of these commodities over an edge $i$ by $x_{i} \in \mathbf{R}^{2 K}$, where the first 2 elements denote the flow of the first commodity through edge $i$, the next 2 elements denote the flow of the second, and so on. The set $T_{i}$ then allows us to specify joint constraints on these flows. For example, we can model a total flow capacity by the set
$$
T_{i}=\left\{x \in \mathbf{R}^{K} \mid \sum_{k=1}^{K} w_{k} x_{2 k} \leq b_{i}, 0 \leq x_{2 k}, \text { and } x_{2 k}=-x_{2 k-1} \text { for } k=1,2, \ldots, K\right\},
$$
where $b_{i}$ denotes the capacity of edge $i$ and $w_{k}$ denotes the capacity required per unit of commodity $k$. In other words, each commodity $k$ has a per-unit 'weight' of $w_{k}$, and the maximum weighted capacity through edge $i$ is $b_{i}$. If $K=1$, then this set of allowable flows reduces to the original definition (3.1). Additionally, note that $T_{i}$ is still a polyhedral set, but more complicated convex constraints may be added as well.

We denote the net flows at each node by a vector $y \in \mathbf{R}^{n K}$, where the first $K$ elements denote the net flows of the first commodity, the next $K$ elements denote the net flows of the second, and so on, while the $A_{i}$ matrices map the local flows of each good to the corresponding indices in $y$. For example, if edge $i$ connects node $j$ to node $k$, the edge would have the associated matrix $A_{i} \in \mathbf{R}^{n K \times 2 K}$ given by
$$
A_{i}=\left[\begin{array}{ccccccc}
\mid & \mid & \mid & \mid & & \mid & \mid \\
e_{j} & e_{k} & e_{j+n} & e_{k+n} & \cdots & e_{j+(K-1) n} & e_{k+(K-1) n} \\
\mid & \mid & \mid & \mid & & \mid & \mid
\end{array}\right] .
$$

The problem is now analogous to those in the previous sections, only with $y$ and $x_{i}$ having larger dimension and $T_{i}$ modified as described above.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-037.jpg?height=410&width=1548&top_left_y=240&top_left_x=285}
\captionsetup{labelformat=empty}
\caption{Figure 3.4: The power loss function (left), the corresponding power output (middle), and the corresponding set of allowable flows (right).}
\end{figure}

\subsection*{3.2 Optimal power flow}

The optimal power flow problem [WWS13] seeks a cost-minimizing plan to generate power satisfying demand in each region. We consider a network of $m$ transmission lines (edges) between $n$ regions (nodes). We assume that the region-transmission line graph is directed for simplicity.

Line flows. When power is transmitted along a line, the line heats up and, as a result, dissipates power. As greater amounts of power are transmitted along this line, the line further heats up, which, in turn, causes it to dissipate even more power. We model this dissipation as a convex function of the power transmitted, which captures the fact that the dissipation increases as the power transmitted increases. We use the logarithmic power loss function from [Stu19, §2.1.3]. With this loss function, the 'transport model' optimal power flow solution matches that of the more-complicated 'DC model', assuming a uniform line material. (See [Stu19, §2] for details and discussion.) The logarithmic loss function is given by
$$
\ell_{i}(w)=\alpha_{i}\left(\log \left(1+\exp \left(\beta_{i} w\right)\right)-\log 2\right)-2 w,
$$
where $\alpha_{i}$ and $\beta_{i}$ are known constants and $\alpha_{i} \beta_{i}=4$ for each line $i$. This function can be easily verified to be convex, increasing, and have $\ell_{i}(0)=0$. The power output of a line with input $w$ can then be written as $w-\ell(w)$. We also introduce capacity constraints for each line $i$, given by $b_{i}$. Taken together, for a given line $i$, the power flow $x_{i}$ must lie within the set
$$
\begin{equation*}
T_{i}=\left\{z \in \mathbf{R}^{2} \mid-b_{i} \leq z_{1} \leq 0 \text { and } z_{2} \leq-z_{1}-\ell_{i}\left(-z_{1}\right)\right\}, \quad i=1, \ldots, m . \tag{3.6}
\end{equation*}
$$

This set is convex, as it is the intersection of two halfspaces and the epigraph of a convex function. Note that we relaxed the line flow constraint to an inequality. This inequality has the following physical interpretation: we may dissipate additional power along the line (for example, by adding a resistive load), but in general we expect this inequality to hold with equality, as discussed in §2. Figure 3.4 shows a loss function and the corresponding edge's allowable flows.

Net flows. Each region $i=1, \ldots, n$ demands $d_{i}$ units of power. In addition, region $i$ can generate power $p_{i}$ at cost $c_{i}: \mathbf{R} \rightarrow \mathbf{R}_{+} \cup\{\infty\}$, where infinite values denote constraints (e.g.,
a region may have a maximum power generation capacity). We assume that $c_{i}$ is convex and nondecreasing, with $c_{i}\left(p_{i}\right)=0$ for $p_{i} \leq 0$ (i.e., we can dissipate power at zero cost). Similarly to the max flow problem in §3.1, we take the indexing matrices $A_{i}$ as defined in (3.2). To meet demand, we must have that
$$
d=p+y, \quad \text { where } \quad y=\sum_{i=1}^{m} A_{i} x_{i}
$$

In other words, the power produced, plus the net flow of power, must satisfy the demand in each region. We write the network utility function as
$$
\begin{equation*}
U(y)=\sum_{i=1}^{n}-c_{i}\left(d_{i}-y_{i}\right) \tag{3.7}
\end{equation*}
$$

Since each $c_{i}$ is convex and nondecreasing, the utility function $U$ is concave and nondecreasing in $y$. This problem can then be cast as a special case of the convex flow problem (2.3):
$$
\begin{array}{ll}
\operatorname{maximize} & \sum_{i=1}^{n}-c_{i}\left(d_{i}-y_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i} \\
& x_{i} \in T_{i}, \quad i=1, \ldots, m
\end{array}
$$
with the same variables $\left\{x_{i}\right\}$ and $y$, zero edge utilities ( $V_{i}=0$ ), and the feasible flow sets $T_{i}$ given in (3.6).

Extensions and related problems. This model can be extended in a variety of ways. For example, a region may have some joint capacity over all the power it outputs. When there are constraints such as these resulting from the interactions between edge flows, a hypergraph model is more appropriate. On the other hand, simple two-edge concave flows model behavior in number of other types of networks: in queuing networks, throughput is a concave function of the input due to convex delays [BG92, §5.4]; similarly, in routing games [Rou07, §18], a convex cost function often implies a concave throughput; in perishable product supply chains, such as those for produce, increased volume leads to increased spoilage [NB22, §2.3]; and in reservoir networks [Ber98, §8.1], seepage may increase as volume increases. Our framework not only can model these problems, but also allows us to easily extend them to more complicated settings.

\subsection*{3.3 Routing and resource allocation in wireless networks}

In many applications, standard graph edges do not accurately capture interactions between multiple flows coming from a single node - there may be joint, possibly nonlinear, constraints on all the flows involving this node. To represent these constraints in our problem, we make use of the fact that an edge may connect more than two nodes in (2.3). In this section, we illustrate this structure through the problem of jointly optimizing the data flows and the power allocations for a wireless network, heavily inspired by the formulation of this problem in [XJB04].

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-039.jpg?height=317&width=1045&top_left_y=242&top_left_x=539}
\captionsetup{labelformat=empty}
\caption{Figure 3.5: Wireless ad-hoc network. The (outgoing) hyperedge associated with user $u$ is shown in blue, and the corresponding set of outgoing neighbors $O_{u}$ contains user $v$ and the two base stations, $b_{1}$ and $b_{2}$.}
\end{figure}

Data flows. We represent the topology of a data network by a directed graph with $n$ nodes and $m=n$ edges: one for each node. We want to route traffic from a particular source to a particular destination in the network. (This model can be easily extended to handle multiple source-destination pairs, potentially with different latency or bandwidth constraints, using the multi-commodity flow ideas discussed in §3.1.4.) We model the network with a hypergraph, where edge $i=1, \ldots, n$ is associated with node $i$ and connects $i$ to all its outgoing neighbors, which we denote by the set $O_{i}$, as shown in figure 3.5. (In other words, if $j \in O_{i}$, then node $j$ is a neighbor of node $i$.) On each edge, we have a rate at which we transmit data, denoted by a vector $x_{i} \in \mathbf{R}^{\left|O_{i}\right|+1}$, where the $k$ th element of $x_{i}$ denotes the rate from node $i$ to its $k$ th outgoing neighbor and the last component is the total outgoing rate from node $i$. The net flow vector $y \in \mathbf{R}^{n}$ can be written as
$$
y=\sum_{i=1}^{n} A_{i} x_{i},
$$
for indexing matrices $A_{i}$ in $\mathbf{R}^{n \times\left(\left|O_{i}\right|+1\right)}$, given by
$$
A_{i}=\left[\begin{array}{cccc}
\mid & & \mid & \mid \\
e_{O_{i 1}} & \ldots & e_{O_{i\left|O_{i}\right|}} & e_{i} \\
\mid & & \mid & \mid
\end{array}\right],
$$
where $O_{i k}$ denotes the $k$ th neighbor of $O_{i}$ in any order fixed ahead of time.

Communications constraints. Hyperedges allow us to more easily model the communication constraints in the examples of [XJB04]. We associate some communication variables with each edge $i$. These variables might be, for example, power allocations, bandwidth allocations, or time slot allocations. We assume that the joint constraints on the transmission rate and the communication variables are some convex set. For example, take the communication variables to be ( $p_{i}, w_{i}$ ), where $p \in \mathbf{R}^{\left|O_{i}\right|}$ is a vector of power allocations and $w \in \mathbf{R}^{\left|O_{i}\right|}$ is a vector of bandwidth allocations to each of node $i$ 's outgoing neighbors. We may have maximum power and bandwidth constraints, given by $p_{i}^{\text {max }}$ and $w_{i}^{\text {max }}$, so the set of feasible powers and bandwidths is
$$
P_{i}=\left\{(p, w) \in \mathbf{R}^{\left|O_{i}\right|} \times \mathbf{R}^{\left|O_{i}\right|} \mid p \geq 0, w \geq 0, \mathbf{1}^{T} p \leq p^{\max }, \mathbf{1}^{T} w \leq w^{\max }\right\} .
$$

These communication variables determine the rate at which node $i$ can transmit data to its neighbors. For example, in the Gaussian broadcast channel with frequency division multiple access, this rate is governed by the Shannon capacity of a Gaussian channel [Sha48]. The set of allowable flows can be written as
$$
T_{i}=\left\{(z, t) \in \mathbf{R}^{\left|O_{i}\right|} \times \mathbf{R} \mid \mathbf{1}^{T} z=-t, \quad z \leq w \circ \log _{2}\left(\mathbf{1}+\frac{p}{\sigma w}\right), \quad(p, w) \in P_{i}\right\}
$$
where $\sigma \in \mathbf{R}_{+}^{n}$ is a parameter that denotes the average power of the noise in each channel, the logarithm and division, along with the inequality, are applied elementwise, and ∘ denotes the elementwise (Hadamard) product. The set $T_{i}$ is a convex set, as the logarithm is a concave function and $w_{k} \log \left(1+p_{k} / \sigma w_{k}\right)$, viewed as a function over the $k$ th element of each of the communication variables ( $p, w$ ), is the perspective transformation of $\log \left(1+p_{k} / \sigma\right)$, viewed as a function over $p_{k}$, which preserves concavity [BV04, §3.2.6]. The remaining sets are all affine or polyhedral, and intersections of convex sets are convex, which gives the final result.

Importantly, the communication variables (here, the power allocations $p$ and bandwidth allocations $w$ ) can be private to a node $i$; the optimizer only cares about the resulting public data flow rates $x_{i} \in T_{i}$. This structure not only simplifies the problem formulation but also hints at efficient, decentralized algorithms to solve this problem. We note that the hypergraph model allows us to also consider the general multicast case as well.

The optimization problem. Without loss of generality, denote the source node by 1 and the sink node by $n$. We may simply want to maximize the rate of data from the source to the sink, in which case we can take the network utility function to be
$$
U(y)=y_{n}-I_{S}(y)
$$
where the flow conversation constraints $S$ are the same as those of the classic maximum flow problem, defined in (3.4). We may also use the functions $V_{i}$ to include utilities or costs associated with the transmission of data by node $i$. We can include communication variables in the objective as well by simply redefining the allowable flows $T_{i}$ to include the relevant communication variables and modifying the $A_{i}$ 's accordingly to ignore these entries of $x_{i}$. This modification is useful when we have costs associated with these variables-for example, costs on power consumption. Equipped with the set of allowable flows and these utility functions, we can write this problem as a convex flow problem (2.3).

Related problems. Many different choices of the objective function and constraint sets for communication network allocation problems appear in the literature [XJB04; Ber98]. This setup also encompasses a number of other 'resource allocation' problems where the network structure isn't immediately obvious, one of which we discuss in the next section.

\subsection*{3.4 Market equilibrium and Nash bargaining}

Our framework includes and generalizes the concave network flow model used by Végh [Vég14] to study market equilibrium problems such as Arrow-Debreu Nash bargaining [Vaz12].

Market clearing problem. Consider a market with a set of $n_{b}$ buyers and $n_{g}$ goods. There is one divisible unit of each good to be sold. Buyer $i$ has a budget $b_{i} \geq 0$ and receives utility $u_{i}: \mathbf{R}_{+}^{n_{g}} \rightarrow \mathbf{R}_{+}$from some allocation $x_{i} \in[0,1]^{n_{g}}$ of goods. We assume that $u_{i}$ is concave and nondecreasing, with $u_{i}(0)=0$ for each $i=1, \ldots, n_{b}$. An equilibrium solution to this market is an allocation of goods $x_{i} \in \mathbf{R}^{n_{g}}$ for each buyer $i=1, \ldots, n_{b}$, and a price $p_{j} \in \mathbf{R}_{+}$for each good $j=1, \ldots, n_{g}$, such that: (1) all goods are sold; (2) all money of all buyers is spent; and (3) each buyer buys a 'best' (i.e., utility-maximizing) bundle of goods, given these prices. An equilibrium allocation for this market is given by a solution to the following convex program:
$$
\begin{array}{ll}
\operatorname{maximize} & \sum_{i=1}^{n_{b}} b_{i} \log \left(u_{i}\left(x_{i}\right)\right) \\
\text { subject to } & \sum_{i=1}^{n_{b}}\left(x_{i}\right)_{j}=1, \quad j=1, \ldots, n_{g}  \tag{3.8}\\
& x_{i} \geq 0, \quad i=1, \ldots, n_{b}
\end{array}
$$

Equilibrium conditions. Eisenberg and Gale [EG59] proved that the optimality conditions of this convex optimization problem give the equilibrium conditions in the special case that $u_{i}(x)$ is linear (and therefore separable across goods), i.e.,
$$
u_{i}\left(x_{i}\right)=v_{i}^{T} x_{i}
$$
for constant weights $v_{i} \in \mathbf{R}_{+}^{n_{g}}$. The same result can be easily derived for the general case. The Lagrangian of the Fisher market problem (3.8) is
$$
L(x, \mu, \lambda)=\sum_{i=1}^{n_{b}} b_{i} \log \left(U\left(x_{i}\right)\right)+\mu^{T}\left(\mathbf{1}-\sum_{i=1}^{n_{b}} x_{i}\right)-\sum_{i=1}^{n_{b}} x_{i}^{T} \lambda_{i}
$$
where $\left\{x_{i} \in \mathbf{R}^{n_{g}}\right\}$ are the primal variables and $\mu \in \mathbf{R}^{n_{g}}$ and $\left\{\lambda_{i} \in \mathbf{R}_{+}^{n_{g}}\right\}$ are the dual variables. Let $x^{\star}, \mu^{\star}, \lambda^{\star}$ be a primal-dual solution to this problem. The optimality conditions [BV04, §5.5] are primal feasibility, complementary slackness, and the dual condition
$$
\partial_{x_{i}} L\left(x^{\star}, \mu^{\star}, \lambda^{\star}\right)=\frac{b_{i}}{U\left(x_{i}^{\star}\right)} \nabla U\left(x_{i}^{\star}\right)-\mu^{\star}-\lambda_{i}^{\star}=0, \quad \text { for } i=1, \ldots, n_{b}
$$

This condition simplifies to
$$
\nabla U\left(x_{i}^{\star}\right) \geq\left(U\left(x_{i}\right) / b_{i}\right) \cdot \mu^{\star}, \quad \text { for } i=1, \ldots, n_{b} .
$$

If we let the prices of the goods be $\mu^{\star} \in \mathbf{R}^{n_{g}}$, this condition says that the marginal utility gained by an agent $i$ from an additional small amount of any good is at least as large as that agent's budget-weighted price times their current utility. As a result, the prices $\mu^{\star}$ will cause all agents to spend their entire budget on a utility-maximizing basket of goods, and all goods will be sold.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-042.jpg?height=587&width=1301&top_left_y=249&top_left_x=412}
\captionsetup{labelformat=empty}
\caption{Figure 3.6: Network representation of the linear Fisher market model, where we aim to maximize the utility of the net flows $y$. Colored edges represent the $n_{b}+1$ hyperedges connecting each buyer to all the goods, so each of these edges is incident to $n_{g}+1$ vertices. Flows on the right indicate that there is at most one unit of each good to be divided among the buyers.}
\end{figure}

Special cases. The linear case is called the 'linear Fisher market model' [Vaz07] and can be easily recognized as a special case of the standard maximum flow problem (3.5) with nonnegative edge gain factors [Vég14, §3]. Végh showed that all known extensions of the linear Fisher market model are a special case of the generalized market problem (3.8), where the utility functions $u_{i}$ are separable across the goods, i.e.,
$$
u_{i}\left(x_{i}\right)=\sum_{j=1}^{n_{g}} u_{i j}\left(\left(x_{i}\right)_{j}\right)
$$
for $u_{i j}: \mathbf{R} \rightarrow \mathbf{R}$ concave and increasing. Végh casts this problem as a maximum network flow problems with concave edge input-output functions. We make a further extension here by allowing the utilities to be concave nondecreasing functions of the entire basket of goods rather than sums of functions of the individual allocations. This generalization allows us to model complementary goods and is an immediate consequence of our framework.

The convex flow problem. This problem can be turned into a convex flow problem in a number of ways. Here, we follow Végh [Vég14] and use a similar construction. First, we represent both the $n_{b}$ buyers and $n_{g}$ goods as nodes on the graph: the buyers are labeled as nodes $1, \ldots, n_{b}$, and the goods are labeled as nodes $n_{b}+1, \ldots, n_{b}+n_{g}$. For each buyer $i=1, \ldots, n_{b}$, we introduce a hyperedge connecting buyer $i$ to all goods $j=n_{b}+1, \ldots, n_{b}+n_{g}$. We denote the flow along this edge by $x_{i} \in \mathbf{R}^{n_{g}+1}$, where $\left(x_{i}\right)_{j}$ is the amount of the $j$ th good bought by the $i$ th buyer, and $\left(x_{i}\right)_{n_{g}+1}$ denotes the amount of utility that buyer $i$ receives from this basket of goods denoted by the first $n_{g}$ entries of $x_{i}$. The flows on this edge are given by the convex set
$$
\begin{equation*}
T_{i}=\left\{(z, t) \in \mathbf{R}^{n_{g}} \times \mathbf{R} \mid-\mathbf{1} \leq z \leq 0, \quad t \leq u_{i}(-z)\right\} \tag{3.9}
\end{equation*}
$$

This set converts the 'goods' flow into a utility flow at each buyer node $i$. This setup is depicted in figure 3.6. Since the indices $1, \ldots, n_{b}$ of the net flow vector $y \in \mathbf{R}^{n_{g}+n_{b}}$ correspond to the buyers and the elements $n_{b}+1, \ldots, n_{g}+n_{b}$ to correspond to the goods, the utility function above can be written
$$
\begin{equation*}
U(y)=\sum_{i=1}^{n_{b}} b_{i} \log \left(y_{i}\right)-I\left(y_{n_{b}+1: n_{b}+n_{g}} \geq-\mathbf{1}\right) \tag{3.10}
\end{equation*}
$$
which includes the implicit constraint that at most one unit of each good can be sold in the indicator function $I$, above, and where $y_{n_{b}+1: n_{b}+n_{g}}$ is a vector containing only the $\left(n_{b}+1\right)$ st to the $\left(n_{b}+n_{g}\right)$ th entries of $y$. Since $U$ is nondecreasing and concave in $y, V_{i}=0$, and the $T_{i}$ are convex, we know that (3.8) is a special case of the convex flow problem (2.3).

Related problems. A number of other resource allocation can be cast as generalized network flow problems. For example, Agrawal et al. [Agr+22] consider a price adjustment algorithm for allocating compute resources to a set of jobs, and Schutz et al. [STA09] do the same for supply chain problems. In many problems, network structure implicitly appears if we are forced to make decisions over time or over decision variables which directly interact only with a small subset of other variables.

\subsection*{3.5 Routing orders through financial exchanges}

Financial asset networks are also well-modeled by convex network flows. If each asset is a node and each market between assets is an edge between the corresponding nodes, we expect the edge input-output functions to be concave, as the price of an asset is nondecreasing in the quantity purchased. In many markets, this input-output function is probabilistic; the state of the market when the order 'hits' is unknown due to factors such as information latency, stale orders, and front-running. However, in certain batched exchanges, including decentralized exchanges running on public blockchains, this state can be known in advance. We explore the order routing problem in this decentralized exchange setting.

Decentralized exchanges and automated market makers. Automated market makers have reached mass adoption after being implemented as decentralized exchanges on public blockchains. These exchanges (including Curve Finance [Ego19], Uniswap [AZR20], and Balancer [MM19a], among others) have facilitated trillions of dollars in cumulative trading volume since 2019 and maintain a collective daily trading volume of several billion dollars. These exchanges are almost all implemented as constant function market markers (CFMMs) [AC20a; Ang+23]. In CFMMs, liquidity providers contribute reserves of assets. Users can then trade against these reserves by tendering a basket of assets in exchange for another basket. CFMMs use a simple rule for accepting trades: a trade is only valid if the value of a given function at the post-trade reserves is equal to the value at the pre-trade reserves. This function is called the trading function and gives CFMMs their name.

Constant function market makers. A CFMM which allows $r$ assets to be traded is defined by two properties: its reserves $R \in \mathbf{R}^{r}$, which denotes the amount of each asset available to the CFMM, and its trading function $\varphi: \mathbf{R}^{r} \rightarrow \mathbf{R}$, which specifies its behavior and includes a fee parameter $0<\gamma \leq 1$, where $\gamma=1$ denotes no fee. We assume that $\varphi$ is concave and nondecreasing. Any user is allowed to submit a trade to a CFMM, which we write as a vector $z \in \mathbf{R}^{r}$, where positive entries denote values to be received from the CFMM and negative entries denote values to be tendered to the CFMM. (For example, if $r=2$, then a trade $z=(-1,10)$ would denote that the user wishes to tender 1 unit of asset 1 and receive 10 units of asset 2.) The submitted trade is then accepted if the following condition holds:
$$
\varphi\left(R-\gamma z_{-}-z_{+}\right) \geq \varphi(R)
$$
and $R-\gamma z_{-}-z_{+} \geq 0$. Here, we denote $z_{+}$to be the 'elementwise positive part' of $x$, i.e., $\left(z_{+}\right)_{j}=\max \left\{z_{j}, 0\right\}$ and $z_{-}$to be the 'elementwise negative part' of $x$, i.e., $\left(z_{-}\right)_{j}=\min \left\{z_{j}, 0\right\}$ for every asset $j=1, \ldots, r$. Note that, since $\varphi$ is concave, the set of acceptable trades is a convex set:
$$
T=\left\{z \in \mathbf{R}^{r} \mid \varphi\left(R-\gamma z_{-}-z_{+}\right) \geq \varphi(R)\right\}
$$
as we can equivalently write it as
$$
T=\left\{z \in \mathbf{R}^{r} \mid \varphi(R+\gamma u-v) \geq \varphi(R), u, v \geq 0, z=v-u\right\}
$$
which is easily seen to be a convex set since $\varphi$ is a concave function.
Examples. Almost all examples of decentralized exchanges currently in production are constant function market makers. For example, the most popular trading function (as measured by most metrics) is the product trading function:
$$
\varphi(R)=\sqrt{R_{1} R_{2}},
$$
originally proposed for Uniswap [ZCP18]. The associated set of allowable flows is
$$
T=\left\{z \in \mathbf{R}^{2} \mid\left(R_{1}+\gamma u_{1}-v_{1}\right)_{+}\left(R_{2}+\gamma u_{2}-v_{2}\right)_{+} \geq R_{1} R_{2}, z=v-u\right\}
$$
where $0 \leq \gamma \leq 1$ is the fee parameter and is set by the CFMM at creation time. We show this set in figure 3.7. This writing is bit verbose and difficult to parse, so we often work directly with the functional form. Other examples include the weighted geometric mean (as used by Balancer [MM19a])
$$
\begin{equation*}
\varphi(R)=\prod_{i=1}^{r} R_{i}^{w_{i}} \tag{3.11}
\end{equation*}
$$
where $r$ is the number of assets the exchange trades, and $w \in \mathbf{R}_{+}^{r}$ with $\mathbf{1}^{T} w=1$ are known as the weights, along with the Curve trading function
$$
\varphi(R)=\alpha \mathbf{1}^{T} R-\left(\prod_{i=1}^{r} R_{i}^{-1}\right),
$$
where $\alpha>0$ is a parameter set by the CFMM [Ego]. Note that the 'product' trading function is the special case of the weighted geometric mean function when $r=2$ and $w_{1}=w_{2}=1 / 2$.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-045.jpg?height=596&width=1636&top_left_y=232&top_left_x=247}
\captionsetup{labelformat=empty}
\caption{Figure 3.7: Left: the trading set for Uniswap (without fees) for $R=(1,1)$ (light gray) and $R=(2,2)$ (dark gray). Right: the trading set for Uniswap v3.}
\end{figure}

Bounded liquidity. We say the trading set $T$ has bounded liquidity in asset $i$ if the supremum
$$
\sup \left\{\Delta_{i} \mid \Delta \in T\right\}=\Delta_{i}^{\star},
$$
is achieved at some $\Delta^{\star} \in T$. This has the interpretation that there is a finite basket of assets such that we receive all possible amount of asset $i$ from the CFMM. We say a trading set has bounded liquidity if it has bounded liquidity for each asset $i=1, \ldots, n$. Examples of bounded liquidity CFMMs include Uniswap v3 [Ada+21a], given by trading function
$$
\begin{equation*}
\varphi(R)=\sqrt{\left(R_{1}+\alpha\right)\left(R_{2}+\beta\right)}, \tag{3.12}
\end{equation*}
$$
with $\alpha, \beta \geq 0$ (see figure 3.7), and CFMMs with linear trading functions. These bounded liquidity CFMMs are useful since arbitrage can be easily computed in many important practical cases. We discuss the arbitrage problem in more detail in §5.3.

Net trade. Consider a collection of $m$ CFMMs, each of which trades a subset of $n$ possible assets. Denoting the trade with the $i$ th CFMM by $x_{i}$, which must lie in the convex set $T_{i}$, we can write the net trade across all markets by $y \in \mathbf{R}^{n}$, where
$$
y=\sum_{i=1}^{m} A_{i} x_{i}, \quad \text { and } \quad x_{i} \in T_{i} .
$$

If $y_{j}>0$, we receive some amount of asset $j$ after executing all trades $\left\{x_{i}\right\}$. On the other hand, if $y_{j}<0$, we tender some of asset $j$ to the network.

Optimal routing. Finally, we denote the trader's utility of the network trade vector by $U: \mathbf{R}^{n} \rightarrow \mathbf{R} \cup\{-\infty\}$, where infinite values encode constraints. We assume that this function is concave and nondecreasing. We can choose $U$ to encode several important actions in markets, including liquidating a portfolio, purchasing a basket of assets, and finding arbitrage. For example, if we wish to find risk-free arbitrage, we may take
$$
U(y)=c^{T} y-I(y \geq 0),
$$
for some vector of prices $c \in \mathbf{R}^{n}$. See $[\mathrm{Ang}+22 \mathrm{~b}, § 5.2]$ for several additional examples. Letting $V_{i}=0$ for all $i=1, \ldots, m$, it's clear that the optimal routing problem in CFMMs is a special case of (2.3).

\section*{Chapter 4}

\section*{The dual problem and flow prices}

The remainder of the paper focuses on efficiently solving the convex flow problem (2.3). This problem has only one constraint coupling the edge flows and the net flow variables. As a result, we turn to dual decomposition methods [Boy+07; Ber16]. The general idea of dual decomposition methods is to solve the original problem by splitting it into a number of subproblems that can be solved quickly and independently. In this section, we will design a decomposition method that parallelizes over all edges and takes advantage of structure present in the original problem. This decomposition allows us to quickly evaluate the dual function and a subgradient. Importantly, our decomposition method also provides a clear programmatic interface to specify and solve the convex flow problem.

\subsection*{4.1 Dual decomposition}

To get a dual problem, we introduce a set of (redundant) additional variables for each edge and rewrite (2.3) as
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i} \\
& x_{i}=\tilde{x}_{i}, \quad \tilde{x}_{i} \in T_{i}, \quad i=1, \ldots, m,
\end{array}
$$
where we added the 'dummy' variables $\tilde{x}_{i} \in \mathbf{R}^{n_{i}}$ for $i=1, \ldots, m$. Next, we pull the constraint $\tilde{x}_{i} \in T_{i}$ for $i=1, \ldots, m$ into the objective by defining the indicator function
$$
I_{i}\left(\tilde{x}_{i}\right)= \begin{cases}0 & \tilde{x}_{i} \in T_{i} \\ +\infty & \text { otherwise }\end{cases}
$$

This rewriting gives the augmented problem,
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m}\left(V_{i}\left(x_{i}\right)-I_{i}\left(\tilde{x}_{i}\right)\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i}  \tag{4.1}\\
& x_{i}=\tilde{x}_{i}, \quad i=1, \ldots, m
\end{array}
$$
with variables $x_{i}, \tilde{x}_{i} \in \mathbf{R}^{n_{i}}$ for $i=1, \ldots, m$ and $y \in \mathbf{R}^{n}$. The Lagrangian [BV04, §5.1.1] of this problem is then
$$
\begin{equation*}
L(y, x, \tilde{x}, \nu, \eta)=U(y)-\nu^{T} y+\sum_{i=1}^{m}\left(V_{i}\left(x_{i}\right)+\left(A_{i}^{T} \nu-\eta_{i}\right)^{T} x_{i}\right)+\sum_{i=1}^{m}\left(\eta_{i}^{T} \tilde{x}_{i}-I_{i}\left(\tilde{x}_{i}\right)\right), \tag{4.2}
\end{equation*}
$$
where we have introduced the dual variables $\nu \in \mathbf{R}^{n}$ for the net flow constraint and $\eta_{i} \in \mathbf{R}^{n_{i}}$ for $i=1, \ldots, m$ for each of the individual edge constraints in (4.1). (We will write $x, \tilde{x}$, and $\eta$ as shorthand for $\left\{x_{i}\right\},\left\{\tilde{x}_{i}\right\}$, and $\left\{\eta_{i}\right\}$, respectively.) It's easy to see that the Lagrangian is separable over the primal variables $y, x$, and $\tilde{x}$.

Dual function. Maximizing the Lagrangian (4.2) over the primal variables $y, x$, and $\tilde{x}$ gives the dual function
$$
g(\nu, \eta)=\sup _{y}\left(U(y)-\nu^{T} y\right)+\sum_{i=1}^{m}\left(\sup _{x}\left(V_{i}(x)-\left(A_{i}^{T} \nu-\eta_{i}\right)^{T} x\right)+\sup _{\tilde{x}_{i} \in T_{i}} \eta_{i}^{T} \tilde{x}_{i}\right) .
$$

To evaluate the dual function, we must solve three subproblems, each parameterized by the dual variables $\nu$ and $\eta$. We denote the optimal values of these problems, which depend on the $\nu$ and $\eta$, by $\bar{U}, \bar{V}_{i}$, and $f_{i}$ :
$$
\begin{align*}
\bar{U}(\nu) & =\sup _{y}\left(U(y)-\nu^{T} y\right)  \tag{4.3а}\\
\bar{V}_{i}(\xi) & =\sup _{x_{i}}\left(V_{i}\left(x_{i}\right)-\xi^{T} x_{i}\right)  \tag{4.3b}\\
f_{i}(\tilde{\eta}) & =\sup _{\tilde{x}_{i} \in T_{i}} \tilde{\eta}_{i}^{T} \tilde{x}_{i} \tag{4.3c}
\end{align*}
$$

The functions $\bar{U}$ and $\left\{\bar{V}_{i}\right\}$ are essentially the Fenchel conjugate [BV04, §3.3] of the corresponding $U$ and $\left\{V_{i}\right\}$. Closed-form expressions for $\bar{U}$ and the $\left\{\bar{V}_{i}\right\}$ are known for many practical functions $U$ and $\left\{V_{i}\right\}$. Similarly, the functions $\left\{f_{i}\right\}$ are the support functions $[\operatorname{Roc} 70$, §13] for the sets $T_{i}$. For future reference, note that the $\bar{U}, \bar{V}_{i}$, and $f_{i}$ are convex, as they are the pointwise supremum of a family of affine functions, and may take on value $+\infty$, which we interpret as an implicit constraint. We can rewrite the dual function in terms of these functions (4.3) as
$$
\begin{equation*}
g(\nu, \eta)=\bar{U}(\nu)+\sum_{i=1}^{m}\left(\bar{V}_{i}\left(\eta_{i}-A_{i}^{T} \nu\right)+f_{i}\left(\eta_{i}\right)\right) \tag{4.4}
\end{equation*}
$$

Our ability to quickly evaluate these functions and their gradients governs the speed of any optimization algorithm we use to solve the dual problem. The dual function (4.4) also has very clear structure: the 'global' dual variables $\nu$ are connected to the 'local' dual variables $\eta$, only through the functions $\left\{\bar{V}_{i}\right\}$. If the $\bar{V}_{i}$ were all affine functions, then the problem would be separable over $\nu$ and each $\eta_{i}$.

Dual variables as prices. Subproblem (4.3a) for evaluating $\bar{U}(\nu)$ has a simple interpretation: if the net flows $y$ have per-unit prices $\nu \in \mathbf{R}^{n}$, find the maximum net utility, after
removing costs, over all net flows. (There need not be feasible edge flows $x$ which correspond to this net flow.) Assuming $U$ is differentiable, a $y$ achieving this maximum satisfies
$$
\nabla U(y)=\nu
$$
i.e., the marginal utilities for flows $y$ are equal to the prices $\nu$. (A similar statement for a non-differentiable $U$ follows directly from subgradient calculus.) The subproblems over the $V_{i}(4.3 \mathrm{~b})$ have a similar interpretation as utility maximization problems.

On the other hand, subproblem (4.3c) for evaluating $f_{i}(\tilde{\eta})$ can be interpreted as finding a most 'valuable' allowable flow over edge $i$. In other words, if there exists an external, infinitely liquid reference market where we can buy or sell flows $x_{i}$ for prices $\tilde{\eta} \in \mathbf{R}^{n_{i}}$, then $f_{i}(\tilde{\eta})$ gives the highest net value of any allowable flow $x_{i} \in T_{i}$. Due to this interpretation, we will refer to (4.3c) as the arbitrage problem. This price interpretation is also a natural consequence of the optimality conditions for this subproblem. The optimal flow $x^{0}$ is a point in $T_{i}$ such that there exists a supporting hyperplane to $T_{i}$ at $x^{0}$ with slope $\tilde{\eta}$. In other words, for any small deviation $\delta \in \mathbf{R}^{n_{i}}$, if $x^{0}+\delta \in T_{i}$, then
$$
\tilde{\eta}^{T}\left(x^{0}+\delta\right) \leq \tilde{\eta}^{T} x^{0} \Longrightarrow \tilde{\eta}^{T} \delta \leq 0
$$

If, for example, $\delta_{j}$ and $\delta_{k}$ are the only two nonzero entries of $\delta$, we would have
$$
\delta_{j} \leq-\frac{\tilde{\eta}_{k}}{\tilde{\eta}_{j}} \delta_{k}
$$
so the exchange rate between $j$ and $k$ is at most $\tilde{\eta}_{j} / \tilde{\eta}_{k}$. This observation lets us interpret the dual variables $\eta$ as 'marginal prices' on each edge, up to a constant multiple. With this interpretation, we will soon see that the function $\bar{V}_{i}$ also connects the 'local prices' $\eta_{i}$ on edge $i$ to the 'global prices' $\nu$ over the whole network.

Duality. An important consequence of the definition of the dual function is weak duality [BV04, §5.2.2]. Letting $p^{\star}$ be an optimal value for the convex flow problem (2.3), we have that
$$
\begin{equation*}
g(\nu, \eta) \geq p^{\star} \tag{4.5}
\end{equation*}
$$
for every possible choice of $\nu$ and $\eta$. An important (but standard) result in convex optimization states that there exists a set of prices $\left(\nu^{\star}, \eta^{\star}\right)$ which actually achieve the bound:
$$
g\left(\nu^{\star}, \eta^{\star}\right)=p^{\star}
$$
under mild conditions on the problem data [BV04, §5.2]. One such condition is if all the $T_{i}$ 's are affine sets, as in §3.1. Another is Slater's condition: if there exists a point in the relative interior of the feasible set, i.e., if the set
$$
\left(\sum_{i=1}^{m} \operatorname{relint}\left(A_{i}\left(T_{i} \cap \operatorname{dom} V_{i}\right)\right)\right) \cap \operatorname{relint} \operatorname{dom} U
$$
is nonempty. (We have used the fact that the $A_{i}$ are one-to-one projections.) These conditions are relatively technical but almost always hold in practice. We assume they hold for the remainder of this section.

\subsection*{4.2 The dual problem}

The dual problem is then to find a set of prices $\nu^{\star}$ and $\eta^{\star}$ which saturate the bound (4.5) at equality; or, equivalently, the problem is to find a set of prices that minimize the dual function $g$. Using the definition of $g$ in (4.4), we may write this problem as
$$
\begin{equation*}
\operatorname{minimize} \bar{U}(\nu)+\sum_{i=1}^{m}\left(\bar{V}_{i}\left(\eta_{i}-A_{i}^{T} \nu\right)+f_{i}\left(\eta_{i}\right)\right), \tag{4.6}
\end{equation*}
$$
over variables $\nu$ and $\eta$. The dual problem is a convex optimization problem since $\bar{U}, \bar{V}_{i}$, and $f_{i}$ are all convex functions. For fixed $\nu$, the dual problem (4.7) is also separable over the dual variables $\eta_{i}$ for $i=1, \ldots, m$; we will later use this fact to speed up solving the problem by parallelizing our evaluations of each $\bar{V}_{i}$ and $f_{i}$.

Implicit constraints. The 'unconstrained' problem (4.6) has implicit constraints due to the fact that the $U$ and $V_{i}$ are nondecreasing functions. More specifically, if $U$ is nondecreasing and $U(0)<\infty$, then, if $\nu_{i}<0$, we have
$$
\bar{U}(\nu)=\sup _{y}\left(U(y)-\nu^{T} y\right) \geq U\left(t e_{i}\right)-t \nu_{i} \geq U(0)-t \nu_{i} \rightarrow \infty
$$
as $t \uparrow \infty$. Here, in the first inequality, we have chosen $y=t e_{i}$, where $e_{i}$ is the $i$ th unit basis vector. This implies that $\bar{U}(\nu)=\infty$ if $\nu \nsucceq 0$, which means that $\nu \geq 0$ is an implicit constraint. A similar proof shows that $\bar{V}_{i}(\xi)=\infty$ if $\xi \nsucceq 0$. Adding both implicit constraints as explicit constraints gives the following constrained optimization problem:
$$
\begin{array}{ll}
\operatorname{minimize} & \bar{U}(\nu)+\sum_{i=1}^{m}\left(\bar{V}_{i}\left(\eta_{i}-A_{i}^{T} \nu\right)+f_{i}\left(\eta_{i}\right)\right)  \tag{4.7}\\
\text { subject to } & \nu \geq 0, \quad \eta_{i} \geq A_{i}^{T} \nu, \quad i=1, \ldots, m
\end{array}
$$

Note that this implicit constraint exists even if $U(0)=\infty$; we only require that the domain of $U$ is nonempty, i.e., that there exists some $y$ with $U(y)<\infty$, and similarly for the $V_{i}$. The result follows from a nearly-identical proof. This fact has a simple interpretation in the context of utility maximization as discussed previously: if we have a nondecreasing utility function and are paid to receive some flow, we will always choose to receive more of it.

In general, the rewriting of problem (4.6) into problem (4.7) is useful since, in practice, $\bar{U}(\nu)$ is finite (and often differentiable) whenever $\nu \geq 0$; a similar thing is true for the functions $\left\{V_{i}\right\}$. Of course, this need not always be true, in which case the additional implicit constraints need to be made explicit in order to use standard, off-the-shelf solvers for types of problems.

Optimality conditions. Let $\left(\nu^{\star}, \eta^{\star}\right)$ be an optimal point for the dual problem, and assume that $g$ is differentiable at this point. The optimality conditions for the dual problem are then
$$
\nabla g\left(\nu^{\star}, \eta^{\star}\right)=0
$$
(The function $g$ need not be differentiable, in which case a similar argument holds using subgradient calculus.) For a differentiable $\bar{U}$, we have that
$$
\nabla_{\nu} \bar{U}\left(\nu^{\star}\right)=-y^{\star}
$$
where $y^{\star}$ is the optimal point for subproblem (4.3a). For a differentiable $\bar{V}_{i}$, we have that
$$
\begin{aligned}
& \nabla_{\nu} \bar{V}_{i}\left(\eta_{i}^{\star}-A_{i}^{T} \nu^{\star}\right)=A_{i} x_{i}^{\star} \\
& \nabla_{\eta_{i}} \bar{V}_{i}\left(\eta_{i}^{\star}-A_{i}^{T} \nu^{\star}\right)=-x_{i}^{\star}
\end{aligned}
$$
where $x_{i}^{\star}$ is the optimal point for subproblem (4.3b). Finally, we have that
$$
\nabla f_{i}\left(\eta_{i}^{\star}\right)=\tilde{x}_{i}^{\star}
$$
where $\tilde{x}_{i}^{\star}$ is the optimal point for subproblem (4.3c). Putting these together with the definition of the dual function (4.4), we recover primal feasibility at optimality:
$$
\begin{align*}
& \nabla_{\nu} g\left(\nu^{\star}, \eta^{\star}\right)=y^{\star}-\sum_{i=1}^{m} A_{i} x_{i}^{\star}=0,  \tag{4.8}\\
& \nabla_{\eta_{i}} g\left(\nu^{\star}, \eta^{\star}\right)=x_{i}^{\star}-\tilde{x}_{i}^{\star}=0, \quad i=1, \ldots, m
\end{align*}
$$

In other words, by choosing the 'correct' prices $\nu^{\star}$ and $\eta^{\star}$ (i.e., those which minimize the dual function), we find that the optimal solutions to the subproblems in (4.3) satisfy the resulting coupling constraints, when the functions in (4.3) are all differentiable at the optimal prices $\nu^{\star}$ and $\eta^{\star}$. This, in turn, implies that the $\left\{x_{i}^{\star}\right\}$ and $y^{\star}$ are a solution to the original problem (2.3). In the case that the functions are not differentiable, there might be many optimal solutions to the subproblems of (4.3), and we are only guaranteed that at least one of these solutions satisfies primal feasibility. We give some heuristics to handle this latter case in §5.2.

Dual optimality conditions. For problem (4.7), if the functions $U$ and $V_{i}$ are differentiable at optimality, the dual conditions state that
$$
\begin{align*}
\nabla U\left(y^{\star}\right) & =\nu^{\star} \\
\nabla V_{i}\left(x_{i}^{\star}\right) & =\eta_{i}^{\star}-A_{i}^{T} \nu^{\star}, \quad i=1, \ldots, m  \tag{4.9}\\
\eta_{i}^{\star} & \in \mathcal{N}_{i}\left(\tilde{x}_{i}^{\star}\right), \quad i=1, \ldots, m
\end{align*}
$$
where $\mathcal{N}_{i}(x)$ is the normal cone for set $T_{i}$ at the point $x$, defined as
$$
\mathcal{N}_{i}(x)=\left\{u \in \mathbf{R}^{n_{i}} \mid u^{T} x \geq u^{T} z \text { for all } z \in T_{i}\right\}
$$

Note that, since $U$ is nondecreasing, then, if $U$ is differentiable, its gradient must be nonnegative, which includes the implicit constraints in (4.7). (A similar thing is true for the $V_{i}$.) A similar argument holds in the case that $U$ and the $V_{i}$ are not differentiable at optimality, via subgradient calculus, and the implicit constraints are similarly present.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-052.jpg?height=579&width=695&top_left_y=242&top_left_x=713}
\captionsetup{labelformat=empty}
\caption{Figure 4.1: At optimality, the local prices $\eta_{i}^{\star}$ define a supporting hyperplane of the set of allowable flows $T_{i}$ at an optimal flow $x_{i}^{\star}$.}
\end{figure}

Interpretations. The dual optimality conditions (4.9) each have lovely economic interpretations. In particular, they state that, at optimality, the marginal utilities from the net flows $y^{\star}$ are equal to the prices $\nu^{\star}$, the marginal utilities from the edge flows $x_{i}^{\star}$ are equal to the difference between the local prices $\eta_{i}^{\star}$ and the global prices $\nu^{\star}$, and the local prices $\eta_{i}^{\star}$ are a supporting hyperplane of the set of allowable flows $T_{i}$. This interpretation is natural in problems involving markets for goods, such as those discussed in §3.2 and §3.5, where one may interpret the normal cone $\mathcal{N}_{i}\left(x_{i}^{\star}\right)$ as the no-arbitrage cone for the market $T_{i}$ : any change in the local prices $\eta_{i}^{\star}$ such that $\eta_{i}^{\star}$ is still in the normal cone $\mathcal{N}_{i}\left(\tilde{x}_{i}^{\star}\right)$ does not affect our action $\tilde{x}_{i}^{\star}$ with respect to market $i$. We illustrate this interpretation in figure 4.1.

We can also interpret the dual problem (4.7) similarly to the primal. Here, each subsystem has local 'prices' $\eta_{i}$ and is described by the functions $f_{i}$ and $\bar{V}_{i}$, which implicitly include the edge flows and associated constraints. The global prices $\nu$ are associated with a cost function $\bar{U}$, which implicitly depends on (potentially infeasible) net flows, $y$. The function $\bar{V}_{i}$ may be interpreted as a convex cost function of the difference between the local subsystem prices $\eta_{i}$ and the global prices $\nu$. At optimality, this difference will be equal to the marginal utility gained from the optimal flows in subsystem $i$, and the global prices will be equal to the marginal utility of the overall system.

Finally, we note that the convex flow problem (2.3) and its dual (4.6) examined in this section are closely related to the extended monotropic programming problem [Ber08]. We make this connection explicit in appendix A. Importantly, the extended monotropic programming problem is self-dual, whereas the convex flow problem is not evidently self-dual. This observation suggests an interesting avenue for future work.

\subsection*{4.2.1 Cycle condition}

We define the vectors $\delta_{1}, \ldots, \delta_{m}$ to be an arbitrage with respect to the flows $\left\{x_{i}\right\}$ if the following conditions hold:
1. The vector $\delta_{i} \in T_{i}^{\star}\left(x_{i}\right)$ for all $i=1, \ldots, m$, where
$$
T_{i}^{\star}\left(x_{i}\right)=\left\{\delta \mid x_{i}+t \delta_{i} \in T_{i} \text { for some } t>0\right\} .
$$
2. For some subgradient $\hat{\nu} \in U\left(\sum_{i=1}^{m} x_{i}\right)$, we have that
$$
\hat{\nu}^{T}\left(\sum_{i=1}^{m} A_{i} \delta_{i}\right)>0
$$

Defining the matrix
$$
\tilde{A}=\left[\begin{array}{lll}
A_{1} & \cdots & A_{m}
\end{array}\right],
$$
the second condition says that there exists a nonnegative vector $\delta=\left(\delta_{1}, \ldots, \delta_{m}\right)$ with at least one strictly positive component in the nullspace of $\tilde{A}$. This nullspace can be interpreted as a 'cycle basis' of the hypergraph.

When all edges are between two nodes, the optimality conditions have a nice interpretation in terms of cycle conditions, similar to the augmenting path condition for max flow, given by Ford and Fulkerson. In fact, for the standard max flow problem, this condition is exactly the augmenting path condition: if there exists an augmenting path, then the flow is not optimal.

\subsection*{4.2.2 Downward closure and monotonicity}

Here, we outline a proof, using the dual problem, that downward closure of the sets $T_{i}$ implies that the objective functions $U$ and $\left\{V_{i}\right\}$ are nondecreasing, and vice versa.

Monotonicity implies downward closure. Assume that the objective functions $U$ and $\left\{V_{i}\right\}$ are nondecreasing. Then the dual variables $\nu$ and $\left\{\eta_{i}\right\}$ must all be nonnegative for the dual to be finite-valued. Consider the arbitrage problem for some $i$ :
$$
f_{i}(\eta)=\sup _{x \in T} \eta_{i}^{T} x
$$

The solution to the arbitrage problem can be broken into two cases. First, if the set $T_{i}$ contains a positive ray, then the optimal value is unbounded, and the primal problem is therefore infeasible. Second, if the optimal value is finite, then the solution will lie on the boundary of $T_{i}$, as $T_{i}$ is closed and convex. As a result, we can replace $T_{i}$ with its 'downward extension',
$$
\tilde{T}_{i}=T_{i}-\mathbf{R}_{+}^{n}
$$
without affecting the solution.
Downward closure implies monotonicity. Assume that the sets $\left\{T_{i}\right\}$ are downward closed. We use $x_{i}^{\star}$ to denote a solution to the arbitrage problem:
$$
x_{i}^{\star}=\underset{x \in T_{i}}{\operatorname{argmax}} \eta_{i}^{T} x
$$
for a fixed price vector $\eta_{i}$ when this solution is finite. Consider two possible cases for $f_{i}$. First, if $f_{i}$ is unbounded, then the primal problem is again infeasible. Second, if $f_{i}$ is is finite, then the set $T_{i}$ is contained in the halfspace $\left\{x \mid \eta_{i}^{T} x \leq f_{i}\left(\eta_{i}\right)\right\}$, and $x_{i}^{\star}$ is on the boundary of $T_{i}$. Since $T_{i}$ is downward closed, we must have $\eta_{i} \geq 0$ in this case. As a result, we can replace the objective function with its monotonic concave envelope without affecting the solution.

\subsection*{4.3 Special cases}

\subsection*{4.3.1 Zero edge utilities}

An important special case is when the edge flow utilities are zero, i.e., if $V_{i}=0$ for $i= 1, \ldots, m$. In this case, the convex flow problem reduces to the routing problem discussed in §3.5, originally presented in [Ang + 22a; Dia + 23] in the context of constant function market makers [AC20a]. Note that $\bar{V}_{i}$ becomes
$$
\bar{V}_{i}\left(\xi_{i}\right)= \begin{cases}0 & \xi_{i}=0 \\ +\infty & \text { otherwise },\end{cases}
$$
which means that the dual problem is
$$
\begin{array}{ll}
\operatorname{minimize} & \bar{U}(\nu)+\sum_{i=1}^{m} f_{i}\left(\eta_{i}\right) \\
\text { subject to } & \eta_{i}=A_{i}^{T} \nu, \quad i=1, \ldots, m .
\end{array}
$$

This equality constraint can be interpreted as ensuring that the local prices for each node are equal to the global prices over the net flows of the network. If we substitute $\eta_{i}=A_{i}^{T} \nu$ in the objective, we have
$$
\begin{equation*}
\operatorname{minimize} \quad \bar{U}(\nu)+\sum_{i=1}^{m} f_{i}\left(A_{i}^{T} \nu\right), \tag{4.10}
\end{equation*}
$$
which is exactly the dual of the optimal routing problem, originally presented in $[\operatorname{Dia}+23]$. In the case of constant function market makers (see §3.5), we interpret the subproblem of computing the value of $f_{i}$, at some prices $A_{i}^{T} \nu$, as finding the optimal arbitrage with the market described by $T_{i}$, given 'true' (global) asset prices $\nu$. This interpretation also follows directly from the optimality conditions. Specializing (4.9), we see that an optimal price-trade pair must satisfy
$$
\nu^{\star}=\nabla U\left(y^{\star}\right) \in \bigcap_{i=1}^{m} A_{i} \mathcal{N}_{i}\left(x_{i}^{\star}\right) .
$$

In other words, the global prices $\nu^{\star}$ must define a supporting hyperplane for each set $T_{i}$ at the optimal trade $x_{i}^{\star}$ for $i=1, \ldots, m$. These normal cones may be interpreted as 'no arbitrage' cones for the sets $T_{i}$ at $x_{i}$; for some local prices $A_{i}^{T} \nu$, we have that, by definition, there is no $z \in T_{i}$ with $\left(A_{i}^{T} \nu\right)^{T} z>\left(A_{i}^{T} \nu\right)^{T} x_{i}$.

Problem size. Because this problem has only $\nu$ as a variable, which is of length $n$, this problem is often much smaller than the original dual problem of (4.7). Indeed, the number of variables in the original dual problem is $n+\sum_{i=1}^{m} n_{i} \geq n+2 m$, whereas this problem has exactly $n$ variables. (Here, we have assumed that the feasible flow sets lie in a space of at least two dimensions, $n_{i} \geq 2$.) This special case is very common in practice and identifying it often leads to significantly faster solution times, as the number of edges in many practical networks is much larger than the total number of nodes, i.e., $m \gg n$.

Example. Using this special case, is easy to show that the dual for the maximum flow problem (3.5), introduced in §3.1, is the minimum cut problem, as expected from the celebrated result of [DF56; EFS56; FF56]. Recall from §3.1 that
$$
U(y)=y_{n}-I_{S}(y), \quad T_{i}=\left\{z \in \mathbf{R}^{2} \mid 0 \leq z_{2} \leq b_{i}, \quad z_{1}+z_{2}=0\right\}
$$
where $S=\left\{y \in \mathbf{R}^{n} \mid y_{1}+y_{n} \geq 0, y_{i} \geq 0\right.$ for all $\left.i \neq 1, n\right\}$ and $b_{i} \geq 0$ is the maximum allowable flow across edge $i$. Using the definitions of $\bar{U}$ and $f_{i}$ in (4.3), we can easily compute $\bar{U}(\nu)$,
$$
\bar{U}(\nu)=\sup _{y \in S}\left(y_{n}-\nu^{T} y\right)= \begin{cases}0 & \nu_{n} \geq 1, \nu_{n}-\nu_{1}=1, \nu_{i} \geq 0 \text { for all } i \neq 1, n \\ +\infty & \text { otherwise }\end{cases}
$$
and $f_{i}(\eta)$,
$$
f_{i}(\eta)=\sup _{z \in T_{i}} \eta^{T} z=b_{i}\left(\eta_{2}-\eta_{1}\right)_{+}
$$
where we write $(w)_{+}=\max \{w, 0\}$. Using the special case of the problem when we have zero edge utilities (4.10) and adding the constraints gives the dual problem
$$
\begin{array}{ll}
\text { minimize } & \sum_{i=1}^{m} b_{i}\left(\left(A_{i}^{T} \nu\right)_{2}-\left(A_{i}^{T} \nu\right)_{1}\right)_{+} \\
\text {subject to } & \nu_{n}-\nu_{1}=1 \\
& \nu_{n} \geq 1, \nu_{i} \geq 0, \text { for all } i \neq 1, n
\end{array}
$$

Note that this problem is 1 -translation invariant: the problem has the same objective value and remains feasible if we replace any feasible $\nu$ by $\nu+\alpha \mathbf{1}$ for any constant $\alpha$ such that $\nu_{n}+\alpha \geq 1$. Thus, without loss of generality, we may always set $\nu_{1}=0$ and $\nu_{n}=1$. We then use an epigraph transformation and introduce new variables for each edge, $t \in \mathbf{R}^{m}$, so the problem becomes
$$
\begin{array}{ll}
\operatorname{minimize} & b^{T} t \\
\text { subject to } & \left(A_{i}^{T} \nu\right)_{1}-\left(A_{i}^{T} \nu\right)_{2} \leq t_{i}, \quad i=1, \ldots n \\
& \nu_{n}=1, \nu_{1}=0 \\
& t \geq 0, \nu \geq 0
\end{array}
$$

The substitution of $\nu_{n}=1$ and $\nu_{1}=0$ in the first constraint recovers a standard formulation of the minimum cut problem (see, e.g., [DF56, §3]).

\subsection*{4.3.2 Circulation problem}

If the net flow utility simply constrains the net flow to be zero, i.e.,
$$
U(y)=-I_{\{0\}}(y)
$$
then we recover a generalized circulation problem. The dual problem becomes
$$
\operatorname{minimize} \sum_{i=1}^{m}\left(\bar{V}_{i}\left(\eta_{i}-A_{i}^{T} \nu\right)+f_{i}\left(\eta_{i}\right)\right)
$$

Taking the infimum of the objective over $\eta$ and introducing a new variables $z_{i} \in \mathbf{R}^{n_{i}}$ for $i=1, \ldots, m$, we can rewrite this problem as
$$
\begin{array}{ll}
\operatorname{minimize} & \sum_{i=1}^{m} \tilde{V}_{i}\left(z_{i}\right) \\
\text { subject to } & z_{i}=A_{i}^{T} \nu, \quad i=1, \ldots, m \\
& \nu \geq 0
\end{array}
$$
where we define
$$
\tilde{V}_{i}\left(z_{i}\right)=\inf _{\eta_{i}}\left\{\bar{V}_{i}\left(\eta_{i}-z_{i}\right)+f_{i}\left(\eta_{i}\right)\right\}
$$
(This is very close to, but not quite, the infimal convolution of $\bar{V}_{i}$ and $f_{i}$.) Note that $\tilde{V}_{i}$ is a convex function, as convexity is preserved under partial minimization [BV04, §3.2.5]. This problem has a nice interpretation: we are finding the pin voltages on the $m$ components in a passive electrical circuit [Boy+07, §6]. The optimality conditions simplify to
$$
\begin{aligned}
\nabla \tilde{V}_{i}\left(z_{i}\right) & =x_{i}, \quad i=1, \ldots, m \\
A_{i}^{T} \nu & =z_{i}, \quad i=1, \ldots, m \\
\sum_{i=1}^{m} A_{i} x_{i} & \geq 0, \quad \nu \geq 0
\end{aligned}
$$

Viewing $z_{i}$ and $x_{i}$ as the voltage and current, respectively, at the terminals of component $i$, and viewing $\nu$ as the voltages at every node in the circuit, then the first equation can can be interpreted as the voltage-current characteristic for component $i$, the second as the Kirchoff voltage law, and the last as Kirchoff's current law, respectively.

\subsection*{4.4 Conic dual and self-duality}

Now that we know that problem (2.3) and problem (2.4) are essentially equivalent (even though the conic problem (2.4) 'seems' more restrictive) we give a dual reformulation of (2.4) that is 'almost' self-dual in this section.

\section*{Dual problem}

We will write a simple dual for the conic problem (2.4) using standard duality results and a basic rewriting of the problem.

Lagrangian. First, we write the conic flow problem (2.4) here again for convenience:
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i}  \tag{2.4}\\
& x_{i} \in K_{i}, \quad i=1, \ldots, m
\end{array}
$$

We pull the conic constraint $x_{i} \in K_{i}$ into the objective by defining the indicator functions
$$
I_{i}\left(x_{i}\right)= \begin{cases}0 & x_{i} \in K_{i} \\ +\infty & \text { otherwise }\end{cases}
$$
for $i=1, \ldots, m$. We can then rewrite the conic problem as
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right)-I_{i}\left(\tilde{x}_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i} \\
& \tilde{x}_{i}=x_{i}, \quad i=1, \ldots, m
\end{array}
$$
where we have introduced the new redundant variables $\tilde{x}_{i} \in \mathbf{R}^{n_{i}}$ for each $i=1, \ldots, m$. This resulting problem is just a convex problem with linear constraints. Introducing the Lagrange multipliers $\nu \in \mathbf{R}^{n}$ for the first equality constraint and $\eta_{i} \in \mathbf{R}^{n_{i}}$ for the second equality constraint, we get the Lagrangian:
$$
L(x, \tilde{x}, y, \nu, \eta)=U(y)+\sum_{i=1}^{m}\left(V_{i}\left(x_{i}\right)-I_{i}\left(\tilde{x}_{i}\right)\right)+\nu^{T}\left(y-\sum_{i=1}^{m} A_{i} x_{i}\right)+\sum_{i=1}^{m} \eta_{i}^{T}\left(x_{i}-\tilde{x}_{i}\right)
$$

Collecting terms over the primal variables:
$$
L(x, \tilde{x}, y, \nu, \eta)=U(y)+\nu^{T} y+\sum_{i=1}^{m}\left(V_{i}\left(x_{i}\right)+\left(\eta_{i}-A_{i}^{T} \nu\right)^{T} x_{i}\right)+\sum_{i=1}^{m}\left(\eta_{i}^{T} \tilde{x}_{i}-I_{i}\left(\tilde{x}_{i}\right)\right)
$$

Dual function. To find the dual function (and therefore the dual problem) we partially maximize $L$ over the primal variables $x, \tilde{x}$, and $y$ :
$$
\begin{equation*}
g(\nu, \eta)=\bar{U}(\nu)+\sum_{i=1}^{m} \bar{V}_{i}\left(\eta_{i}-A_{i}^{T} \nu\right)+\sum_{i=1}^{m} \bar{I}_{i}\left(\eta_{i}\right) \tag{4.11}
\end{equation*}
$$

Here we have defined
$$
\bar{U}(\nu)=\sup _{y}\left(U(y)+\nu^{T} y\right), \quad \bar{V}_{i}\left(\xi_{i}\right)=\sup _{x_{i}}\left(V_{i}\left(x_{i}\right)+\xi_{i}^{T} x_{i}\right)
$$
and the functions $\left\{\bar{I}_{i}\right\}$ as
$$
\bar{I}_{i}\left(\eta_{i}\right)=\sup _{\tilde{x}_{i}}\left(-I\left(\tilde{x}_{i}\right)+\tilde{x}_{i}^{T} \eta_{i}\right)
$$
for each $i=1, \ldots, m$. Note that the function $\bar{I}_{i}$ is simple the indicator for the polar cone of $K_{i}$, defined in (2.5). In other words,
$$
\bar{I}_{i}\left(x_{i}\right)= \begin{cases}0 & \eta_{i} \in K_{i}^{\circ} \\ +\infty & \text { otherwise }\end{cases}
$$

Dual problem. The dual problem is then to minimize the dual function $g$; i.e.,
$$
\text { minimize } g(\nu, \eta) .
$$

When there exists a point in the relative interior of the domain, strong duality holds and, therefore, the optimal values of the dual problem and the primal problem are identical. Plugging in the definition of $g$ from (4.11) into the objective of the dual problem, and pulling out the indicator functions $\left\{\bar{I}_{i}\right\}$ into explicit constraints gives
$$
\begin{array}{ll}
\operatorname{minimize} & \bar{U}(\nu)+\sum_{i=1}^{m} \bar{V}_{i}\left(\eta_{i}-A_{i}^{T} \nu\right) \\
\text { subject to } & \eta_{i} \in K_{i}^{\circ}, \quad i=1, \ldots, m
\end{array}
$$

We can rewrite the problem to make it more similar to the original (2.4). If we define $\xi_{i}=A_{i}^{T} \nu$ then
$$
D \nu=\sum_{i=1}^{m} A_{i} \xi_{i}
$$
where $D$ is a diagonal matrix
$$
D=\sum_{i=1}^{m} A_{i} A_{i}^{T}
$$
with nonnegative diagonal entries. The $j$ th diagonal entry, $D_{j j}$, denotes the degree of node $j$, for $j=1, \ldots, n$. The diagonal entries of $D$ are strictly positive if the hypergraph corresponding to the $A_{i}$ has no isolated nodes, or, equivalently, if, for each node $j=1, \ldots, n$ there is some edge $i=1, \ldots, m$ such that the $j$ th row of $A_{i}$ is nonzero. In this case, which we may always assume in practice by removing isolated nodes, the inverse of $D$ exists so the relationship between $\nu$ and the $\xi_{i}$ is bijective. This means we can rewrite the dual problem:
$$
\begin{array}{ll}
\operatorname{minimize} & \bar{U}(\nu)+\sum_{i=1}^{m} \bar{V}_{i}\left(\eta_{i}-\xi_{i}\right) \\
\text { subject to } & D \nu=\sum_{i=1}^{m} A_{i} \xi_{i} \\
& \eta_{i} \in K_{i}^{\circ}, \quad i=1, \ldots, m
\end{array}
$$

We may absorb the matrix $D$ into the definition of $\bar{U}$ by replacing $\bar{U}(\nu)$ with $\bar{U}\left(D^{-1} \nu\right)$ to get the slightly more familiar-looking problem
$$
\begin{array}{ll}
\operatorname{minimize} & \bar{U}(\nu)+\sum_{i=1}^{m} \bar{V}_{i}\left(\eta_{i}-\xi_{i}\right) \\
\text { subject to } & \nu=\sum_{i=1}^{m} A_{i} \xi_{i}  \tag{4.12}\\
& \eta_{i} \in K_{i}^{\circ}, \quad i=1, \ldots, m
\end{array}
$$

For the sake of convenience, we reiterate the problem's variables: these are the node dual prices $\nu \in \mathbf{R}^{n}$ and the edge dual prices $\xi_{i} \in \mathbf{R}^{n_{i}}$, for $i=1, \ldots, n$. We call this problem the dual conic flow problem. Compare this problem (4.12) with the original conic flow problem (2.4).

\section*{Chapter 5}

\section*{Solving the dual problem.}

The dual problem (4.7) is a convex optimization problem that is easily solvable in practice, even for very large $n$ and $m$. For small problem sizes, we can use an off-the-self solver, such as such as SCS [ODo+16], Hypatia [CKV21], or Mosek [ApS24a], to tackle the convex flow problem (2.3) directly; however, these methods, which rely on conic reformulations, destroy problem structure and may be unacceptably slow for large problem sizes. The dual problem, on the other hand, preserves this structure, so our approach is to solve this dual problem.

A simple transformation. For the sake of exposition, we will introduce the new variable $\mu=(\nu, \eta)$ and write the dual problem (4.7) as
$$
\begin{array}{ll}
\operatorname{minimize} & g(\mu) \\
\text { subject to } & F \mu \geq 0
\end{array}
$$
where $F$ is the constraint matrix
$$
F=\left[\begin{array}{cccc}
I & 0 & \cdots & 0 \\
-A_{1}^{T} & I & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
-A_{m}^{T} & 0 & \cdots & I
\end{array}\right]
$$

Since the matrix $F$ is lower triangular with a diagonal that has no nonzero entries, the matrix $F$ is invertible. Its inverse is given by
$$
F^{-1}=\left[\begin{array}{cccc}
I & 0 & \cdots & 0 \\
A_{1}^{T} & I & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
A_{m}^{T} & 0 & \cdots & I
\end{array}\right]
$$
which can be very efficiently applied to a vector. With these matrices defined, we can rewrite the dual problem as
$$
\begin{array}{ll}
\operatorname{minimize} & g\left(F^{-1} \tilde{\mu}\right) \\
\text { subject to } & \tilde{\mu} \geq 0 \tag{5.1}
\end{array}
$$
where $\tilde{\mu}=F \mu$. Note that the matrix $F$ preserves the separable structure of the problem: edges are only directly coupled with their adjacent nodes.

An efficient algorithm. After this transformation, we can apply any first-order method that handles nonnegativity constraints to solve (5.1). We use the quasi-Newton algorithm L -BFGS-B [Byr +95 ; Zhu +97 ; MN11], which has had much success in practice. This algorithm only requires the ability to evaluate the function $g$ and its gradient $\nabla g$ at a given point $\mu=F^{-1} \tilde{\mu}$. The function $g(\mu)$ and its gradient $\nabla g(\mu)$ are, respectively, a sum of the optimal values and a sum of the optimal points for the subproblems (4.3).

Interface. The use of a first-order method suggests a natural interface for specifying the convex flow problem (2.3). By definition, the function $g$ is easy to evaluate if the subproblems (4.3a), (4.3b), and (4.3c) are easy to evaluate. Given a way to evaluate the functions $\bar{U}, \bar{V}_{i}$, and $f_{i}$, and to get the values achieving the suprema in these subproblems, we can easily evaluate the function $g$ via (4.4) and its gradient $\nabla g$ via (4.8), which we write below:
$$
\begin{aligned}
& \nabla_{\nu} g(\nu, \eta)=y^{\star}-\sum_{i=1}^{m} A_{i} x_{i}^{\star} \\
& \nabla_{\eta_{i}} g(\nu, \eta)=x_{i}^{\star}-\tilde{x}_{i}^{\star}, \quad i=1, \ldots, m
\end{aligned}
$$
(Here, as before, $y^{\star}$ and the $\left\{x_{i}^{\star}\right\}$ and $\left\{\tilde{x}_{i}^{\star}\right\}$ are the optimal points for the subproblems (4.3), evaluated at $\eta$ and $\nu$.) Often $\bar{U}$ and $\bar{V}_{i}$, which are closely related to conjugate functions, have a closed form expression. In general, evaluating the support function $f_{i}$ requires solving a convex optimization problem. However, in many practical scenarios, this function either has a closed form expression or there is an easy and efficient iterative method for evaluating it. We discuss a method for quickly evaluating this function in the special case of two-node edges in §5.1 and implement more general subproblem solvers for the examples in §5.3.

Parallelization. The evaluation of $g(\nu, \eta)$ and $\nabla g(\nu, \eta)$ can be parallelized over all edges $i=1, \ldots, m$. The bulk of the computation is in evaluating the arbitrage subproblem $f_{i}$ for each edge $i$. The optimality conditions of the subproblem suggest a useful shortcut: if the vector 0 is in the normal cone of $T_{i}$ at $\tilde{x}_{i}^{\star}$, then the zero vector is a solution to the subproblem, i.e., the edge is not used (see (4.9)). Often, this condition is not only easier to evaluate than the subproblem itself, but also has a nice interpretation. For example, in the case of financial markets, this condition is equivalent to the prices $\eta$ being within the bid-ask spread of the market. We will see that, in many examples, these subproblems are quick to solve and have further structure that can be exploited.

\subsection*{5.1 Two-node edges}

In many applications, edges are often between only two vertices. Since these edges are so common, we will discuss how the additional structure allows the arbitrage problem (4.3c) to be solved quickly for such edges. Some practical examples will be given later in the numerical examples in §5.3. In this section, we will drop the index $i$ with the understanding that we are referring to the flow along a particular edge. Note that much of this section is similar to some of the authors' previous work $[$ Dia $+23, \S 3]$, where two-node edges were explored in the context of the CFMM routing problem, also discussed in §3.5.

\subsection*{5.1.1 Gain functions}

To efficiently deal with two-node edges, we will consider their gain functions, which denote the maximum amount of output one can receive given a specified input. Note that our gain function is equivalent to the concave gain functions introduced in [Shi06], and, in the case of asset markets, the forward exchange function introduced in $[\mathrm{Ang}+22 \mathrm{~b}]$. In what follows, the gain function $h: \mathbf{R} \rightarrow \mathbf{R} \cup\{-\infty\}$ will denote the maximum amount of flow that can be output, $h(w)$, if there is some amount amount $w$ of flow into the edge: i.e., if $T \subseteq \mathbf{R}^{2}$ is the set of allowable flows for an edge, then
$$
h(w)=\sup \{t \in \mathbf{R} \mid(-w, t) \in T\} .
$$
(If the set is empty, we define $h(w)=-\infty$.) In other words, $h(w)$ is defined as the largest amount that an edge can output given an input $(-w, 0)$. There is, of course, a natural 'inverse' function which takes in an output instead, but only one such function is necessary. Since the set $T$ is closed by assumption, the supremum, when finite, is achieved so we have that
$$
(-w, h(w)) \in T .
$$

We can also view $h$ as a specific parametrization of the boundary of the set $T$ that will be useful in what follows.

Lossless edge example. A simple practical example of a gain function is the gain function for an edge which conserves flows and has finite capacity, as in (3.1):
$$
T=\left\{z \in \mathbf{R}^{2} \mid 0 \leq z_{2} \leq b, \quad z_{1}+z_{2}=0\right\} .
$$

In this case, it is not hard to see that
$$
h(w)= \begin{cases}w & 0 \leq w \leq b  \tag{5.2}\\ -\infty & \text { otherwise }\end{cases}
$$

The fact that $h$ is finite only when $w \geq 0$ can be interpreted as 'the edge only accepts incoming flow in one direction.'

Nonlinear power loss. A more complicated example is the allowable flow set in the optimal power flow example (3.6), which is, for some convex function $\ell: \mathbf{R}_{+} \rightarrow \mathbf{R}$,
$$
T_{i}=\left\{z \in \mathbf{R}^{2} \mid-b \leq z_{1} \leq 0, \quad z_{2} \leq-z_{1}-\ell\left(-z_{1}\right)\right\} .
$$

The resulting gain function is again fairly easy to derive:
$$
h(w)= \begin{cases}w-\ell(w) & 0 \leq w \leq b \\ -\infty & \text { otherwise } .\end{cases}
$$

Note that, if $\ell=0$, then we recover the original lossless edge example. Figure 3.4 displays this set of allowable flows $T$ and the associated gain function $h$.

\subsection*{5.1.2 Properties}

The gain function $h$ is concave because the allowable flows set $T$ is convex, and we can interpret the positive directional derivative of $h$ as the current marginal price of the output flow, denominated in the input flow. Defining this derivative as
$$
\begin{equation*}
h^{+}(w)=\lim _{\delta \rightarrow 0^{+}} \frac{h(w+\delta)-h(w)}{\delta} \tag{5.3}
\end{equation*}
$$
then $h^{+}(0)$ is the marginal change in output if a small amount of flow were to be added when the edge is unused, while $h^{+}(w)$ denotes the marginal change in output for adding a small amount $\varepsilon>0$ to a flow of size $w$, for very small $\varepsilon$. In the case of financial markets, this derivative is sometimes referred to as the 'price impact function'. We define a reverse derivative:
$$
h^{-}(w)=\lim _{\delta \rightarrow 0^{+}} \frac{h(w)-h(w-\delta)}{\delta}
$$
which acts in much the same way, except the limit is approached in the opposite direction. (Both limits are well defined as they are the limits of functions monotone on $\delta$ since $h$ is concave.) Note that, if $h$ is differentiable at $w$, then, of course, the left and right limits are equal to the derivative,
$$
h^{\prime}(w)=h^{+}(w)=h^{-}(w),
$$
but this need not be true since we do not assume differentiability of the function $h$. Indeed, in many standard applications, $h$ is piecewise linear and therefore unlikely to be differentiable at optimality. On the other hand, since the function $h$ is concave, we know that
$$
h^{+}(w) \leq h^{-}(w)
$$
for any $w \in \mathbf{R}$.
Two-node subproblem. Equipped with the gain function, we can specialize the problem (4.3c). We define the arbitrage problem (4.3c) for an edge with gain function $h$ as
$$
\begin{equation*}
\operatorname{maximize}-\eta_{1} w+\eta_{2} h(w) \tag{5.4}
\end{equation*}
$$
with variable $w \in \mathbf{R}$. Since $h$ is concave, the problem is a scalar convex optimization problem, which can be easily solved by bisection (if the function $h$ is subdifferentiable) or ternary search. Since we know that $\eta \geq 0$ by the constraints of the dual problem (4.7), the optimal value of this problem (5.4) and that of the subproblem (4.3c) are identical.

Optimality conditions. The optimality conditions for problem (5.4) are that $w^{\star}$ is a solution if, and only if,
$$
\begin{equation*}
\eta_{2} h^{+}\left(w^{\star}\right) \leq \eta_{1} \leq \eta_{2} h^{-}\left(w^{\star}\right) \tag{5.5}
\end{equation*}
$$

If the function $h$ is differentiable then $h^{+}=h^{-}=h^{\prime}$ and the expression above simplifies to finding a root of a monotone function:
$$
\begin{equation*}
\eta_{2} h^{\prime}\left(w^{\star}\right)=\eta_{1} . \tag{5.6}
\end{equation*}
$$

If there is no root and condition (5.5) does not hold, then $w^{\star}= \pm \infty$. However, the solution will be finite for any feasible flow set that does not contain a line; i.e., if the edge cannot create 'infinite flow'.

No-flow condition. The inequality (5.5) gives us a simple way of verifying whether we will use an edge with allowable flows $T$, given some prices $\eta_{1}$ and $\eta_{2}$. In particular, not using this edge is optimal whenever
$$
h^{+}(0) \leq \frac{\eta_{1}}{\eta_{2}} \leq h^{-}(0)
$$

We can view the interval $\left[h^{+}(0), h^{-}(0)\right]$ as a 'no-flow interval' for the edge with feasible flows $T$. (In many markets, for example, this interval is a bid-ask spread related to the fee required to place a trade.) This 'no-flow condition' lets us save potentially wasted effort of computing an optimal arbitrage problem, as most flows in the original problem will be 0 in many applications. In other words, an optimal flow often will not use most edges.

\subsection*{5.1.3 Bounded edges}

In some cases, we can similarly easily check when an edge will be saturated. We say an edge is bounded in forward flow if there is a finite $w^{0}$ such that $h\left(w^{0}\right)=\sup h$; i.e., there is a finite input $w^{0}$ which will give the maximum possible amount of output flow. An edge is bounded if it is bounded in forward flow by $w^{0}$ and if the set $\operatorname{dom} h \cap\left(-\infty, w^{0}\right]$ is bounded. Capacity constraints, such as those of (3.1), imply an edge is bounded.

Minimum supported price. In the dual problem, a bounded edge then has a notion of a 'minimum price'. First, define
$$
w^{\max }=\inf \{w \in \mathbf{R} \mid h(w)=\sup h\}
$$
i.e., $w^{\text {max }}$ is the smallest amount of flow that can be tendered to maximize the output of the provided edge. We can then define the minimum supported price as the left derivative of $h$ at $w^{\text {max }}$, which is written $h^{-}\left(w^{\text {max }}\right)$, from before. The first-order optimality conditions imply that $w^{\text {max }}$ is a solution to the scalar optimal arbitrage problem (5.4) whenever
$$
h^{-}\left(w^{\max }\right) \geq \frac{\eta_{1}}{\eta_{2}}
$$

In English, this can be stated as: if the minimum supported marginal price we receive for $w^{\text {max }}$ is still larger than the price being arbitraged against, $\eta_{1} / \eta_{2}$, it is optimal use all available flow in this edge.

Active interval. Defining $w^{\text {min }}$ as
$$
w^{\min }=\inf \left(\operatorname{dom} h \cap\left(-\infty, w^{\max }\right]\right)
$$
where we allow $w^{\text {min }}=-\infty$ to mean that the edge is not bounded. We then find that the full problem (5.4) needs to be solved only when
$$
\begin{equation*}
h^{-}\left(w^{\max }\right)<\frac{\eta_{1}}{\eta_{2}}<h^{+}\left(w^{\min }\right) \tag{5.7}
\end{equation*}
$$

We will call this interval of prices the active interval for an edge, as the optimization problem (5.4) only needs to be solved when the prices $\eta$ are in the interval (5.7), otherwise, the solution is one of $w^{\text {min }}$ or $w^{\text {max }}$.

\subsection*{5.2 Restoring primal feasibility}

Unfortunately, dual decomposition methods do not, in general, find a primal feasible solution; given optimal dual variables $\eta^{\star}$ and $\nu^{\star}$ for the dual problem (4.7), it is not the case that all solutions $y^{\star}, x^{\star}$, and $\tilde{x}^{\star}$ for the subproblems (4.3) satisfy the constraints of the original augmented problem (4.1). Indeed, we are guaranteed only that some solution to the subproblems satisfies these constraints. We develop a second phase of the algorithm to restore primal feasibility.

For this section, we will assume that the net flow utility $U$ is strictly concave, and that the edge utilities $\left\{V_{i}\right\}$ are each either strictly concave or identically zero. If $V_{i}$ (or $U$ ) is nonzero, then it has a unique solution for its corresponding subproblem at the optimal dual variables. This, in turn, implies that the solutions to the dual subproblems are feasible, and therefore optimal, for the primal problem. However, when some edge utilities are zero and the corresponding sets of allowable flows are not strictly convex, we must take care to recover edge flows that satisfy the net flow conservation constraint.

We note that, if the $\left\{V_{i}\right\}$ are all strictly concave (i.e., none are equal to zero) with no restrictions on $U$, one may directly construct a solution ( $y,\left\{x_{i}\right\}$ ) by setting $x_{i}=\tilde{x}_{i}^{\star}$, the solutions to the arbitrage subproblems for optimal dual variables $\eta^{\star}$ and $\nu^{\star}$. We can then set
$$
y=\sum_{i=1}^{m} A_{i} x_{i}
$$
to get feasible - and therefore optimal-flows for problem (2.3).
Example. Consider a lossless edge with capacity constraints, which has the allowable flow set
$$
\begin{equation*}
T=\left\{\left(z_{1}, z_{2}\right) \mid 0 \leq z_{2} \leq b, \quad z_{1}+z_{2}=0\right\} . \tag{5.8}
\end{equation*}
$$

The associated gain function is $h(w)=w$, if $0 \leq w \leq b$, and $h(w)=-\infty$ otherwise. This gives the arbitrage problem (5.4) for the lossless edge
$$
\begin{array}{ll}
\operatorname{maximize} & -\eta_{1} w+\eta_{2} w \\
\text { subject to } & 0 \leq w \leq b
\end{array}
$$

Proceeding analytically, we see that the optimal solutions to this problem are
$$
w^{\star} \in \begin{cases}\{0\} & \eta_{1}>\eta_{2} \\ \{b\} & \eta_{1}<\eta_{2} \\ {[0, b]} & \eta_{1}=\eta_{2}\end{cases}
$$

In words, we will either use the full edge capacity if $\eta_{1}<\eta_{2}$, or we will not use the edge if $\eta_{1}>\eta_{2}$. However, if $\eta_{1}=\eta_{2}$, then any usage from zero up to capacity is an optimal solution for the arbitrage subproblem. Unfortunately, not all of these solutions will return a primal feasible solution for the original problem (4.1).

Dual optimality. More generally, given an optimal dual point $\left(\nu^{\star}, \eta^{\star}\right)$, an optimal flow over edge $i$ (i.e., a flow that solves the original problem (2.3)) given by $x_{i}^{\star}$, will satisfy
$$
x_{i}^{\star} \in \partial f_{i}\left(\eta_{i}^{\star}\right)
$$
by strong duality, as does the solution $\tilde{x}_{i}^{\star}$ to the arbitrage subproblem (4.3c),
$$
\tilde{x}_{i}^{\star} \in \partial f_{i}\left(\eta_{i}^{\star}\right)
$$
by definition. The subdifferential $\partial f_{i}\left(\eta_{i}^{\star}\right)$ is a closed convex set, as it is the intersection of hyperplanes defined by subgradients. We distinguish between two cases. First, if the set $T_{i}$ is strictly convex, then the set $\partial f_{i}\left(\eta_{i}^{\star}\right)$ consists of a single point and $x_{i}^{\star}=\tilde{x}_{i}^{\star}$. However, if $T_{i}$ is not strictly convex, then we only are guaranteed that
$$
x_{i}^{\star} \in T_{i}^{\star}\left(\eta_{i}^{\star}\right)=T_{i} \cap \partial f_{i}\left(\eta_{i}^{\star}\right) .
$$

This set $T_{i}^{\star}\left(\eta_{i}^{\star}\right)$ is the intersection of two convex sets and, therefore, is convex. In fact, this set is exactly a 'face' of $T_{i}$ with supporting hyperplane defined by $\eta_{i}^{\star}$. In general, this set can be as hard to describe as the original set $T_{i}$. On the other hand, in the common case that $T_{i}$ is polyhedral or two-dimensional, the set has a concise representation that is easier to optimize over than the set itself. (Note that, in practice, numerical precision issues may also need to be taken into account, as we only know $\eta_{i}^{\star}$ up to some tolerance.)

Two-node edges. For two-node edges, observe that a piecewise linear set of allowable flows $T_{i}$ can be written as a Minkowski sum of its segments. Equivalently, a piecewise linear gain function is equivalent to adding bounded linear edges for each of its segments (cf. (3.1)). For a given optimal price vector $\eta_{i}^{\star}$, the optimal flow $x_{i}^{\star}$ will be nonzero on at most one of these segments, and the set $T_{i}^{\star}$ is a single point unless $\eta_{i}^{\star}$ is normal to one of these line segments. This idea, of course, may be extended to general two-node allowable flows whose boundary may include smooth regions as well as line segments. Returning to example (5.8) above, if $\eta_{i}^{\star}=\alpha \mathbf{1}$ for some $\alpha>0$, then
$$
T_{i}^{\star}\left(\eta_{i}^{\star}\right)=\left\{z \in \mathbf{R}^{2} \mid \mathbf{1}^{T} z=0, \quad 0 \leq z_{2} \leq b\right\}
$$

Otherwise, $T_{i}^{\star}\left(\eta_{i}^{\star}\right)$ is an endpoint of this line segment: either
$$
T_{i}^{\star}\left(\eta_{i}^{\star}\right)=\{(0,0)\}
$$
or
$$
T_{i}^{\star}\left(\eta_{i}^{\star}\right)=\{(-b, b)\} .
$$

Recovering the primal variables. Recall that the objective function $U$ is strictly concave by assumption, so there is a unique solution that solves the associated subproblem (4.3a) at optimality. Let $S$ be a set containing the indices of the strictly convex feasible flows; that is, the index $i \in S$ if $T_{i}$ is strictly convex. Now, let the dual optimal points be ( $\nu^{\star}, \eta^{\star}$ ), and
the optimal points for the subproblems (4.3a) and (4.3c) be $y^{\star}$ and $\tilde{x}_{i}^{\star}$ respectively. We can then recover the primal variables by solving the problem
$$
\begin{array}{ll}
\operatorname{minimize} & \left\|y^{\star}-\sum_{i=1}^{m} A_{i} x_{i}\right\| \\
\text { subject to } & x_{i}=\tilde{x}_{i}^{\star}, \quad i \in S \\
& x_{i} \in T_{i}^{\star}\left(\eta_{i}^{\star}\right), \quad i \notin S .
\end{array}
$$

Here, the objective is to simply find a set of feasible $x_{i}$ (i.e., that 'add up' to $y^{\star}$ ) which are consistent with the dual prices discovered by the original problem, in the sense that they minimize the error between their net flows and the net flow vector $y^{\star}$. Indeed, if the problem is correctly specified (and solution errors are not too large), the optimal value should always be 0 . When the sets $\left\{T_{i}^{\star}\right\}$ can be described by linear constraints and we use the $\ell_{1}$ or $\ell_{\infty}$ norm, this problem is a linear program and can be solved very efficiently. The two-node linear case is most common in practice, and we leave further exploration of the reconstruction problem to future work.

\subsection*{5.3 Numerical examples}

We illustrate our interface by revisiting some of the examples in §3. We do not focus on the linear case, as this case is better solved with special-purpose algorithms such as the network simplex method or the augmenting path algorithm. In all experiments, we examine the convergence of our method, ConvexFlows, and compare its runtime to the commercial solver Mosek [ApS24a], accessed through the JuMP modeling language [DHL17; Leg+21; Lub +23 ]. We note that the conic formulations of these problems often do not preserve the network structure and may introduce a large number of additional variables.

Our method ConvexFlows is implemented in the Julia programming language [Bez+17], and the package may be downloaded from
https://github.com/tjdiamandis/ConvexFlows.jl.
Code for all experiments is in the paper directory of the repository. All experiments were run using ConvexFlows v0.1.1 on a MacBook Pro with a M1 Max processor ( 8 performance cores) and 64 GB of RAM. We suspect that our method could take advantage of further parallelization than what is available on this machine, but we leave this for future work.

\subsection*{5.3.1 Optimal power flow}

We first consider the optimal power flow problem from §3.2. This problem has edges with only two adjacent nodes, but each edge flow has a strictly concave gain function due to transmission line loss. These line losses are given by the constraint set (3.6), and we use the objective function (3.7) with the quadratic power generation cost functions
$$
c_{i}(w)= \begin{cases}(1 / 2) w^{2} & w \geq 0 \\ 0 & w<0\end{cases}
$$

Since the flow cost functions are identically zero, we only have two subproblems (cf., §4.3.1). The first subproblem is the evaluation of $\bar{U}$, which can be worked out in closed form:
$$
\bar{U}(\nu)=(1 / 2)\|\nu\|_{2}^{2}-d^{T} \nu,
$$
with domain $\nu \geq 0$. We could easily add additional constraints, such as an upper bound on power generation, but do not for the sake of simplicity. The second subproblem is the arbitrage problem (5.4),
$$
f_{i}\left(\eta_{i}\right)=\sup _{0 \leq w \leq b_{i}}\left\{-\eta_{1} w+\eta_{2}\left(w-\ell_{i}(w)\right)\right\},
$$
which can generally be solved as a single-variable root finding problem because the allowable flows set is strictly convex. Here, the edge is, in addition, 'bounded' (see §5.1) with a closed form solution. We provide the details in appendix B.1.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-067.jpg?height=710&width=713&top_left_y=1039&top_left_x=708}
\captionsetup{labelformat=empty}
\caption{Figure 5.1: Sample network for $n=100$ nodes.}
\end{figure}

Problem data. We model the network as in $[\mathrm{Kra}+13]$ with the same parameters, which results in a network with high local connectivity and a few longer transmission lines. Figure 5.1 shows an example with $n=100$. We draw the demand $d_{i}$ for each node uniformly at random from the set $\{0.5,1,2\}$. For each transmission line, we set $\alpha_{i}=16$ and $\beta_{i}=1 / 4$. We draw the maximum capacity for each line uniformly at random from the set $\{1,2,3\}$. These numbers imply that a line with maximum capacity 1 operating at full capacity will loose about $10 \%$ of the power transmitted, whereas a line with maximum capacity 3 will loose almost $40 \%$ of the power transmitted. For the purposes of this example, we let all lines be bidirectional: if there is a line connecting node $j$ to node $j^{\prime}$, we add a line connecting node $j^{\prime}$ to node $j$ with the same parameters.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-068.jpg?height=531&width=1636&top_left_y=249&top_left_x=244}
\captionsetup{labelformat=empty}
\caption{Figure 5.2: Convergence of ConvexFlows with $n=100$ using L-BFGS-B (left) and BFGS (right). On the left, the objective is compared to a high-precision solution from Mosek. The primal residual measures the net flow constraint violation, with $\left\{x_{i}\right\}$ from (4.3c) and $y$ from (4.3a). Note the linear convergence of L-BFGS and the superlinear convergence of BFGS.}
\end{figure}

\section*{Numerical results.}

We first examine the convergence per of our method on an example with $n=100$ nodes and $m=198$ transmission lines. In figure 5.2, we plot the relative duality gap, net flow constraint violation, and difference between our objective value and the 'optimal' objective value, obtained using the commercial solver Mosek. (See appendix B. 1 for the conic formulation). The primal feasible point used to compute the relative duality gap and the primal residual is constructed as
$$
\hat{y}=\sum_{i=1}^{m} A_{i} \tilde{x}_{i},
$$
where $\tilde{x}_{i}$ solves the subproblem (4.3c) with the current dual variable iterate. Note that the point ( $\hat{y}, \tilde{x}$ ) is feasible by construction.

The results in the left figure 5.2 suggest that our method, when using L-BFGS-B enjoys linear convergence. The difference in objective value at 'optimality' is likely due to floating point numerical inaccuracies, as it is below the square root of machine precision, denoted by $\sqrt{\text { eps }}$. For this small example, we also examine convergence using full BFGS , with a line search modified to enforce positivity constraints (discussed further in 6.2). In figure 5.2, we see that BFGS appears to enjoy quadratic convergence. We emphasize, however, that BFGS does not scale well to large networks, compared to L-BFGS.

In figure 5.3, we compare the runtime of our method with L-BFGS-B to Mosek for increasing values of $n$, with ten trials for each value. For each $n$, we plot the median time, the 25th to 75th quantile, and the maximum time. Our method clearly results in a significant and consistent speedup over all problem sizes. Notably, our method exhibits less variance in solution time as well. We emphasize, however, that our implementation is not highly optimized and relies on an 'off-the-shelf' L-BFGS-B solver. We expect that further software improvement could yield even better performance.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-069.jpg?height=760&width=1140&top_left_y=259&top_left_x=485}
\captionsetup{labelformat=empty}
\caption{Figure 5.3: Comparison of ConvexFlows with Mosek. Lines indicate the median time over 10 trials, and the shaded region indicates the 25th to 75 th quantile range. Dots indicate the maximum time over the 10 trials.}
\end{figure}

\subsection*{5.3.2 Routing orders through financial exchanges}

Next, we consider a problem which includes both edges connecting more than two nodes and utilities on the edge flows: routing trades through constant function market makers (see §3.5). For all experiments, we use the net flow utility function
$$
U(y)=c^{T} y-I_{\mathbf{R}_{+}^{n}}(y)
$$

We interpret this function as finding arbitrage in the network of markets. More specifically, we wish to find the most valuable trade $y$, measured according to price vector $c$, which, on net, tenders no assets to the network. The associated subproblem (4.3a) can be computed as
$$
\bar{U}(\nu)= \begin{cases}0 & \nu \geq c \\ \infty & \text { otherwise }\end{cases}
$$

We also want to ensure our trade with any one market is not too large, so we add a penalty term to the objective:
$$
V_{i}\left(x_{i}\right)=-(1 / 2)\left\|\left(x_{i}\right)-\right\|_{2}^{2}
$$
(Recall that negative entries denote assets tendered to an exchange.) The associated subproblem is
$$
\bar{V}_{i}(\xi)=(1 / 2)\|\xi\|_{2}^{2}
$$

Finally, the arbitrage problem here is exactly the problem of computing an optimal arbitrage trade with each market, given prices on some infinitely-liquid external market.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-070.jpg?height=594&width=1647&top_left_y=240&top_left_x=240}
\captionsetup{labelformat=empty}
\caption{Figure 5.4: Convergence of ConvexFlows on an example with $n=100$ assets and $m=2500$ markets.}
\end{figure}

Problem data. We generate $m$ markets and $n=2 \sqrt{m}$ assets. We use three popular market implementations for testing: Uniswap (v2) [ZCP18], Balancer [MM19a] two-asset markets, and Balancer three-asset markets. We provide the details of these markets and the associated arbitrage problems in appendix B.2. Markets are Uniswap-like with probability 2/5, Balancer-like two-asset markets with probability $2 / 5$, and Balancer-like three-asset markets with probability $1 / 5$. Each market $i$ connects randomly selected assets and has reserves sampled uniformly at random from the interval $[100,200]^{n_{i}}$.

Numerical results. We first examine the convergence of our method on an example with $m=2500$ and $n=100$. In figures 5.4a and 5.4b, we plot the convergence of the relative duality gap, the feasibility violation of $y$, and the relative difference between the current objective value and the optimal objective value, obtained using the commercial solver Mosek. (See appendix B. 2 for the conic formulation we used.) Note that, here, we reconstruct $y$ as
$$
y=\sum_{i=1}^{m} A_{i} x_{i},
$$
instead of using the solution to the subproblem (4.3a) as we did in the previous example. As a result, this $y$ satisfies the net flow constraint by construction. We measure the feasibility violation relative to the implicit constraint in the objective function $U$, which is that $y \geq$ 0 . We again see that our method enjoys linear convergence in both cases; however, the convergence is significantly slower when edge objectives are added (figure 5.4b). We then compare the runtime of our method and the commercial solver Mosek, both without edge penalties and with only two-node edges, in figure 5.5. Again, ConvexFlows enjoys a significant speedup over Mosek.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-071.jpg?height=785&width=1140&top_left_y=889&top_left_x=485}
\captionsetup{labelformat=empty}
\caption{Figure 5.5: Comparison of ConvexFlows and Mosek for $m$ varying from 100 to 100,000 and $n=2 \sqrt{m}$. Lines indicate the median time over 10 trials, and the shaded region indicates the 25th to 75th quantile range. Dots indicate the maximum time over the 10 trials.}
\end{figure}

\section*{Chapter 6}

\section*{Solver: ConvexFlows.jl}

The core of ConvexFlows. jl solves (2.3) using the L-BFGS-B algorithm $[\mathrm{Byr}+95 ; \mathrm{Zhu}+97 ;$ MN11] on the transformed dual problem (5.1). In general, the solver requires user-specified definitions of the subproblems (4.3). However, it is unreasonable to expect an average, unsophisticated user to directly specify conjugate-like functions and solutions to convex optimization problems as in (4.3a) and (4.3c). In ConvexFlows.jl, we developed an interface that allows the user to choose from a set of utility functions $U$ and directly specify the edge gain functions $h_{i}$ for each edge $i=1, \ldots, m$ for two-node edges. With these inputs, our software can specify the subproblems and their maximizers automatically on behalf of the user.

In this chapter, we discuss the user-friendly interface. We note that this interface has some restrictions: it may only be used for convex flow problems with two-node edges and without edge flow costs. Adding these capabilities to ConvexFlows. jl is straightforward, but we leave this task for future work. We conclude this chapter with several small examples using our interface, including a decentralized exchange order routing problem, as in §3.5, a market clearing problem, as in §3.4, and a optimal power flow problem where we wish to devise a multi-step power generation plan for a heterogenous network with storage capacity, which extends the setup of §3.2.

For more details on the interface and code examples, please check out the documentation:
https://tjdiamandis.github.io/ConvexFlows.jl/dev/.

\subsection*{6.1 Interface}

\subsection*{6.1.1 The first subproblem}

The first subproblem (4.3a) typically has a closed form expression. Since, from before, $\bar{U}(\nu)=(-U)^{*}(-\nu)$, where $U^{*}$ denotes the Fenchel conjugate of $U$, we can use standard results in conjugate function calculus to compute $\bar{U}$ from a number of simple 'atoms'. For example, $U$ is often separable, i.e., $U(y)=u_{1}\left(y_{1}\right)+\cdots+u_{n}\left(y_{n}\right)$, in which case we have that
$$
\bar{U}(y)=\bar{u}_{1}\left(y_{1}\right)+\cdots+\bar{u}_{n}\left(y_{n}\right),
$$
where $\bar{u}_{j}$ is defined similarly to $\bar{U}$ (see (4.3a)).

For computational reasons, we recommend using a strictly concave $U$. A non-strictly concave $U$ may be transformed into a strictly concave one by, for example, subtracting a very small quadratic term. This common trick (e.g., as used in the linear system solve in the popular OSQP solver [Ste+20]) helps avoid computational difficulties.

Atoms. A user may construct the utility function $U$ using any of the atoms provided by ConvexFlows.jl. Some examples of scalar utility atoms include the linear, nonnegative linear, and nonpositive quadratic atoms. (The nonpositive quadratic atom is useful for specifying nonnegative quadratic costs.) Since $U$ is nondecreasing, we can support lower bounds on the variables but not upper bounds.

General functions. While it is most straightforward to build $U$ (and therefore $\bar{U}$ ) from known atoms, more general functions without constraints may be handled by solving the first subproblem (4.3a) directly. A vector $\tilde{y}$ achieving the supremum must satisfy $\nabla U(\tilde{y})=\nu$ (for a differentiable $U)$. This equation may be solved via Newton's method, and the gradient and Hessian may be computed via automatic differentiation.

Functions with constraints. A user may incorporate constraints in the objective by writing $U$ as the solution to a conic optimization problem, which may be expressed using a modeling language such as JuMP [DHL17; Lub+23] or Convex. jl [Ude+14], both of which can compile problems into a standard conic form using MathOptInterface. jl [Leg+21]. Of course, in the most general case, computing $\bar{U}(y)$ and a subgradient requires solving a convex optimization problem with at least $n$ variables at each iteration, which may be slow.

\subsection*{6.1.2 The arbitrage subproblem}

For each edge $i$ we require the user to specify the gain function $h_{i}$, which can be done in native Julia code. Recall that the second subproblem (4.3c) can be written as
$$
\operatorname{maximize}-\eta_{1} w+\eta_{2} h(w)
$$
with variable $w$. Denote the solution point of the second problem (4.3c) by $w^{\star}$. We write $h^{+}(w)$ and $h^{-}(w)$ for the right and left derivatives of $h$ at $w$, respectively. Specifically, we define
$$
h^{+}(w)=\lim _{\delta \rightarrow 0^{+}} \frac{h(w+\delta)-h(w)}{\delta}
$$
and $h^{-}(w)$ by
$$
h^{-}(w)=\lim _{\delta \rightarrow 0^{+}} \frac{h(w)-h(w-\delta)}{\delta}
$$

The optimality conditions for problem (4.3c) are then that $w^{\star}$ is a solution if, and only if,
$$
\begin{equation*}
h^{+}\left(w^{\star}\right) \leq \eta_{1} / \eta_{2} \leq h^{-}\left(w^{\star}\right) \tag{6.1}
\end{equation*}
$$
(We may assume $\eta_{2}>0$ for any increasing $U$, since $\nu>0$.) Note that the optimality condition suggests a simple method to check if an edge will be used at all: zero flow is optimal if and only if
$$
h^{+}(0) \leq \eta_{1} / \eta_{2} \leq h^{-}(0)
$$

This 'no flow condition' is often much easier to check in practice than solving the complete subproblem and allows the solver to 'short circuit' many edge computations.

If the zero flow is not optimal, then we can solve the arbitrage subproblem (4.3c) via a one-dimensional root-finding method. We assume that $h$ is differentiable almost everywhere (e.g., $h$ is a piecewise smooth function) and use bisection search or Newtown's method to find a $w^{\star}$ that satisfies (6.1). Since we use directed edges, and typically an upper bound $b$ on the flow exists for physical systems, we begin with the bounds $(0, b)$ and terminate after $\log _{2}(b / \varepsilon)$ iterations for a small tolerance $\varepsilon$. (If no bound is specified, an upper bound $b$ may be computed with, for example, a doubling method.) We compute the first derivative of $h$ using forward mode automatic differentiation, implemented in ForwardDiff.jl [RLP16]. Computing a derivative can be done simultaneously with a function evaluation and, as a result, these subproblems can be solved very quickly. Alternatively, a user may specify a closed-form solution to the subproblem, which exists for many problems in practice (see, for example, the examples in §5.3.)

\subsection*{6.2 Algorithmic modifications}

For small to medium-sized problems specified by a user, we use the quasi-Newton method BFGS instead of L-BFGS-B, as BFGS has been shown to have superior performance for nonsmooth problems, compared to L-BFGS [LO13; AO21]. We again note that ConvexFlows.jl also includes an interface to L-BFGS-B [Byr+95; Zhu+97; MN11], which we recommend for larger problems, but this interface requires more a more sophisticated problem specification. Futhermore, L-BFGS-B may be less robust to nonsmoothness in the problem [AO21] and, consequently, require more user fine-tuning.

To use the unconstrained BFGS method, we implement the bracketing line search from Lewis and Overton [LO13], modified to prevent steps outside of the positive orthant. Specifically, we bound the step size to ensure that every iterate remains strictly positive. This approach keeps the problem otherwise unconstrained, which allows us to use BFGS. Since $U$ is strictly increasing, a solution cannot lie at the boundary. The bracketing line search also ensures that the step size satisfies the weak Wolfe conditions (see [NW06, §3]).

\subsection*{6.3 Simple examples}

In this section, we provide a number of simple examples using the ConvexFlows.jl interface.

\subsection*{6.3.1 Optimal power flow.}

First, we return to the optimal power flow example in §3.2 and §5.3.1. We wish to find a cost-minimizing power generation plan that meets demand over a network of generators and consumers connected by transmission lines. Given problem parameters demand d , line capacities ub, and graph adjacency matrix Adj, the entire optimal power flow problem may be defined and solved in less than ten lines of code:
```
# Parameters: demand d, graph Adj, upper bounds ub
obj = NonpositiveQuadratic(d)
h(w) = 3w - 16.0*(log1pexp(0.25 * w) - log(2))
lines = Edge[]
for i in 1:n, j in i+1:n
    Adj[i, j] \leq 0 && continue
    push!(lines, Edge((i, j); h=h, ub=ub[i]))
end
prob = problem(obj=obj, edges=lines)
result = solve!(prob)
```


In this example, we used the special function log1pexp from the LogExpFunctions package, which is a numerically well-behaved implementation of the function $x \mapsto \log (1+ e^{x}$ ). Since $h$ may be specified as native Julia code, using non-standard functions does not introduce any additional complexity.

In this case, the arbitrage problem has a closed-form solution, easily derived from the first-order optimality conditions. With a small modification, we can give this closed-form solution to the solver directly. We write this closed form solution as
```
# Closed for solution to the arbitrage problem, i.e. the wstar that solves
# h'(wstar) == ratio
function wstar(ratio, b)
    if ratio \geq 1.0
        return 0.0
    else
        return min(4.0 * log((3.0 - ratio)/(1.0 + ratio)), b)
    end
end
```


We only need to modify the line in which we define the edges, changing it to
```
push!(lines, Edge((i, j); h=h, ub=ub_i, wstar = @inline w -> wstar(w, ub_i)))
```


After these modifications, we lose little computational efficiency compared to specifying the solution to the arbitrage problem directly. However, if we specify the problem directly, we may pre-compute the zero-flow region, inside which a line is not used at all. Compare the code for this example with the code for the fully-specified example in §5.3.1, which can be found at
https://github.com/tjdiamandis/ConvexFlows.jl/tree/main/paper/opf.

\subsection*{6.3.2 Trading with constant function market makers}

Similarly, we can easily specify the problem of finding an optimal trade given a network of decentralized exchanges. Here, we assume that the constant function market makers are governed by the trading function (see §3.5 for additional discussion)
$$
\varphi(R)=\sqrt{R_{1} R_{2}} .
$$

This trading function results in the gain function
$$
h(w)=\frac{w R_{2}}{R_{1}+w},
$$
which one can easily verify is strictly concave and increasing for $w \geq 0$. Given the adjacency matrix Adj and constant function market maker reserves Rs, we may specify the problem of finding the optimal trade as
```
obj = Linear(ones(n));
h(w, R1, R2) = R2*w/(R1 + w)
cfmms = Edge[]
for (i, inds) in enumerate(edge_inds)
    i1, i2 = inds
    push!(cfmms, Edge((i1, i2); h=w->h(w, Rs[i][1], Rs[i][2]), ub=1e6))
    push!(cfmms, Edge((i2, i1); h=w->h(w, Rs[i][2], Rs[i][1]), ub=1e6))
end
prob = problem(obj=obj, edges=cfmms)
result = solve!(prob)
```


Here, our objective function $U(y)=\mathbf{1}^{T} y$ indicates that we value all tokens equally.

\subsection*{6.3.3 Market clearing}

Finally, we revisit the market clearing example from §3.4. In this example, the objective function includes a constraint, so we must specify it directly. However, we may still specify edge gain functions instead of specifying the arbitrage problems directly.

Recall that the objective function is given by
$$
U(y)=\sum_{i=1}^{n_{b}} c_{i} \log y_{i}-I\left(y_{n_{b}+1: n_{b}+n_{g}} \geq-1\right)
$$
where $n_{b}$ is the number of buyers, $n_{g}$ is the number of goods, and $c \in \mathbf{R}_{+}^{n_{b}}$ is a vector of budgets. We will define a struct MarketClearingObjective to hold the problem parameters and then define the methods $U$ to evaluate the objective, Ubar to evaluate the associated subproblem (4.3a), and $\nabla$ Ubar! to evaluate the gradient of the subproblem. The full implementation of the first subproblem is below.
```
const CF = ConvexFlows
struct MarketClearingObjective{T} <: Objective
    budget::Vector{T}
    nb::Int
    ng::Int
    \epsilon:: T
end
function MarketClearingObjective(budget::Vector{T}, nb::Int, ng::Int; tol=1e-8)
    where T
    @assert length(budget) == nb
    return MarketClearingObjective{T}(budget, nb, ng, tol)
end
Base.length(obj::MarketClearingObjective) = obj.nb + obj.ng
function CF.U(obj::MarketClearingObjective{T}, y) where T
    any(y[obj.nb+1] .< -1) && return -Inf
    return sum(obj.budget .* log.(y[1:obj.nb])) - obj.\epsilon/2*sum(abs2, y[obj.nb+1:
        end])
end
function CF.Ubar(obj::MarketClearingObjective{T}, \nu) where T
    return sum(log.(obj.budget ./ \nu[1:obj.nb]) .- 1) + sum(\nu[obj.nb+1:end])
end
function CF.\nablaUbar!(g, obj::MarketClearingObjective{T}, \nu) where T
    g[1:obj.nb] .= -obj.budget ./ \nu[1:obj.nb]
    g[obj.nb+1:end] .= 1.0
    return nothing
end
```


We specify the utility that buyer $b$ gets from good $g$ as
$$
h(w)=\sqrt{b+g w}-\sqrt{b} .
$$

With the objective defined, we may easily specify and solve this problem as before:
```
obj = MarketClearingObjective(budgets, nb, ng)
u(x, b, g) = sqrt(b + g*x) - sqrt(b)
edges = Edge[]
for b in 1:nb, g in 1:ng
    # ub arbitrary since 1 unit per good enforced in objective
    push!(edges, Edge((nb + g, b); h=x->u(x, b, g), ub=1e3))
end
prob = problem(obj=obj, edges=edges)
result = solve!(prob)
```


See the documentation for additional details and commentary.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-078.jpg?height=343&width=830&top_left_y=253&top_left_x=640}
\captionsetup{labelformat=empty}
\caption{Figure 6.1: Graph representation of a power network with three nodes over time. Each solid line corresponds to a transmission line edge, and each dashed line corresponds to a storage edge.}
\end{figure}

\subsection*{6.4 Example: multi-period optimal power flow}

Finally, we extend the optimal power flow problem from §3.2. to the multi-period setting. We aim to find a cost-minimizing plan to generate power, which may be transmitted over a network of $m$ transmission lines, to satisfy the power demand of $n$ regions over some number of time periods $T$. We again use the transport model for power networks along with a nonlinear transmission line loss function from [Stu19], which has been shown to be a good approximation of the DC power flow model.

The loss function models the phenomenon that, as more power is transmitted along a line, the line dissipates an increasing fraction of the power transmitted. Following [Stu19, §2], we use the convex, increasing loss function
$$
\ell_{i}(w)=\alpha_{i}\left(\log \left(1+\exp \left(\beta_{i} w\right)\right)-\log 2\right)-2 w,
$$
where $\alpha_{i}$ and $\beta_{i}$ are known constants for each line and satisfy $\alpha_{i} \beta_{i}=4$. The gain function of a line with input $w$ can then be written as
$$
h_{i}(w)=w-\ell_{i}(w) .
$$

Each line $i$ also has a maximum capacity, given by $b_{i}$. Figure 3.4 in §3.2 shows a power line gain function and its corresponding set of allowable flows. Note that this edge gain function is strictly concave and increasing.

Each node $i$ may also store power generated at time $t$ for use at time $t+1$. If $w$ units are stored, then $\gamma_{i} w$ units are available at time $t+1$ for some $\gamma \in[0,1)$. These parameters may describe, for example, the battery storage efficiency. We model this setup by introducing $T$ nodes in the graph for each node, with an edge from the $t$ th node to the $(t+1)$ th node corresponding to node $i$ with the appropriate linear gain function, as depicted in figure 6.1. (Note that, for numerical stability, we subtract a small quadratic term, $(\varepsilon / 2) w^{2}$, from the linear gain functions, where $\varepsilon$ is very small.)

At time $t=1, \ldots, T$, node $i=1, \ldots, n$ demands $d_{i t}$ units of power and can generate power $p_{i}$ at a cost $c_{i}: \mathbf{R} \rightarrow \mathbf{R}_{+}$, given by
$$
c_{i}(p)= \begin{cases}\left(\alpha_{i} / 2\right) p^{2} & p \geq 0 \\ 0 & p<0\end{cases}
$$
which is a convex, increasing function parameterized by $\alpha_{i}>0$. Power dissipation has no cost but also generates no profit. To meet demand, we must have that
$$
d=p+y, \quad \text { where } \quad y=\sum_{i=1}^{m} A_{i} x_{i} \text {. }
$$

In other words, the power produced, plus the net flow of power, must satisfy the demand in each node. We write the network utility function as
$$
U(y)=\sum_{t=1}^{T} \sum_{i=1}^{n}-c_{i}\left(d_{i t}-y_{i t}\right)
$$

Since $c_{i}$ is convex and nondecreasing in its argument, the utility function $U$ is concave and nondecreasing in $y$. This problem can then be cast as a special case of the convex flow problem (2.3).

Note that the subproblems associated with the optimal power flow problem may be worked out in closed form. The first subproblem is
$$
\bar{U}(\nu)=(1 / 2)\|\nu\|_{2}^{2}-d^{T} \nu
$$
with domain $\nu \geq 0$. The second subproblem is
$$
f_{i}\left(\eta_{i}\right)=\sup _{0 \leq w \leq b_{i}}\left\{-\eta_{1} w+\eta_{2}\left(w-\ell_{i}(w)\right)\right\} .
$$

Using the first order optimality conditions, we can compute the solution:
$$
w_{i}^{\star}=\left(4 \log \left(\frac{3 \eta_{2}-\eta_{1}}{\eta_{2}+\eta_{1}}\right)\right)_{\left[0, b_{i}\right]}
$$
where $(\cdot)_{\left[0, b_{i}\right]}$ denotes the projection onto the interval $\left[0, b_{i}\right]$. These closed form solutions can be directly specified by the user in ConvexFlows. jl for increased efficiency, as in the example in §6.3.1.

\section*{Problem data}

We create a hour-by-hour power generation plan for an example network with three nodes over a time period of 5 days. The first two nodes are users who consume power and have a sinusoidal demand with a period of 1 day. These users may generate power at a very high $\operatorname{cost}\left(\alpha_{i}=100\right)$. The third node is a generator, which may generate power at a low cost $\left(\alpha_{i}=1\right)$ and demands no power for itself. The parameters are defined as follows:
```
# Problem parameters
n = 3
days = 5
T = 24*days
N = n*T
```

```
d_user = sin.((1:T) .* 2\pi ./ 24) .+ 1.5
c_user = 100.0
d_gen = 0.0*ones(T)
c_gen = 1.0
d = vec(vcat(d_user', d_user', d_gen'))
c = repeat([c_user, c_user, c_gen], T)
obj = NonpositiveQuadratic(d; a=c)
```


Next, we build a network between these three nodes. We create the transmission line as in §6.3.1. Then we build the storage edges, which 'transmit' power from time $t$ to time $t+1$. We equip the second user with a battery, which can store power between time periods with efficiency $\gamma=1.0$. The network has a total of 360 nodes and 359 edges.
```
# Network: two nodes, both connected to generator
function build_edges(n, T; bat_node)
    net_edges = [(i,n) for i in 1:n-1]
    edges = Edge[]
    # Transmission line edges
    h(w) = 3w - 16.0*(log1pexp(0.25 * w) - log(2))
    function wstar( }\eta, b
        \eta \ 1.0 && return 0.0
        return min(4.0 * log((3.0 - η)/(1.0 + η)), b)
    end
    for (i,j) in net_edges
        bi = 4.0
        for t in 1:T
            it = i + (t-1)*n
            jt = j + (t-1)*n
            push!(edges, Edge((it, jt); h=h, ub=bi, wstar=\eta -> wstar(\eta, bi)))
            push!(edges, Edge((jt, it); h=h, ub=bi, wstar=η -> wstar(η, bi)))
        end
    end
    # Storage edges
    \epsilon = 1 \mathrm { e } - 2
    wstar_storage (\eta, \gamma, b) = \eta \}\gamma\mathrm{ ? 0.0:min(1/ t*(γ - η), b)
    # only node 2 has storage
    for t in 1:T-1
        it = bat_node + (t-1)*n
        it_next = bat_node + t*n
        \zeta \gamma i ~ = ~ 1 . 0 + m
        storage_capacity = 10.0
        push!(edges, Edge(
            (it, it_next);
            h= w -> \gammai*w - \epsilon/2*w^2,
            ub=storage_capacity,
            wstar = η -> wstar_storage( }\eta,\gammai, storage_capacity
```

```
        ))
    end
    return edges
end
```


With the hard work of defining the network completed, we can construct and solve the problem as before. We solve this problem with BFGS, as L-BFGS does not exhibit good convergence on our problem, which is consistent to the results in [AO21]. The (almost) linear edges mean this problem is (almost) nonsmooth.
```
edges = build_edges(n, T, bat_node=2)
prob = problem(obj=obj, edges=edges)
result_bfgs = solve!(prob; method=:bfgs)
```


\section*{Numerical results}

We display the minimum cost power generation schedule in figure 6.2. Notice that during period of high demand, the first user must generate power at a high cost. The second user, on the other hand, purchases more power during periods of low demand to charge their battery and then uses this stored power during periods of high demand. As a result, the power purchased by this user stays roughly constant over time, after some initial charging.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-082.jpg?height=1579&width=1375&top_left_y=516&top_left_x=378}
\captionsetup{labelformat=empty}
\caption{Figure 6.2: Power generated (top), power used by the first node (middle) and by the second node, which has a battery (bottom).}
\end{figure}

\section*{Chapter 7}

\section*{Fixed edge fees}

Now that we have looked at the convex network flow problem (2.3) and its conic form (2.4), we will consider the convex network flow problem with fixed fees for the use of an edge. In particular, we consider the following extension of the convex network flow problem (2.3), which we call the network flow problem with fees:
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right)+q_{i} \lambda_{i} \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i}  \tag{7.1}\\
& \left(x_{i}, \lambda_{i}\right) \in\{0\} \cup\left(T_{i} \times\{-1\}\right), \quad i=1, \ldots, m
\end{array}
$$
where the set up and variables are exactly those of the original convex flow problem (2.3) except with the additional variable $\lambda \in \mathbf{R}^{m}$ and the problem data has the additional fee vector $q \in \mathbf{R}_{+}^{m}$. We note that this problem is not convex since the constraint set is not convex (in fact, this constraint set is not even connected!) and the problem is NP-hard to solve, which we prove shortly. Note that this objective is also nondecreasing in all of its variables (as $V_{i}$ and $U$ are, along with the fact that $q \geq 0$ ).

Interpretation. The interpretation of the constraint
$$
\begin{equation*}
\left(x_{i}, \lambda_{i}\right) \in\{0\} \cup\left(T_{i} \times\{-1\}\right) \tag{7.2}
\end{equation*}
$$
is that if $x_{i}$ is nonzero, then $\lambda_{i}=-1$. In other words, if we use edge $i$ by putting any nonzero flow through it, then $x_{i} \neq 0$ and we are charged $q_{i} \geq 0$ for its use. In general, we note that if $q_{i}>0$ and $x_{i}=0$, then we will have $\lambda_{i}=0$ at optimality, so we may view $\lambda_{i}$ as a variable that indicates whether or not edge $i$ is being used.

NP-hardness. We show that the network flow problem with fees is NP-hard by reducing the knapsack problem, which is known to be NP-hard [Kar72], to an instance of (7.1). The knapsack problem is the following: given a vector of nonnegative integers $c \in \mathbb{Z}_{+}^{m}$ and some integer $b \geq 0$, find a binary vector $z \in\{0,1\}^{m}$ such that $c^{T} z=b$. This problem can be reduced to an instance of (7.1) with $n=1$ by setting $U(y)=y-I(y \geq b), A_{i}=1 \in \mathbf{R}$,
$V_{i}=0, T_{i}=\left\{z \mid z \leq c_{i}\right\}$, and $q=c$. The problem becomes
$$
\begin{array}{ll}
\operatorname{maximize} & y-I(y \geq b)+c^{T} \lambda \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i} \\
& \left(x_{i}, \lambda_{i}\right) \in\{(0,0)\} \cup\left(\left(-\infty, c_{i}\right] \times\{-1\}\right), \quad i=1, \ldots, m
\end{array}
$$

We have that the objective is
$$
y+c^{T} \lambda=\sum_{i=1}^{m}\left(-\lambda_{i}\right) x_{i}+c^{T} \lambda \leq \sum_{i=1}^{m} c_{i} \lambda_{i}(-1+1)=0
$$

Since $y$ is constrained to be at least $b$, this problem has optimal value 0 if and only if there exists a solution to the knapsack problem. Therefore, the network flow problem with fees is NP-hard.

\subsection*{7.1 Integrality constraint}

For the sake of convenience, we will define the set
$$
Q_{i}=\{0\} \cup\left(T_{i} \times\{-1\}\right),
$$
such that the constraint (7.2) can be written as
$$
\left(x_{i}, \lambda_{i}\right) \in Q_{i},
$$
for each $i=1, \ldots, m$. In a certain sense, this constraint encodes the 'hard' part of the problem: if the set $Q_{i}$ were convex, then the problem would almost be a special case of the original convex flow problem (2.3), by pulling the constraint that $\lambda_{i} \geq-1$ into the objective.

Convex relaxation. Given the above discussion, the next natural thing to do is to write the convex hull of $Q_{i}$ : if we can easily write this convex hull in a compact way, then we immediately have a convex relaxation of the potentially hard problem (7.1). In general, finding the convex hull of a set can be at least as hard as solving the original. (In a number of senses: for example, it may require an exponential number of constraints.) In this particular special case, we will show that the convex hull of the set $Q_{i}$ is intimately related to the flow cone (2.6) introduced in the rewriting of the original convex flow problem (2.3) into the conic flow problem (2.4). In many practical scenarios, finding the flow cone $K_{i}$ corresponding to the allowable flows $T_{i}$ is fairly straightforward, which, in turn, means that finding the convex hull of $Q_{i}$ is also fairly straightforward.

\subsection*{7.2 Convex hull}

Given the above discussion, we will show that the convex hull of $Q_{i}$, written $\boldsymbol{\operatorname { c o n v }}\left(Q_{i}\right)$, is equal to all elements of the corresponding flow cone $K_{i}$ whose last entry (corresponding to

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-085.jpg?height=390&width=1442&top_left_y=330&top_left_x=352}
\captionsetup{labelformat=empty}
\caption{Figure 7.1: The set $Q_{i}$ (left) and its convex hull $\operatorname{conv}\left(Q_{i}\right)$ (right).}
\end{figure}
$\lambda_{i}$ ) lies between -1 and 0 . Written out:
$$
\operatorname{conv}\left(Q_{i}\right)=K_{i} \cap\left(\mathbf{R}^{n} \times[-1,0]\right) .
$$

See figure 7.1 for an example. As a reminder, the flow cone $K_{i} \subseteq \mathbf{R}^{n+1}$ for a given allowable flow set $T_{i} \subseteq \mathbf{R}^{n}$ is defined, using (2.6), as
$$
K_{i}=\mathbf{c l}\left\{(x,-\lambda) \in \mathbf{R}^{n} \times \mathbf{R} \mid x / \lambda \in T_{i}, \lambda>0\right\} .
$$

Finally, to simplify notation, we define
$$
\begin{equation*}
\bar{K}_{i}=K_{i} \cap\left(\mathbf{R}^{n} \times[-1,0]\right), \tag{7.3}
\end{equation*}
$$
which is the cone $K_{i}$ with the last element restricted to lie between -1 and 0 . Of course, this set is also convex as it is the intersection of two convex sets.

Reverse inclusion. First, we show the reverse inclusion: that $\bar{K}_{i} \subseteq \boldsymbol{\operatorname { c o n v }}\left(Q_{i}\right)$. Let $(x,-\lambda) \in \bar{K}_{i}$ (note the negative here) with $\lambda>0$, then, we will show that ( $x,-\lambda$ ) can be written as the convex combination of one element in $T_{i} \times\{-1\}$ and 0 and so also must lie in the convex hull of $Q_{i}, \boldsymbol{\operatorname { c o n v }}\left(Q_{i}\right)$. By definition, if $(x,-\lambda) \in \bar{K}_{i}$, then $x / \lambda \in T_{i}$ for $0<\lambda \leq 1$. But, this is the same as saying
$$
(x / \lambda,-1) \in Q_{i} .
$$

Finally, since $0 \in Q_{i}$, then any convex combination of 0 and ( $x / \lambda,-1$ ) is in the convex hull of $Q_{i}, \boldsymbol{\operatorname { c o n v }}\left(Q_{i}\right)$. So,
$$
(x,-\lambda)=\lambda(x / \lambda,-1)+(1-\lambda) 0 \in \operatorname{conv}\left(Q_{i}\right),
$$
so long as $\lambda>0$. On the other hand, if $\lambda=0$, then we know that $x / \lambda^{\prime} \in T_{i}$ for all $\lambda^{\prime}>0$, so
$$
\left(x, \lambda^{\prime}\right)=\lambda^{\prime}\left(x / \lambda^{\prime},-1\right)+\left(1-\lambda^{\prime}\right) 0 \in \operatorname{conv}\left(Q_{i}\right) .
$$

Sending $\lambda^{\prime} \rightarrow 0$ gives the result, since $Q_{i}$ is closed as it is the union of two closed sets. Putting it all together, this implies that $\boldsymbol{\operatorname { c o n v }}\left(Q_{i}\right) \supseteq \bar{K}_{i}$.

Forward inclusion. Now, we show the forward inclusion: that $\boldsymbol{\operatorname { c o n v }}\left(Q_{i}\right) \subseteq \bar{K}_{i}$. Note that $Q_{i} \subseteq \bar{K}_{i}$ since, by definition
$$
T_{i} \times\{-1\} \subseteq \bar{K}_{i}
$$
and $0 \in \bar{K}_{i}$, also essentially by definition. This immediately implies that
$$
\operatorname{conv}\left(Q_{i}\right) \subseteq \operatorname{conv}\left(\bar{K}_{i}\right)=\bar{K}_{i}
$$
where we have used the fact that the convex hull of a convex set is itself.

Discussion. Putting the above two points together, we get the claim that
$$
\begin{equation*}
\operatorname{conv}\left(Q_{i}\right)=\bar{K}_{i} \tag{7.4}
\end{equation*}
$$

In other words, the convex hull of the 'hard' set is exactly the cone $K_{i}$ with the additional constraint that the last entry must be restricted to lie between 0 and -1 . One interesting interpretation of this claim is that we may view the cone $K_{i}$ as the conic completion of the set $Q_{i}$. More generally, $\boldsymbol{\operatorname { c o n e }}\left(Q_{i}\right)$ is defined as the set containing all conic (i.e., nonnegative) combinations of the elements of $Q_{i}$. Since is it not hard to show that $\operatorname{cone}\left(Q_{i}\right)=\operatorname{cone}\left(\operatorname{conv}\left(Q_{i}\right)\right)$, we have
$$
\operatorname{cone}\left(Q_{i}\right)=\operatorname{cone}\left(\operatorname{conv}\left(Q_{i}\right)\right)=\operatorname{cone}\left(\bar{K}_{i}\right)=K_{i}
$$
where the second equality follows from (7.4), while the last simply follows from definitions.

\subsection*{7.3 Convex relaxation}

Using the fact derived in the previous section, a convex relaxation of the network problem with fees (7.1) is
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right)+q_{i} \lambda_{i} \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i}  \tag{7.5}\\
& \left(x_{i}, \lambda_{i}\right) \in K_{i},-1 \leq \lambda_{i} \leq 0, \quad i=1, \ldots, m
\end{array}
$$

Note that we have replaced the nonconvex constraint $\left(x_{i}, \lambda_{i}\right) \in Q_{i}$ with the convex constraint $\left(x_{i}, \lambda_{i}\right) \in \boldsymbol{\operatorname { c o n v }}\left(Q_{i}\right)$, or, equivalently, using the facts derived in the previous section, replaced it with the constraint that $\left(x_{i}, \lambda_{i}\right) \in K_{i}$ and $-1 \leq \lambda_{i} \leq 0$.

Conic formulation. This convex relaxation is also a special case of the conic flow problem (2.4) in a very natural way. First, note that the constraint that $\lambda \leq 0$ is redundant using the definition of $K_{i}$. We may then pull the remaining constraint on $\lambda_{i}$, that $\lambda_{i} \geq-1$ into an indicator function, $I\left(\lambda_{i} \geq-1\right)$ and place it in the objective. Note that this indicator function is nonincreasing, so its negation is nondecreasing, and we get the final problem
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right)+q_{i} \lambda_{i}-I\left(\lambda_{i} \geq-1\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i}  \tag{7.6}\\
& \left(x_{i}, \lambda_{i}\right) \in K_{i}, \quad i=1, \ldots, m
\end{array}
$$

Using the same trick used in §2.3.2 to rewrite the matrices $A_{i}$, we receive an instance of the conic flow problem (2.4) since the objective is nondecreasing and the $\left\{K_{i}\right\}$ are cones. Indeed, the formulation found here (7.6) is essentially identical to the formulation found in (2.11), except with the addition of the fixed costs $q \geq 0$.

Integrality gap. While it may be tempting to use an identical argument to the one in §2.3.2 to show that, indeed, this problem has no integrality gap (i.e., the relaxation provided here is always tight), we note that the addition of the fixed costs $q$ makes things considerably more tricky. In the previous argument in §2.3.2, we used the fact that, if $\lambda_{i}>-1$, then we could set $\lambda_{i}=-1$ and always remain feasible with no change in objective value. Using this naïve replacement in problem (7.6), the objective value would decrease, which breaks the corresponding argument. We will show next that we expect the solution to be tight in the special case that the $V_{i}=0$, which is common in practice (see, for example, the applications in §3).

\subsection*{7.4 Tightness of the relaxation}

The next natural question is, then, just how tight do we expect the relaxation to be? We will show that in the case that $V_{i}=0$, if $m$, the number of edges, is much larger than $n$, the number of nodes, then most of the $\lambda_{i}$ will be integral. More specifically, we will show that, given any solution to the relaxation, we can recover a solution such that $m-n+1$ indices $i$ satisfy $\left(x_{i}, \lambda_{i}\right) \in Q_{i}$. If $m \gg n$, i.e., the number of nodes is much smaller than the number of edges, as is usually the case in practice, then this would mean that we expect most of the solution to be integral.

Shapley-Folkman lemma. We state the Shapley-Folkman lemma here in its standard form. Let $S_{1}, \ldots, S_{m} \subseteq \mathbf{R}^{n+1}$ be any subsets (convex or nonconvex) of $\mathbf{R}^{n+1}$. Then, for any
$$
y=x_{1}+\cdots+x_{m}
$$
where $x_{i} \in \boldsymbol{\operatorname { c o n v }}\left(S_{i}\right)$ for $i=1, \ldots, m$, there exists $\tilde{x}_{i} \in \boldsymbol{\operatorname { c o n v }}\left(S_{i}\right)$ with $i=1, \ldots, m$, such that
$$
y=\tilde{x}_{1}+\cdots+\tilde{x}_{m}
$$
which satisfy $\tilde{x}_{i} \in S_{i}$ for at least $m-n-1$ indices $i$. In other words, given any vector $y$, which lies in the (Minkowski) sum of the convex hulls of the $S_{i}$, we can find $\tilde{x}_{i}$, which sum to $y$, such that at least $m-n-1$ lie in the original sets $S_{i}$, while the remainder lie in the convex hull, $\operatorname{conv}\left(S_{i}\right)$. Intuitively, this lemma states that the sum of convex sets becomes closer and closer to its convex hull as the number of sets gets large. See figure 7.2 for an example.

Given a solution to the convex relaxation (7.6), we will use this lemma to construct a solution that has the same objective value as the original, yet almost all penalties $\lambda_{i}$ will be integral: either -1 or 0 . This will then let us construct an approximate solution to (7.1) and bound the difference between the optimal objective value for (7.1) and the approximate solution's objective value.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-088.jpg?height=218&width=1227&top_left_y=240&top_left_x=449}
\captionsetup{labelformat=empty}
\caption{Figure 7.2: A visual representation of the Shapley-Folkman lemma for the $1 / 2$-norm ball. As we take the Minkowski sum of the set with itself, it becomes closer and closer to its convex hull.}
\end{figure}

Constructing an approximate solution. Assume we are given feasible flows and penalties for the relaxation; i.e., we have a solution $\left\{\left(x_{i}^{\star}, \lambda_{i}^{\star}\right)\right\}$ to the relaxation of the fixed-fee problem (7.5). From this solution to the relaxation (7.5), we will construct a feasible point for the original fixed-fee problem (7.1) which we will then show is 'close' to the optimal value, under certain conditions. We write the exact problem we are considering (the special case of (7.5) when $V_{i}=0$ ) for convenience:
$$
\begin{array}{ll}
\operatorname{maximize} & U(y)+q^{T} \lambda \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i} \\
& \left(x_{i}, \lambda_{i}\right) \in \bar{K}_{i}, \quad i=1, \ldots, m
\end{array}
$$

Here, we have used the definition of $\bar{K}_{i}$ from (7.3), and, from the previous discussion (7.4), we know that $\bar{K}_{i}=\boldsymbol{\operatorname { c o n v }}\left(Q_{i}\right)$.

First, note that, by definition
$$
y^{\star}=\sum_{i=1}^{m} A_{i} x_{i}^{\star},
$$
and that $\left(x_{i}^{\star}, \lambda_{i}^{\star}\right) \in \boldsymbol{\operatorname { c o n v }}\left(Q_{i}\right)$ for $i=1, \ldots, m$. Now, from the problem statement, we have
$$
c=q^{T} \lambda^{\star}
$$
with $q \geq 0$, where $c \leq 0$ stands for the 'fixed cost' part of the objective. Rewriting slightly,
$$
\left[\begin{array}{c}
y^{\star} \\
c
\end{array}\right]=\sum_{i=1}^{m}\left[\begin{array}{cc}
A_{i} & 0 \\
0 & q_{i}
\end{array}\right]\left[\begin{array}{c}
x_{i}^{\star} \\
\lambda_{i}^{\star}
\end{array}\right],
$$
where $\left(x_{i}^{\star}, \lambda_{i}^{\star}\right) \in \bar{K}_{i}$, or, equivalently, using (7.4), $\left(x_{i}^{\star}, \lambda_{i}^{\star}\right) \in \operatorname{conv}\left(Q_{i}\right)$, for $i=1, \ldots, m$. From the Shapley-Folkman lemma, there exist $\left(\tilde{x}_{i}^{\star}, \tilde{\lambda}_{i}^{\star}\right) \in \boldsymbol{\operatorname { c o n v }}\left(Q_{i}\right)$ such that
$$
\left[\begin{array}{c}
y^{\star} \\
c
\end{array}\right]=\sum_{i=1}^{m}\left[\begin{array}{cc}
A_{i} & 0 \\
0 & q_{i}
\end{array}\right]\left[\begin{array}{c}
\tilde{x}_{i}^{\star} \\
\tilde{\lambda}_{i}^{\star}
\end{array}\right],
$$
and at least $m-n$ indices $i$ satisfy $\left(\tilde{x}_{i}^{\star}, \tilde{\lambda}_{i}^{\star}\right) \in Q_{i}$. In other words, if $m \gg n$, we have an 'almost' feasible solution for the original problem (7.1), except for $n$ indices $i$. Consider the remaining indices, of which there are at most $n$. In this case, we know, from the dominated point condition (2.9) for $K_{i}$ (and therefore for $\bar{K}_{i}$ ) that if $\left(\tilde{x}_{i}^{\star}, \tilde{\lambda}_{i}^{\star}\right) \in \bar{K}_{i}$, then $\left(\tilde{x}_{i}^{\star},-1\right) \in \bar{K}_{i}$.

But we know that $\left(\tilde{x}_{i}^{\star},-1\right) \in Q_{i}$, making this point also feasible for the original problem (7.1). In English: if we were charged less than the full amount due to the relaxation (i.e., we were charged $q_{i} \lambda_{i}$ ), we can always choose to be charged the full amount for the same flow ( $-q_{i}$ ) and be feasible for the original problem. This means that a feasible solution for the original problem (7.1) will be to set
$$
\left(x_{i}^{0}, \lambda_{i}^{0}\right)= \begin{cases}\left(\tilde{x}_{i}^{\star}, \tilde{\lambda}_{i}^{\star}\right) & \left(\tilde{x}_{i}^{\star}, \tilde{\lambda}_{i}^{\star}\right) \in Q_{i}  \tag{7.7}\\ \left(\tilde{x}_{i}^{\star},-1\right) & \text { otherwise }\end{cases}
$$
for $i=1, \ldots, m$. Note that $\left(x_{i}^{0}, \lambda_{i}^{0}\right) \in Q_{i}$ for each $i$ and so is a feasible point for the original fixed-fee problem (7.1), leading to the same net flows $y^{\star}$ as the solution to the relaxation, but the cost incurred, $q^{T} \lambda^{0}$ differs by at most
$$
q^{T}\left(\lambda^{0}-\tilde{\lambda}^{\star}\right)
$$
from that of the relaxation, $q^{T} \tilde{\lambda}^{\star}=c$. Since most entries of $\lambda^{0}-\tilde{\lambda}^{\star}$ are zero, by the previous argument, then we expect this cost to be small. We give a simple bound on this, and therefore in the objective gap between the relaxation and the original problem, in what follows.

Bounding the optimal objective value. Let $p^{0}$ be the optimal objective value for the relaxation (7.5) and let $p^{\star}$ be the optimal objective value for the original problem (7.1). Then, since (7.5) is a relaxation of (7.1), we know that
$$
p^{\star} \leq p^{0}
$$

From the previous discussion, we have a feasible point (7.7) for the original problem. By construction, we know that the net flows $y^{\star}$ remain unchanged, so the net utility $U\left(y^{\star}\right)$ in the objective similarly remains unchanged. On the other hand, the cost incurred $q^{T} \lambda^{0}$ is larger from that of the relaxation, $c$, by $q^{T}\left(\tilde{\lambda}^{\star}-\lambda^{0}\right)$, so we have the following bound
$$
p^{0}+q^{T}\left(\lambda^{0}-\tilde{\lambda}^{\star}\right) \leq p^{\star} \leq p^{0}
$$

If we solved the relaxation, then we immediately have a two-sided bound on the optimal objective value as given above. On the other hand, we can give a simple bound that does not require solving the relaxation. Since we know that at most $n+1$ entries of $\lambda^{0}$ will differ from those of $\tilde{\lambda}^{\star}$ by Shapley-Folkman, then
$$
p^{0}-(n+1)\left(\max _{i} q_{i}\right) \leq p^{\star} \leq p^{0} .
$$

Or, equivalently
$$
0 \leq p^{0}-p^{\star} \leq(n+1)\left(\max _{i} q_{i}\right)
$$

Thus, when the fixed costs scale roughly inversely in the number of nodes, we have that the difference between the optima value of the relaxation and the optimal value of the original problem is roughly constant.

Discussion. While the previous discussion suggests that solutions to the relaxation are likely to be close to integral (and, if not, gives a procedure for constructing near-integral solutions given solutions to the convex relaxation), we find, for small and medium numerical examples in practice that the relaxation is exact. In fact, in the small number of experiments we performed, it was difficult to find a case in which the relaxation was not integral. We suspect that there is a natural set of conditions for which the relaxation can be shown to be integral that hold in practice, but leave this characterization for potential future work.

\subsection*{7.5 Fixed cost dual problem}

Finally, we derive the dual problem of the fixed-fee problem (7.1) and show that the algorithm developed in §5 can still be applied to this problem with minimal modifications. In fact, we lose little computational efficiently by solving the fixed-fee problem directly; of course, the solution is not guaranteed to be optimal.

Dual function. Using a similar derivation to that in §4.4, we can write the dual function as
$$
g(\nu, \eta)=\bar{U}(\nu)+\sum_{i=1}^{m} \bar{V}_{i}\left(\eta_{i}-A_{i}^{T} \nu\right)+\sum_{i=1}^{m} \sup _{\left(x_{i}, \lambda_{i}\right) \in Q_{i}}\left(\eta_{i}^{T} x_{i}+\lambda_{i} q_{i}\right) .
$$

Note that the support function over $Q_{i}$ can be evaluated easily: we simply compute the support function of $T_{i}$ less $q_{i}$ and compare this value to 0 . If we again define
$$
f_{i}\left(\eta_{i}\right)=\sup _{x_{i} \in T_{i}} \eta_{i}^{T} x_{i}
$$
then we can write
$$
\sup _{\left(x_{i}, \lambda_{i}\right) \in Q_{i}}\left(\eta_{i}^{T} x_{i}+\lambda_{i} q_{i}\right)=\max \left(f_{i}\left(\eta_{i}\right)-q_{i}, 0\right)
$$

This observation allows us to apply the algorithm from §5 'off the shelf' to solve the dual problem.

\section*{Chapter 8}

\section*{Conclusion}

In this thesis, we introduced the convex flow problem, which is a natural generalization of many important problems in computer science, operations research, and related fields. We showed that many problems from the literature are special cases of this framework, including max-flow, optimal power flow, routing through financial markets, and equilibrium computation in Fisher markets, among others. This generalization has a number of useful properties including, and perhaps most importantly, that its dual decomposes over the (hyper)graph structure. This decomposition allows for a fast algorithm that easily parallelizes over the edges of the graph and preserves the structure present in the original problem. We implemented this algorithm in the Julia package ConvexFlows.jl, which includes an easy-to-use interface, and we showed order-of-magnitude speedups over a commercial solver applied to the same problem. Finally, we extended the framework to include fixed costs on the edges. While this problem is nonconvex, we argued that a natural extension of this framework often works reasonably well in practice.

Future work. This work prompts a number of questions, any of which suggest an interesting avenue for future research. First, why do we often find integral solutions to the fixed fee problem? Is there a natural condition that guarantees this? Second, what algorithm is best for solving the fixed fee problem? Should we use the same algorithm as for the convex flow problem, with the modification suggested in §7.5? Finally, this problem has a very natural decomposition over the edges (equivalently, over the nodes, since we are working with a bipartite graph). Can we exploit this decomposition to devise efficient distributed, tatonnement-style algorithms? If so, what convergence rates should we expect? These potentially asynchronous algorithms may be of interest in decentralized applications such as power grids or wireless networks similar to those discussed in §3.

\section*{Appendix A}

\section*{Extended monotropic programming}

In this appendix, we explicitly draw the connection between the extended monotropic programming (EMP) problem formulated by Bertsekas [Ber08] and the convex flow problem (2.3). The extended monotropic programming problem can be written as
$$
\begin{array}{ll}
\operatorname{minimize} & \sum_{i=1}^{m+1} f_{i}\left(z_{i}\right) \\
\text { subject to } & z \in S
\end{array}
$$
with variable $z \in \mathbf{R}^{N}$. The functions $f_{i}$ are convex functions of the subvectors $z_{i}$, and the set $S$ is a subspace of $\mathbf{R}^{N}$. Taking $z=\left(y, x_{1}, \ldots, x_{m}\right)$ and changing the minimization to a maximization, we can write the convex flow problem as a monotropic programming problem:
$$
\begin{array}{ll}
\text { maximize } & U(y)+\sum_{i=1}^{m} V_{i}\left(x_{i}\right)-I_{T_{i}}\left(x_{i}\right) \\
\text { subject to } & y=\sum_{i=1}^{m} A_{i} x_{i}
\end{array}
$$
where we took
$$
f_{m+1}=-U, \quad \text { and } \quad f_{i}=-V_{i}+I_{T_{i}}, \quad i=1, \ldots, m
$$

Note that the linear net flow constraint is a subspace constraint.
Duality. The dual of the EMP problem considered by Bertsekas is given by
$$
\begin{array}{ll}
\text { maximize } & -\sum_{i=1}^{m+1} \sup _{z_{i} \in \mathbf{R}^{n_{i}}}\left\{\lambda_{i}^{T} z_{i}-f_{i}\left(z_{i}\right)\right\} \\
\text { subject to } & \lambda \in S^{\perp}
\end{array}
$$

Substituting in $U$ and $\left\{V_{i}\right\}$ and switching the sign of $\lambda$, the objective terms become
$$
\sup _{z_{m+1} \in \mathbf{R}^{n}}\left\{U\left(z_{m+1}\right)-\lambda_{m+1}^{T} z_{m+1}\right\}=\bar{U}\left(\lambda_{m+1}\right) \quad \text { and } \quad \sup _{z_{i} \in T_{i}}\left\{V_{i}\left(z_{i}\right)-\lambda_{i}^{T} z_{i}\right\}
$$

These terms are very close to, but not exactly the same as, the dual terms in the convex flow problem (4.6). In particular, the $U$ subproblem (4.3a) remains the same, but, in our framework, we introduced an additional dual variable to split the $V_{i}$ subproblem into two subproblems: one for the function $V_{i}$ (4.3b) and one for the set $T_{i}$ (4.3c). This split allows for a more efficient algorithm that uses the 'arbitrage' primitive (4.3c), which has a very fast implementation for many edges, especially in the case of two node edges (see §5.1). Our dual problem for the convex flow problem allows us to exploit more structure in our solver.

When the EMP problem matches. In the case of zero edge utilities, however, the EMP problem matches the convex network flow problem exactly. In this case, the $V$ subproblem disappears and we are left only with the arbitrage subproblem:
$$
\sup _{z_{i} \in T_{i}}\left\{V_{i}\left(z_{i}\right)-\lambda_{i}^{T} z_{i}\right\}=\sup _{z_{i} \in T_{i}}\left\{-\lambda_{i}^{T} z_{i}\right\}=f_{i}\left(-\lambda_{i}\right)
$$

Letting $\nu=-\lambda_{m}$, the subspace constraint then becomes
$$
\lambda_{i}=A_{i}^{T} \nu, \quad i=1, \ldots, m
$$

Thus, we recover the exact dual of the convex flow problem with zero edge utilities, given in (4.10). This immediately implies the strong duality result given in [Ber08, Prop 2.1] holds in our setting as well.

Self duality. The EMP dual problem has the same form as the primal; in this sense, the EMP problem is self-dual. The convex flow problem, however, does not appear to be selfdual in the same sense, since we consider a very specific subspace that defines the net flow constraint. We leave exploration of duality in our setting to future work.

\section*{Appendix B}

\section*{Additional details for the numerical experiments}

\section*{B. 1 Optimal power flow}

Arbitrage problem. Here, we explicitly work out the arbitrage subproblem for the optimal power flow problem. Recall that the set of allowable flows is given by (dropping the edge index for convenience)
$$
T=\left\{z \in \mathbf{R}^{2} \mid-b \leq z_{1} \leq 0 z_{2} \leq-z_{1}-\ell\left(-z_{1}\right)\right\}
$$
where
$$
\ell(w)=16(\log (1+\exp (w / 4))-\log 2)-2 w
$$

Given an edge input $w \in[0, b]$, the gain function is
$$
h(w)=w-\ell(w)
$$
where we assume the edge capacity $b$ is chosen such that the function $f$ is increasing for all $w \in[0, b]$, i.e., $f^{\prime}(b)>0$. Using (5.6), we can compute the optimal solution $x^{\star}$ to the arbitrage subproblem (4.3c) as
$$
x_{1}^{\star}=\left(4 \log \left(\frac{3 \eta_{2}-\eta_{1}}{\eta_{2}+\eta_{1}}\right)\right)_{[0, b]}, \quad x_{2}^{\star}=h\left(x_{1}^{\star}\right)
$$
where $(\cdot)_{[0, b]}$ denotes the projection onto the interval $[0, b]$.

Conic formulation. Define the exponential cone as
$$
K_{\exp }=\left\{(x, y, z) \in \mathbf{R}^{3} \mid y>0, y e^{x / y} \leq z \cdot\right\}
$$

The transmission line constraint is of the form
$$
\log \left(1+e^{s}\right) \leq t
$$
which can be written as [ApS24b, §5.2.5]
$$
\begin{aligned}
u+v & \leq 1 \\
(x-t, 1, u) & \in K_{\exp } \\
(-t, 1, v) & \in K_{\exp } .
\end{aligned}
$$

Define the rotated second order cone as
$$
K_{\mathrm{rot} 2}=\left\{(t, u, x) \in \mathbf{R}_{+} \times \mathbf{R}_{+} \times \mathbf{R}^{n} \mid 2 t u \geq\|x\|_{2}^{2}\right\} .
$$

We can write the cost function
$$
c_{i}(w)=(1 / 2) w_{+}^{2},
$$
where $w_{+}=\max (w, 0)$ denotes the negative part of $w$, in conic form as minimizing $t_{1} \in \mathbf{R}$ subject to the second-order cone constraint [ApS24b, §3.2.2]
$$
\left(0.5, t_{1}, t_{2}\right) \in K_{\mathrm{rot} 2}, \quad t_{2} \geq w, \quad t_{2} \geq 0
$$

Putting this together, the conic form problem is
$$
\begin{array}{ll}
\text { maximize } & -\mathbf{1}^{T} t_{1} \\
\text { subject to } & \left(0.5,\left(t_{1}\right)_{i},\left(t_{2}\right)_{i}\right) \in K_{\mathrm{rot} 2}, \text { for } i=1, \ldots n \\
& t_{2} \geq d-y, \quad t_{2} \geq 0 \\
& -b_{i} \leq\left(x_{i}\right)_{1} \leq 0, \text { for } i=1, \ldots m \\
& u_{i}+v_{i} \leq 1 \text { for } i=1, \ldots m \\
& \left(-\beta_{i}\left(x_{i}\right)_{1}+\left(3\left(x_{i}\right)_{1}+\left(x_{i}\right)_{2}\right) / \alpha-\log (2), 1, u_{i}\right) \in K_{\exp } \quad \text { for } i=1, \ldots m \\
& \left(\left(3\left(x_{i}\right)_{1}+\left(x_{i}\right)_{2}\right) / \alpha-\log (2), 1, v_{i}\right) \in K_{\exp } \quad \text { for } i=1, \ldots m
\end{array}
$$

\section*{B. 2 Routing orders through financial exchanges}

In this example, we considered three different types of decentralized exchange markets: Uniswap-like, Balancer-like swap markets, and Balancer-like multi-asset markets. Recall that a constant function market maker (CFMM) allows trades between the $n$ tokens in its reserves $R \in \mathbf{R}_{+}^{n}$ with behavior governed by a trading function $\varphi: \mathbf{R}_{+}^{n} \rightarrow \mathbf{R}$. The CFMM only accepts a trade ( $\Delta, \Lambda$ ) where $\Delta \in \mathbf{R}_{+}^{n}$ is the basket of tendered tokens and $\Lambda \in \mathbf{R}_{+}^{n}$ is the basket of received tokens if
$$
\varphi(R+\gamma \Delta-\Lambda) \geq \varphi(R)
$$

The Uniswap trading function $\varphi_{\text {Uni }}: \mathbf{R}_{+}^{2} \rightarrow \mathbf{R}$ is given by
$$
\varphi_{\mathrm{Uni}}(R)=\sqrt{R_{1} R_{2}} .
$$

The Balancer swap market trading function $\varphi_{\text {Bal }}: \mathbf{R}_{+}^{2} \rightarrow \mathbf{R}$ is given by
$$
\varphi_{\mathrm{Bal}}(R)=R_{1}^{4 / 5} R_{2}^{1 / 5}
$$

The Balancer multi-asset trading function $\varphi_{\text {Mul }}: \mathbf{R}_{+}^{3} \rightarrow \mathbf{R}$ is given by
$$
\varphi_{\mathrm{Mul}}(R)=R_{1}^{1 / 3} R_{2}^{1 / 3} R_{3}^{1 / 3}
$$

These functions are easily recognized as (weighted) geometric means and can be verified as concave, nondecreasing function. Thus, the set of allowable trades
$$
T=\left\{\Lambda-\Delta \mid \Lambda, \Delta \in \mathbf{R}_{+}^{n} \text { and } \varphi(R+\gamma \Delta-\Lambda) \geq \varphi(R)\right\}
$$
is convex. Furthermore, the arbitrage problem (4.3c) has a closed form solution for the case of the swap markets (see $[\mathrm{Ang}+20, \mathrm{App} . \mathrm{A}]$ and the implementation from $[\mathrm{Dia}+23]$ ). Multiasset pools may have closed form solutions as well, which we discuss in the next section.

Separable CFMM arbitrage problem. Consider a separable CFMM with trading function $\varphi: \mathbf{R}_{+}^{n} \rightarrow \mathbf{R}$ of the form
$$
\varphi(R)=\sum_{i=1}^{n} \varphi_{i}\left(R_{i}\right)
$$
where each $\varphi_{i}$ is strictly concave and increasing. (The non-strict case follows from the same argument but requires more care.) Note that many CFMMs may be transformed into this form. For example, a weighted geometric mean CFMM like Balancer can be written in this form using a log transform:
$$
\prod_{i=1}^{n} R_{i}^{w_{i}} \geq k \Longleftrightarrow \sum_{i=1}^{n} w_{i} \log R_{i} \geq \log k
$$

The arbitrage subproblem can be written as
$$
\begin{array}{ll}
\operatorname{maximize} & \eta^{T}(\Lambda-\Delta) \\
\text { subject to } & \sum_{i=1}^{n} \varphi_{i}\left(R_{i}+\gamma \Delta_{i}-\Lambda_{i}\right) \geq k \\
& \Delta, \Lambda \geq 0
\end{array}
$$

After pulling the nonnegativity constraints into the objective, the Lagrangian is separable and can be written as
$$
L(\Delta, \Lambda, \lambda)=\sum_{i=1}^{n} \eta_{i}\left(\Lambda_{i}-\Delta_{i}\right)-I\left(\Delta_{i}\right)-I\left(\Lambda_{i}\right)+\lambda\left(\varphi_{i}\left(R_{i}+\gamma \Delta_{i}-\Lambda_{i}\right)-k\right),
$$
where $\lambda \geq 0$ and $I$ is the nonnegative indicator function satisfying $I(w)=0$ if $w \geq 0$ and $+\infty$ otherwise. Maximizing over the primal variables $\Delta$ and $\Lambda$ gives the dual function:
$$
\begin{equation*}
g(\lambda)=\sum_{i=1}^{n}\left(\sup _{\Delta_{i}, \Lambda_{i} \geq 0} \eta_{i}\left(\Lambda_{i}-\Delta_{i}\right)+\lambda_{i} \varphi_{i}\left(R_{i}+\gamma \Delta_{i}-\Lambda_{i}\right)\right)-\lambda k . \tag{B.1}
\end{equation*}
$$

Consider subproblem $i$ inside of the sum. If $0 \leq \gamma<1$, then at most one of $\Delta_{i}^{\star}$ or $\Lambda_{i}^{\star}$ is nonzero, which turns this two variable problem into two single variable convex optimization
problems, each with a nonnegativity constraint. (This follows from $[\mathrm{Ang}+22 \mathrm{~b}, 82.2]$.) In particular, to solve the original, we can solve two (smaller) problems by considering the two possible cases. In the first case, we have $\Lambda_{i}=0$ and $\Delta_{i} \geq 0$, giving the problem
$$
\begin{array}{ll}
\operatorname{maximize} & -\eta_{i} \Delta_{i}+\lambda_{i} \varphi_{i}\left(R_{i}+\gamma \Delta_{i}\right) \\
\text { subject to } & \Delta_{i} \geq 0
\end{array}
$$
and the second case has $\Delta_{i}=0$ and $\Lambda_{i} \geq 0$, which means that we only have to solve
$$
\begin{array}{ll}
\operatorname{maximize} & \eta_{i} \Lambda_{i}+\lambda_{i} \varphi_{i}\left(R_{i}-\Lambda_{i}\right) \\
\text { subject to } & \Lambda_{i} \geq 0
\end{array}
$$

It would then suffice to take whichever of the two cases has the highest optimal objective value - though, unless $\gamma=1$, at most one problem will have a positive solution and we deal with the $\gamma=1$ case below. These problems can be solved by ternary search (if we only have access to $\varphi_{i}$ via function evaluations), bisection (if we also have access to the derivative, $\varphi_{i}^{\prime}$ ), or Newton's method (if we have access to the second derivative, $\varphi_{i}^{\prime \prime}$ ).

These problems also often have closed form solutions. For example, the optimality conditions for the first of the two cases is: if $\Delta_{i}^{\star}=0$ is optimal, then
$$
\lambda_{i} \gamma \varphi_{i}^{\prime}\left(R_{i}\right) \leq \eta_{i}
$$
or, otherwise, $\Delta_{i}^{\star}>0$ satisfies
$$
\lambda_{i} \gamma \varphi_{i}^{\prime}\left(R_{i}+\gamma \Delta_{i}^{\star}\right)=\eta_{i} .
$$

The former condition is a simple check and the latter condition is a simple root-finding problem that, in many cases, has a closed-form solution. A very similar also holds for the second case.

Finally, if $\gamma=1$, the subproblems in the dual function (B.1) simplify even further to the unconstrained single variable convex optimization problem
$$
\sup _{t}\left(\eta_{i} t+\lambda_{i} \varphi_{i}\left(R_{i}-t\right)\right)
$$
which is easily solved via any number of methods. We can recover a solution to the original subproblem by setting $\Lambda_{i}^{\star}=t^{\star}+\Delta_{i}^{\star}$ for any solution $t^{\star}$, where $\Delta_{i}^{\star}$ is any value.

CFMMs as conic constraints. Define the power cone as
$$
K_{\mathrm{pow}}(w)=\left\{(x, y, z) \in \mathbf{R}^{3}\left|x^{w} y^{1-w} \geq|z|, x \geq 0, y \geq 0\right\}\right.
$$

We model the two-asset market constraints as
$$
\begin{equation*}
(R+\gamma \Delta-\Lambda, \varphi(R)) \in K_{\mathrm{pow}}(w), \quad \text { and } \quad \Delta, \Lambda \geq 0 \tag{B.2}
\end{equation*}
$$
where $w=0.5$ for Uniswap and $w=0.8$ for Balancer. Define the geometric mean cone as
$$
K_{\text {geomean }}=\left\{(t, x) \in \mathbf{R} \times \mathbf{R}^{n} \mid x \geq 0,\left(x_{1} x_{2} \cdots x_{n}\right)^{1 / n} \geq t\right\}
$$

We model the multi-asset market constraint as
$$
\begin{equation*}
(-3 \varphi(R), R+\gamma \Delta-\Lambda) \in K_{\text {geomean }}, \quad \text { and } \quad \Delta, \Lambda \geq 0 . \tag{B.3}
\end{equation*}
$$

Objectives as conic constraints. Define the rotated second order cone as
$$
K_{\mathrm{rot} 2}=\left\{(t, u, x) \in \mathbf{R}_{+} \times \mathbf{R}_{+} \times \mathbf{R}^{n} \mid 2 t u \geq\|x\|_{2}^{2}\right\}
$$

The net flow utility function is
$$
U(y)=c^{T} y-(1 / 2) \sum_{i=1}^{n}\left(y_{i}\right)_{-}^{2},
$$
where $x_{-}=\max (-x, 0)$ denotes the negative part of $x$. In conic form, maximizing $U$ is equivalent to maximizing
$$
c^{T} y-(1 / 2) \sum_{i=1}^{n}\left(p_{1}\right)_{i}
$$
subject to the constraints
$$
\left.p_{2} \geq 0, \quad p_{2} \geq-y, \quad\left(p_{1}\right)_{i},\left(p_{2}\right)_{i}\right) \in K_{\text {rot } 2} \quad \text { for } i=1, \ldots, n
$$
where we introduced new variables $p_{1}, p_{2} \in \mathbf{R}^{n}$ [ApS24b, §3.2.2]. The $V_{i}$ 's can be modeled similarly using the rotated second order cone.

Conic form problem. The CFMM arbitrage example can then be written in conic form as
$$
\begin{array}{ll}
\text { maximize } & c^{T} y-(1 / 2) \sum_{i=1}^{n}\left(p_{1}\right)_{i}-(1 / 2) \sum_{i=1}^{m}\left(t_{1}\right)_{i} \\
\text { subject to } & \left(0.5,\left(p_{1}\right)_{i},\left(p_{2}\right)_{i}\right) \in K_{\text {rot } 2}, \quad i=1, \ldots, n \\
& p_{1} \geq 0 \\
& p_{2} \geq 0, \quad p_{2} \geq-y \\
& \left(0.5,\left(t_{1}\right)_{i},\left(t_{2}\right)_{i}\right) \in K_{\text {rot } 2}, \quad i=1, \ldots, n \\
& t_{1} \geq 0 \\
& t_{2} \geq 0, \quad\left(t_{2}\right)_{i} \geq-\left(\Lambda_{i}-\Delta_{i}\right) \\
& (R+\gamma \Delta-\Lambda, \varphi(R)) \in K_{\text {pow }}\left(w_{i}\right), \quad i=1, \ldots, m_{1} \\
& (-3 \varphi(R), R+\gamma \Delta-\Lambda) \in K_{\text {geomean }}, \quad i=m_{1}+1, \ldots, m \\
& \Delta_{i}, \Lambda_{i} \geq 0, \quad i=1, \ldots, m,
\end{array}
$$
with variables $y \in \mathbf{R}^{n}, p_{1} \in \mathbf{R}^{n}, p_{2} \in \mathbf{R}^{n}, t_{1} \in \mathbf{R}^{m},\left(t_{2}\right)_{i} \in \mathbf{R}^{n_{i}}, \Delta \in \mathbf{R}^{n_{i}}$, and $\Lambda \in \mathbf{R}^{n_{i}}$ for $i=1, \ldots, m$.

\section*{Appendix C}

\section*{An Efficient Algorithm for Optimal Routing Through Constant Function Market Makers}

\section*{C. 1 Introduction}

Decentralized Finance, or DeFi, has been one of the largest growth areas within both financial technologies and cryptocurrencies since 2019. DeFi is made up of a network of decentralized protocols that match buyers and sellers of digital goods in a trustless manner. Within DeFi, some of the most popular applications are decentralized exchanges (DEXs, for short) which allow users to permissionlessly trade assets. While there are many types of DEXs, the most popular form of exchange (by nearly any metric) is a mechanism known as the constant function market maker, or CFMM. A CFMM is a particular type of DEX which allows anyone to propose a trade (e.g., trading some amount of one asset for another). The trade is accepted if a simple rule, which we describe later in §C.2.1, is met.

The prevalence of CFMMs on blockchains naturally leads to questions about routing trades across networks or aggregations of CFMMs. For instance, suppose that one wants to trade some amount of asset A for the greatest possible amount of asset B. There could be many 'routes' that provide this trade. For example, we may trade asset A for asset C , and only then trade asset C for asset B . This routing problem can be formulated as an optimization problem over the set of CFMMs available to the user for trading. Angeris et al. $[\operatorname{Ang}+22 \mathrm{a}]$ showed that the general problem of routing is a convex program for concave utilities, ignoring blockchain transactions costs, though special cases of the routing problem have been studied previously [Wan+22; DKP21a].

This paper. In this paper, we apply a decomposition method to the optimal routing problem, which results in an algorithm that easily parallelizes across all DEXs. To solve the subproblems of the algorithm, we formalize the notions of swap markets, bounded liquidity, and aggregate CFMMs (such as Uniswap v3) and discuss their properties. Finally, we demonstrate that our algorithm for optimal routing is efficient, practical, and can handle the large variety of CFMMs that exist on chain today.

\section*{C. 2 Optimal routing}

In this section, we define the general problem of optimal routing and give concrete examples along with some basic properties.

Assets. In the optimal routing problem, we have a global labeling of $n$ assets which we are allowed to trade, indexed by $j=1, \ldots, n$ throughout this paper. We will sometimes refer to this 'global collection' as the universe of assets that we can trade.

Trading sets. Additionally, in this problem, we have a number of markets $i=1, \ldots, m$ (usually constant function market makers, or collections thereof, which we discuss in §C.2.1) which trade a subset of the universe of tokens of size $n_{i}$. We define market $i$ 's behavior, at the time of the trade, via its trading set $T_{i} \subseteq \mathbf{R}^{n_{i}}$. This trading set behaves in the following way: any trader is able to propose a trade consisting of a basket of assets $\Delta_{i} \in \mathbf{R}^{n_{i}}$, where positive entries of $\Delta_{i}$ denote that the trader receives those tokens from the market, while negative values denote that the trader tenders those tokens to the market. (Note that the baskets here are of a subset of the universe of tokens which the market trades.) The market then accepts this trade (i.e., takes the negative elements in $\Delta_{i}$ from the trader and gives the positive elements in $\Delta_{i}$ to the trader) whenever
$$
\Delta_{i} \in T_{i} .
$$

We make two assumptions about the sets $T_{i}$. One, that the set $T_{i}$ is a closed convex set, and, two, that the zero trade is always an acceptable trade, i.e., $0 \in T_{i}$. All existing DEXs that are known to the authors have a trading set that satisfies these conditions.

Local and global indexing. Each market $i$ trades only a subset of $n_{i}$ tokens from the universe of tokens, so we introduce the matrices $A_{i} \in \mathbf{R}^{n \times n_{i}}$ to connect the local indices to the global indices. These matrices are defined such that $A_{i} \Delta_{i}$ yields the total amount of assets the trader tendered or received from market $i$, in the global indices. For example, if our universe has 3 tokens and market $i$ trades the tokens 2 and 3 , then
$$
A_{i}=\left[\begin{array}{ll}
0 & 0 \\
1 & 0 \\
0 & 1
\end{array}\right]
$$

Written another way, $\left(A_{i}\right)_{j k}=1$ if token $k$ in the market's local index corresponds to global token index $j$, and $\left(A_{i}\right)_{j k}=0$ otherwise. We note that the ordering of tokens in the local index does not need to be the same as the global ordering.

Network trade vector. By summing the net trade in each market, after mapping the local indices to the global indices, we obtain the network trade vector
$$
\Psi=\sum_{i=1}^{m} A_{i} \Delta_{i}
$$

We can interpret $\Psi$ as the net trade across the network of all markets. If $\Psi_{i}>0$, we receive some amount of asset $i$ after executing all trades $\left\{\Delta_{i}\right\}_{i=1}^{m}$. On the other hand, if $\Psi_{i}<0$, we tender some of asset $i$ to the network. Note that having $\Psi_{i}=0$ does not imply we do not trade asset $i$; it only means that, after executing all trades, we received as much as we tendered.

Network trade utility. Now that we have defined the network trade vector, we introduce a utility function $U: \mathbf{R}^{n} \rightarrow \mathbf{R} \cup\{-\infty\}$ that gives the trader's utility of a net trade $\Psi$. We assume that $U$ is concave and increasing (i.e., we assume all assets have value with potentially diminishing returns). Furthermore, we will use infinite values of $U$ to encode constraints; a trade $\Psi$ such that $U(\Psi)=-\infty$ is unacceptable to the trader. We can choose $U$ to encode several important actions in markets, including liquidating or purchasing a basket of assets and finding arbitrage. See [Ang+22b, §5.2] for several examples.

Optimal routing problem. The optimal routing problem is then the problem of finding a set of valid trades that maximizes the trader's utility:
$$
\begin{array}{ll}
\operatorname{maximize} & U(\Psi) \\
\text { subject to } & \Psi=\sum_{i=1}^{m} A_{i} \Delta_{i}  \tag{C.1}\\
& \Delta_{i} \in T_{i}, \quad i=1, \ldots, m
\end{array}
$$

The problem variables are the network trade vector $\Psi \in \mathbf{R}^{n}$ and trades with each market $\Delta_{i} \in \mathbf{R}^{n_{i}}$, while problem data are the utility function $U: \mathbf{R}^{n} \rightarrow \mathbf{R} \cup\{\infty\}$, the matrices $A_{i} \in \mathbf{R}^{n \times n_{i}}$, and the trading sets $T_{i} \subseteq \mathbf{R}^{n_{i}}$, where $i=1, \ldots, m$. Since the trading sets are convex and the utility function is concave, this problem is a convex optimization problem. In the subsequent sections, we will use basic results of convex optimization to construct an efficient algorithm to solve problem (C.1).

\section*{C.2.1 Constant function market makers}

Most decentralized exchanges, such as Uniswap v2, Balancer, Curve, among others, are currently organized as constant function market makers (CFMMs, for short) or collections of CFMMs (such as Uniswap v3) [AC20a; Ang+22b]. A constant function market maker is a type of permissionless market that allows anyone to trade baskets of, say, $r$, assets for other baskets of these same $s$ assets, subject to a simple set of rules which we describe below.

Reserves and trading functions. A constant function market maker, which allows $r$ tokens to be traded, is defined by two properties: its reserves $R \in \mathbf{R}_{+}^{r}$, where $R_{j}$ denotes the amount of asset $j$ available to the CFMM, and a trading function which is a concave function $\varphi: \mathbf{R}_{+}^{r} \rightarrow \mathbf{R}$, which specifies the CFMM's behavior and its trading fee $0<\gamma \leq 1$.

Acceptance condition. Any user is allowed to submit a trade to a CFMM, which is, from before, a vector $\Delta \in \mathbf{R}^{r}$. The submitted trade is then accepted if the following condition holds:
$$
\begin{equation*}
\varphi\left(R-\gamma \Delta_{-}-\Delta_{+}\right) \geq \varphi(R) \tag{C.2}
\end{equation*}
$$
and $R-\gamma \Delta_{-}-\Delta_{+} \geq 0$. Here, we denote $\Delta_{+}$to be the 'elementwise positive part' of $\Delta$, i.e., $\left(\Delta_{+}\right)_{j}=\max \left\{\Delta_{j}, 0\right\}$ and $\Delta_{-}$to be the 'elementwise negative part' of $\Delta$, i.e., $\left(\Delta_{-}\right)_{j}=\min \left\{\Delta_{j}, 0\right\}$ for every asset $j=1, \ldots, r$. The basket of assets $\Delta_{+}$may sometimes be called the 'received basket' and $\Delta_{-}$may sometimes be called the 'tendered basket' (see, e.g., $[\mathrm{Ang}+22 \mathrm{~b}]$ ). Note that the trading set $T$, for a CFMM, is exactly the set of $\Delta$ such that (C.2) holds,
$$
\begin{equation*}
T=\left\{\Delta \in \mathbf{R}^{r} \mid \varphi\left(R-\gamma \Delta_{-}-\Delta_{+}\right) \geq \varphi(R)\right\} \tag{C.3}
\end{equation*}
$$

It is clear that $0 \in T$, and it is not difficult to show that $T$ is convex whenever $\varphi$ is concave, which is true for all trading functions used in practice. If the trade is accepted then the CFMM pays out $\Delta_{+}$from its reserves and receives $-\Delta_{-}$from the trader, which means the reserves are updated in the following way:
$$
R \leftarrow R-\Delta_{-}-\Delta_{+} .
$$

The acceptance condition (C.2) can then be interpreted as: the CFMM accepts a trade only when its trading function, evaluated on the 'post-trade' reserves with the tendered basket discounted by $\gamma$, is at least as large as its value when evaluated on the current reserves.

It can be additionally shown that the trade acceptance conditions in terms of the trading function $\varphi$ and in terms of the trading set $T$ are equivalent in the sense that every trading set has a function $\varphi$ which generates it [AC20a], under some basic conditions.

Examples. Almost all examples of decentralized exchanges currently in production are constant function market makers. For example, the most popular trading function (as measured by most metrics) is the product trading function:
$$
\varphi(R)=\sqrt{R_{1} R_{2}}
$$
originally proposed for Uniswap [ZCP18] and a 'bounded liquidity' variation of this function:
$$
\begin{equation*}
\varphi(R)=\sqrt{\left(R_{1}+\alpha\right)\left(R_{2}+\beta\right)} \tag{C.4}
\end{equation*}
$$
used in Uniswap v3 [Ada + 21a], with $\alpha, \beta \geq 0$. Other examples include the weighted geometric mean (as used by Balancer [MM19a])
$$
\begin{equation*}
\varphi(R)=\prod_{i=1}^{r} R_{i}^{w_{i}} \tag{C.5}
\end{equation*}
$$
where $r$ is the number of assets the exchange trades, and $w \in \mathbf{R}_{+}^{r}$ with $\mathbf{1}^{T} w=1$ are known as the weights, along with the Curve trading function
$$
\varphi(R)=\alpha \mathbf{1}^{T} R-\left(\prod_{i=1}^{r} R_{i}^{-1}\right)
$$
where $\alpha>0$ is a parameter set by the CFMM [Ego]. Note that the 'product' trading function is the special case of the weighted geometric mean function when $r=2$ and $w_{1}=w_{2}=1 / 2$.

Aggregate CFMMs. In some special cases, such as in Uniswap v3, it is reasonable to consider an aggregate CFMM, which we define as a collection of CFMMs, which all trade the same assets, as part of a single 'big' trading set. A specific instance of an aggregate CFMM currently used in practice is in Uniswap v3 [Ada+21a]. Any 'pool' in this exchange is actually a collection of CFMMs with the 'bounded liquidity' variation of the product trading function, shown in (C.4). We will see that we can treat these 'aggregate CFMMs' in a special way in order to significantly improve performance.

\section*{C. 3 An efficient algorithm}

A common way of solving problems such as problem (C.1), where we have a set of variables coupled by only a single constraint, is to use a decomposition method [DW60; Ber16]. The general idea of these methods is to solve the original problem by splitting it into a sequence of easy subproblems that can be solved independently. In this section, we will see that applying a decomposition method to the optimal routing problem gives a solution method which parallelizes over all markets. Furthermore, it gives a clean programmatic interface; we only need to be able to find arbitrage for a market, given a set of reference prices. This interface allows us to more easily include a number of important decentralized exchanges, such as Uniswap v3.

\section*{C.3.1 Dual decomposition}

To apply the dual decomposition method, we first take the coupling constraint of problem (C.1),
$$
\Psi=\sum_{i=1}^{m} A_{i} \Delta_{i}
$$
and relax it to a linear penalty in the objective, parametrized by some vector $\nu \in \mathbf{R}^{n}$. (We will show in §C.3.2 that the only reasonable choice of $\nu$ is a market clearing price, sometimes called a no-arbitrage price, and that this choice actually results in a relaxation that is tight; i.e., a solution for this relaxation also satisfies the original coupling constraint.) This relaxation results in the following problem:
$$
\begin{array}{ll}
\operatorname{maximize} & U(\Psi)-\nu^{T}\left(\Psi-\sum_{i=1}^{m} A_{i} \Delta_{i}\right) \\
\text { subject to } & \Delta_{i} \in T_{i}, \quad i=1, \ldots, m
\end{array}
$$
where the variables are the network trade vector $\Psi \in \mathbf{R}^{n}$ and the trades are $\Delta_{i} \in \mathbf{R}^{n_{i}}$ for each market $i=1, \ldots, m$. Note that this formulation can be viewed as a family of problems parametrized by the vector $\nu$.

A simple observation is that this new problem is actually separable over all of its variables. We can see this by rearranging the objective:
$$
\begin{array}{ll}
\operatorname{maximize} & U(\Psi)-\nu^{T} \Psi+\sum_{i=1}^{m}\left(A_{i}^{T} \nu\right)^{T} \Delta_{i}  \tag{C.6}\\
\text { subject to } & \Delta_{i} \in T_{i}, \quad i=1, \ldots, m
\end{array}
$$

Since there are no additional coupling constraints, we can solve for $\Psi$ and each of the $\Delta_{i}$ with $i=1, \ldots, m$ separately.

Subproblems. This method gives two types of subproblems, each depending on $\nu$. The first, over $\Psi$, is relatively simple:
$$
\begin{equation*}
\operatorname{maximize} \quad U(\Psi)-\nu^{T} \Psi \tag{C.7}
\end{equation*}
$$
and can be recognized as a slightly transformed version of the Fenchel conjugate [BV04, §3.3]. We will write its optimal value (which depends on $\nu$ ) as
$$
\bar{U}(\nu)=\sup _{\Psi}\left(U(\Psi)-\nu^{T} \Psi\right)
$$

The function $\bar{U}$ can be easily derived in closed form for a number of functions $U$. Additionally, since $\bar{U}$ is a supremum over an affine family of functions parametrized by $\nu$, it is a convex function of $\nu$ [BV04, §3.2.3]. (We will use this fact soon.) Another important thing to note is that unless $\nu \geq 0$, the function $\bar{U}(\nu)$ will evaluate to $+\infty$. This can be interpreted as an implicit constraint on $\nu$.

The second type of problem is over each trade $\Delta_{i}$ for $i=1, \ldots, m$, and can be written, for each market $i$, as
$$
\begin{array}{ll}
\text { maximize } & \left(A_{i}^{T} \nu\right)^{T} \Delta_{i}  \tag{C.8}\\
\text { subject to } & \Delta_{i} \in T_{i}
\end{array}
$$

We will write its optimal value, which depends on $A_{i}^{T} \nu$, as $\operatorname{arb}_{i}\left(A_{i}^{T} \nu\right)$. Problem (C.8) can be recognized as the optimal arbitrage problem (see, e.g., $[\mathrm{Ang}+22 \mathrm{~b}]$ ) for market $i$, when the external market price, or reference market price, is equal to $A_{i}^{T} \nu$. Since $\operatorname{arb}_{i}\left(A_{i}^{T} \nu\right)$ is also defined as a supremum over a family of affine functions of $\nu$, it too is a convex function of $\nu$. Solutions to the optimal arbitrage problem are known, in closed form, for a number of trading functions. (See section C. 5 for some examples.)

Dual variables as prices. The optimal solution to problem (C.8), given by $\Delta_{i}^{\star}$, is a point $\Delta_{i}^{\star}$ in $T_{i}$ such that there exists a supporting hyperplane to the set $T_{i}$ at $\Delta_{i}^{\star}$ with slope $A_{i}^{T} \nu$ [BV04, §5.6]. We can interpret these slopes as the 'marginal prices' of the $n_{i}$ assets, since, letting $\delta \in \mathbf{R}^{n_{i}}$ be a small deviation from the trade $\Delta_{i}^{\star}$, we have, writing $\tilde{\nu}=A_{i}^{T} \nu$ as the weights of $\nu$ in the local indexing:
$$
\tilde{\nu}^{T}\left(\Delta_{i}^{\star}+\delta\right) \leq \tilde{\nu}^{T} \Delta_{i}^{\star}
$$
for every $\delta$ with $\Delta_{i}^{\star}+\delta \in T_{i}$. (By definition of optimality.) Canceling terms, we find:
$$
\tilde{\nu}^{T} \delta \leq 0
$$

If, for example, $\delta_{i}$ and $\delta_{j}$ are the only two nonzero entries of $\delta$, we would have
$$
\delta_{i} \leq-\frac{\tilde{\nu}_{j}}{\tilde{\nu}_{i}} \delta_{j}
$$
so the exchange rate between $i$ and $j$ is at most $\tilde{\nu}_{i} / \tilde{\nu}_{j}$. This observation lets us interpret the dual variables $\tilde{\nu}$ (and therefore the dual variables $\nu$ ) as 'marginal prices', up to a constant multiple.

\section*{C.3.2 The dual problem}

The objective value of problem (C.6), which is a function of $\nu$, can then be written as
$$
\begin{equation*}
g(\nu)=\bar{U}(\nu)+\sum_{i=1}^{m} \operatorname{arb}_{i}\left(A_{i}^{T} \nu\right) \tag{C.9}
\end{equation*}
$$

This function $g: \mathbf{R}^{n} \rightarrow \mathbf{R}$ is called the dual function. Since $g$ is the sum of convex functions, it too is convex. The dual problem is the problem of minimizing the dual function,
$$
\begin{equation*}
\text { minimize } \quad g(\nu), \tag{C.10}
\end{equation*}
$$
over the dual variable $\nu \in \mathbf{R}^{n}$, which is a convex optimization problem since $g$ is a convex function.

Dual optimality. While we have defined the dual problem, we have not discussed how it relates to the original routing problem we are attempting to solve, problem (C.1). Let $\nu^{\star}$ be a solution to the dual problem (C.10). Assuming that the dual function is differentiable at $\nu^{\star}$, the first order, unconstrained optimality conditions for problem (C.10) are that
$$
\nabla g\left(\nu^{\star}\right)=0 .
$$
(The function $g$ need not be differentiable, in which case a similar, but more careful, argument holds using subgradient calculus.) It is not hard to show that if $\bar{U}$ is differentiable at $\nu^{\star}$, then its gradient must be $\nabla \bar{U}\left(\nu^{\star}\right)=-\Psi^{\star}$, where $\Psi^{\star}$ is the solution to the first subproblem (C.7), with $\nu^{\star}$. (This follows from the fact that the gradient of a maximum, when differentiable, is the gradient of the argmax.) Similarly, the gradient of $\mathbf{a r b}_{i}$ when evaluated at $A_{i}^{T} \nu^{\star}$ is $\Delta_{i}^{\star}$, where $\Delta_{i}^{\star}$ is a solution to problem (C.8) with marginal prices $A_{i}^{T} \nu^{\star}$, for each market $i=1, \ldots, m$. Using the chain rule, we then have:
$$
\begin{equation*}
0=\nabla g\left(\nu^{\star}\right)=-\Psi^{\star}+\sum_{i=1}^{m} A_{i} \Delta_{i}^{\star} \tag{C.11}
\end{equation*}
$$

Note that this is exactly the coupling constraint of problem (C.1). In other words, when the linear penalties $\nu^{\star}$ are chosen optimally (i.e., chosen such that they minimize the dual problem (C.10)) then the optimal solutions for subproblems (C.7) and (C.8) automatically satisfy the coupling constraint. Because problem (C.6) is a relaxation of the original problem (C.1) for any choice of $\nu$, any solution to problem (C.6) that satisfies the coupling constraint of problem (C.1) must also be a solution to this original problem. All that remains is the question of finding a solution $\nu^{\star}$ to the dual problem (C.10).

\section*{C.3.3 Solving the dual problem}

The dual problem (C.10) is a convex optimization problem that is easily solvable in practice, even for very large $n$ and $m$. In many cases, we can use a number of off-the-shelf solvers such as SCS [ODo+16], Hypatia [CKV21], and Mosek [ApS24a]. For example, a relatively
effective way of minimizing functions when the gradient is easily evaluated is the L-BFGS-B algorithm $[\mathrm{Byr}+95$; Zhu +97 ; MN11]: given a way of evaluating the dual function $g(\nu)$ and its gradient $\nabla g(\nu)$ at some point $\nu$, the algorithm will find an optimal $\nu^{\star}$ fairly quickly in practice. (See §C. 7 for timings.) By definition, the function $g$ is easy to evaluate if the subproblems (C.7) and (C.8) are easy to evaluate. Additionally the right hand side of equation (C.11) gives us a way of evaluating the gradient $\nabla g$, essentially for free, since we typically receive the optimal $\Psi^{\star}$ and $\Delta_{i}^{\star}$ as a consequence of computing $\bar{U}$ and $\mathbf{a r b}_{i}$.

Interface. In order for a user to specify and solve the dual problem (C.10) (and therefore the original problem) it suffices for the user to specify (a) some way of evaluating $\bar{U}$ and its optimal $\Psi$ for problem (C.7) and (b) some way of evaluating the arbitrage problem (C.8) and its optimal trade $\Delta_{i}^{\star}$ for each market $i$ that the user wishes to include. New markets can be easily added by simply specifying how to arbitrage them, which, as we will see next, turns out to be straightforward for most practical decentralized exchanges. The Julia interface required for the software package described in §C. 6 is a concretization of the interface described here.

\section*{C. 4 Swap markets}

In practice, most markets trade only two assets; we will refer to these kinds of markets as swap markets. Because these markets are so common, the performance of our algorithm is primarily governed by its ability to solve (C.8) quickly on these two asset markets. We show practical examples of these computations in section C.5. In this section, we will suppress the index $i$ with the understanding that we are referring to a specific market $i$.

\section*{C.4.1 General swap markets}

Swap markets are simple to deal with because their trading behavior is completely specified by the forward exchange function $[\mathrm{Ang}+22 \mathrm{~b}]$ for each of the two assets. In what follows, the forward trading function $f_{1}$ will denote the maximum amount of asset 2 that can be received by trading some fixed amount $\delta_{1}$ of asset 1 , i.e., if $T \subseteq \mathbf{R}^{2}$ is the trading set for a specific swap market, then
$$
f_{1}\left(\delta_{1}\right)=\sup \left\{\lambda_{2} \mid\left(-\delta_{1}, \lambda_{2}\right) \in T\right\}, \quad f_{2}\left(\delta_{2}\right)=\sup \left\{\lambda_{1} \mid\left(\lambda_{1},-\delta_{2}\right) \in T\right\}
$$

In other words, $f_{1}\left(\delta_{1}\right)$ is defined as the largest amount $\lambda_{2}$ of token 2 that one can receive for tendering a basket of ( $\delta_{1}, 0$ ) to the market. The forward trading function $f_{2}$ has a similar interpretation. If $f_{1}\left(\delta_{1}\right)$ is finite, then this supremum is achieved since the set $T$ is closed.

Trading function. If the set $T$ has a simple trading function representation, as in (C.3), it is not hard to show that the function $f_{1}$ is the unique (pointwise largest) function that satisfies
$$
\begin{equation*}
\varphi\left(R_{1}+\gamma \delta_{1}, R_{2}-f_{1}\left(\delta_{1}\right)\right)=\varphi\left(R_{1}, R_{2}\right) \tag{C.12}
\end{equation*}
$$
whenever $\varphi$ is nondecreasing, which may be assumed for all CFMMs [AC20a], and similarly for $f_{2}$. (Note the equality here, compared to the inequality in the original definition (C.2).)

Properties. The functions $f_{1}$ and $f_{2}$ are concave, since the trading set $T$ is convex, and nonnegative, since $0 \in T$ by assumption. Additionally, we can interpret the directional derivative of $f_{j}$ as the current marginal price of the received asset, denominated in the tendered asset. Specifically, we define
$$
\begin{equation*}
f_{j}^{\prime}\left(\delta_{j}\right)=\lim _{h \rightarrow 0^{+}} \frac{f_{j}\left(\delta_{j}+h\right)-f_{j}\left(\delta_{j}\right)}{h} \tag{C.13}
\end{equation*}
$$

This derivative is sometimes referred to as the price impact function [ACE22a]. Intuitively, $f_{1}^{\prime}(0)$ is the current price of asset 1 quoted by the swap market before any trade is made, and $f_{1}^{\prime}(\delta)$ is the price quoted by the market to add an additional $\varepsilon$ units of asset 1 to a trade of size $\delta$, for very small $\varepsilon$. We note that in the presence of fees, the marginal price to add to a trade of size $\delta$, i.e., $f_{1}^{\prime}(\delta)$, will be lower than the price to do so after the trade has been made [AC20a].

Swap market arbitrage problem. Equipped with the forward exchange function, we can specialize (C.8). Overloading notation slightly by writing $\left(\nu_{1}, \nu_{2}\right) \geq 0$ for $A_{i}^{T} \nu$ we define the swap market arbitrage problem for a market with forward exchange function $f_{1}$ :
$$
\begin{array}{ll}
\operatorname{maximize} & -\nu_{1} \delta_{1}+\nu_{2} f_{1}\left(\delta_{1}\right) \\
\text { subject to } & \delta_{1} \geq 0 \tag{C.14}
\end{array}
$$
with variable $\delta_{1} \in \mathbf{R}$ We can also define a similar arbitrage problem for $f_{2}$ :
$$
\begin{array}{ll}
\text { maximize } & \nu_{1} f_{2}\left(\delta_{2}\right)-\nu_{2} \delta_{2} \\
\text { subject to } & \delta_{2} \geq 0
\end{array}
$$
with variable $\delta_{2} \in \mathbf{R}$. Since $f_{1}$ and $f_{2}$ are concave, both problems are evidently convex optimization problems of one variable. Because they are scalar problems, these problems can be easily solved by bisection or ternary search. The final solution is to take whichever of these two problems has the largest objective value and return the pair in the correct order. For example, if the first problem (C.14) has the highest objective value with a solution $\delta_{1}^{\star}$, then $\Delta^{\star}=\left(-\delta_{1}^{\star}, f\left(\delta_{1}^{\star}\right)\right)$ is a solution to the original arbitrage problem (C.8). (For many practical trading sets $T$, it can be shown that at most one problem will have strictly positive objective value, so it is possible to 'short-circuit' solving both problems if the first evaluation has positive optimal value.)

Problem properties. One way to view each of these problems is that they 'separate' the solution space of the original arbitrage problem (C.8) into two cases: one where an optimal solution $\Delta^{\star}$ for (C.8) has $\Delta_{1}^{\star} \leq 0$ and one where an optimal solution has $\Delta_{2}^{\star} \leq 0$. (Any optimal point $\Delta^{\star}$ for the original arbitrage problem (C.8) will never have both $\Delta_{1}^{\star}<0$ and $\Delta_{2}^{\star}<0$ as that would be strictly worse than the 0 trade for $\nu>0$, and no reasonable market will have $\Delta_{1}^{\star}>0$ and $\Delta_{2}^{\star}>0$ since the market would be otherwise 'tendering free money' to the trader.) This observation means that, in order to find an optimal solution to the original optimal arbitrage problem (C.8), it suffices to solve two scalar convex optimization problems.

Optimality conditions. The optimality conditions for problem (C.14) are that, if
$$
\begin{equation*}
\nu_{2} f_{1}^{\prime}(0) \leq \nu_{1} \tag{C.15}
\end{equation*}
$$
then $\delta_{1}^{\star}=0$ is a solution. Otherwise, we have
$$
\delta_{1}^{\star}=\sup \left\{\delta \geq 0 \mid \nu_{2} f_{1}^{\prime}(\delta) \geq \nu_{1}\right\}
$$

Similar conditions hold for the problem over $\delta_{2}$. If the function $f_{1}^{\prime}$ is continuous, not just semicontinuous, then the expression above simplifies to finding a root of a monotone function:
$$
\begin{equation*}
\nu_{2} f_{1}^{\prime}\left(\delta_{1}^{\star}\right)=\nu_{1} . \tag{C.16}
\end{equation*}
$$

If there is no root and condition (C.15) does not hold, then $\delta_{1}^{\star}=\infty$. However, the solution will be finite for any trading set that does not contain a line, i.e., the market does not have 'infinite liquidity' at a specific price.

No-trade condition. Note that using the inequality (C.15) gives us a simple way of verifying whether we will make any trade with market $T$, given some prices $\nu_{1}$ and $\nu_{2}$. In particular, the zero trade is optimal whenever
$$
f_{1}^{\prime}(0) \leq \frac{\nu_{1}}{\nu_{2}} \leq \frac{1}{f_{2}^{\prime}(0)}
$$

We can view the interval $\left[f_{1}^{\prime}(0), 1 / f_{2}^{\prime}(0)\right]$ as a type of 'bid-ask spread' for the market with trading set $T$. (In constant function market makers, this spread corresponds to the fee $\gamma$ taken from the trader.) This 'no-trade condition' lets us save potentially wasted effort of computing an optimal arbitrage trade as, in practice, most trades in the original problem will be 0 .

Bounded liquidity. In some cases, we can easily check not only when a trade will not be made (say, using condition (C.15)), but also when the 'largest possible trade' will be made. (We will define what this means next.) Markets for which there is a 'largest possible trade' are called bounded liquidity markets. We say a market has bounded liquidity in asset 2 if there is a finite $\delta_{1}$ such that $f_{1}\left(\delta_{1}\right)=\sup f_{1}$, and similarly for $f_{2}$. In other words, there is a finite input $\delta_{1}$ which will give the maximum possible amount of asset 2 out. A market has bounded liquidity if it has bounded liquidity on both of its assets. A bounded liquidity market then has a notion of a 'minimum price'. First, define
$$
\delta_{1}^{-}=\inf \left\{\delta_{1} \geq 0 \mid f_{1}\left(\delta_{1}\right)=\sup f_{1}\right\}
$$
i.e., $\delta_{1}^{-}$is the smallest amount of asset 1 that can be tendered to receive the maximum amount the market is able to supply. We can then define the minimum supported price as the left derivative of $f_{1}$ at $\delta_{1}^{-}$:
$$
f_{1}^{-}\left(\delta_{1}^{-}\right)=\lim _{h \rightarrow 0^{+}} \frac{f\left(\delta_{1}^{-}\right)-f\left(\delta_{1}^{-}-h\right)}{h} .
$$

The first-order optimality conditions imply that $\delta_{1}^{-}$is a solution to the scalar optimal arbitrage problem (C.14) whenever
$$
f_{1}^{-}\left(\delta_{1}^{-}\right) \geq \frac{\nu_{1}}{\nu_{2}}
$$

In English, this can be stated as: if the minimum supported marginal price we receive for $\delta_{1}^{-}$ is still larger than the price being arbitraged against, $\nu_{1} / \nu_{2}$, it is optimal to take all available liquidity from the market. Using the same definitions for $f_{2}$, we find that the only time the full problem (C.14) needs to be solved is when the price being arbitraged against $\nu_{1} / \nu_{2}$ lies in the interval
$$
\begin{equation*}
f_{1}^{-}\left(\delta_{1}^{-}\right)<\frac{\nu_{1}}{\nu_{2}}<\frac{1}{f_{2}^{-}\left(\delta_{2}^{-}\right)} \tag{C.17}
\end{equation*}
$$
(It may be the case that $f_{2}^{-}\left(\delta_{2}^{-}\right)=0$ in which case we define the right hand side to be $\infty$.) We will call this interval of prices the active interval for a bounded liquidity market.

Example. In the case of Uniswap v3 [Ada+21a], we have a collection of, say, $i=1, \ldots, s$ bounded liquidity product functions (C.4), where the parameters $\alpha_{k}, \beta_{k}>0$ are chosen such that all of the active price intervals, as defined in (C.17), are disjoint. (An explicit form for this trading function is given in the next section, equation (C.18).) Solving the arbitrage problem (C.14) over this collection of CFMMs is relatively simple. Since all of the intervals are disjoint, any price $\nu_{1} / \nu_{2}$ can lie in at most one of the active intervals. We therefore do not need to compute the optimal trade for any interval, except the single interval where $\nu_{1} / \nu_{2}$ lies, which can be done in closed form. We also note that this 'trick' applies to any collection of bounded liquidity markets with disjoint active price intervals.

\section*{C. 5 Closed form solutions}

Here, we cover some of the special cases where it is possible to analytically write down the solutions to the arbitrage problems presented previously.

Geometric mean trading function. Some of the most popular swap markets, for example, Uniswap v2 and most Balancer pools, which total over $\$ 2 \mathrm{~B}$ in reserves, are geometric mean markets (C.5) with $n=2$. This trading function can be written as
$$
\varphi(R)=R_{1}^{w} R_{2}^{1-w}
$$
where $0<w<1$ is a fixed parameter. This very common trading function admits a closedform solution to the arbitrage problem (C.8). Using (C.12), we can write
$$
f_{1}\left(\delta_{1}\right)=R_{2}\left(1-\left(\frac{1}{1+\gamma \delta_{1} / R_{1}}\right)^{\eta}\right)
$$
where $\eta=w /(1-w)$. (A similar equation holds for $f_{2}$.) Using (C.15) and (C.16), and defining
$$
\delta_{1}=\frac{R_{1}}{\gamma}\left(\left(\eta \gamma \frac{\nu_{2}}{\nu_{1}} \frac{R_{2}}{R_{1}}\right)^{1 /(\eta+1)}-1\right),
$$
we have that $\delta_{1}^{\star}=\max \left\{\delta_{1}, 0\right\}$ is an optimal point for (C.14). Note that when we take $w=1 / 2$ then $\eta=1$ and we recover the optimal arbitrage for Uniswap given in $[\mathrm{Ang}+20$, App. A].

Bounded liquidity variation. The bounded liquidity variation (C.4) of the product trading function satisfies the definition of bounded liquidity given in §C.4.1, whenever $\alpha, \beta>0$. We can write the forward exchange function for the bounded liquidity product function (C.4), using (C.12), as
$$
f_{1}(\delta)=\min \left\{R_{2}, \frac{\gamma \delta\left(R_{2}+\beta\right)}{R_{1}+\gamma \delta+\alpha}\right\}
$$

The 'min' here comes from the definition of a CFMM: it will not accept trades which pay out more than the available reserves. The maximum amount that a user can trade with this market, which we will write as $\delta_{1}^{-}$, is when $f_{1}\left(\delta_{1}^{-}\right)=R_{2}$, i.e.,
$$
\delta_{1}^{-}=\frac{1}{\gamma} \frac{R_{2}}{\beta}\left(R_{1}+\alpha\right)
$$
(Note that this can also be derived by taking $f_{1}\left(\delta_{1}\right)=R_{2}$ in (C.12) with the invariant (C.4).) This means that
$$
f_{1}^{-}\left(\delta_{1}^{-}\right)=\gamma \frac{\beta^{2}}{\left(R_{1}+\alpha\right)\left(R_{2}+\beta\right)}
$$
is the minimum supported price for asset 1 . As before, a similar derivation yields the case for asset 2 . Writing $k=\left(R_{1}+\alpha\right)\left(R_{2}+\beta\right)$, we see that we only need to solve (C.14) if the price $\nu_{1} / \nu_{2}$ is in the active interval (C.17),
$$
\begin{equation*}
\frac{\gamma \beta^{2}}{k}<\frac{\nu_{1}}{\nu_{2}}<\frac{k}{\gamma \alpha^{2}} \tag{C.18}
\end{equation*}
$$

Otherwise, we know one of the two 'boundary' solutions, $\delta_{1}^{-}$or $\delta_{2}^{-}$, suffices.

\section*{C. 6 Implementation}

We have implemented this algorithm in CFMMRouter.jl, a Julia [Bez+17] package for solving the optimal routing problem. Our implementation is available at
```
https://github.com/bcc-research/CFMMRouter.jl
```

and includes implementations for both weighted geometric mean CFMMs and Uniswap v3. In this section, we provide a concrete Julia interface for our solver.

\section*{C.6.1 Markets}

Market interface. As discussed in §C.3.3, the only function that the user needs to implement to solve the routing problem for a given market is
```
find_arb!(\Delta, \Lambda, mkt, v).
```


This function solves the optimal arbitrage problem (C.8) for a market mkt (which holds the relevant data about the trading set $T$ ) with dual variables v (corresponding to $A_{i}^{T} \nu$ in the original problem (C.8)). It then fills the vectors $\Delta$ and $\Lambda$ with the negative part of the solution, $-\Delta_{-}^{\star}$, and positive part of the solution, $\Delta_{+}^{\star}$, respectively.

For certain common markets (e.g., geometric mean and Uniswap v3), we provide specialized, efficient implementations of find_arb!. For general CFMMs where the trading function, its gradient, and the Hessian are easy to evaluate, one can use a general-purpose primal-dual interior point solver. For other more complicated markets, a custom implementation may be required.

Swap markets. The discussion in §C. 4 and the expression in (C.16) suggests a natural, minimal interface for swap markets. Specifically, we can define a swap market by implementing the function get_price ( $\Delta$ ). This function takes in a vector of inputs $\Delta \in \mathbf{R}_{+}^{2}$, where we assume that only one of the two assets is being tendered, i.e., $\Delta_{1} \Delta_{2}==0$, and returns $f_{1}^{\prime}\left(\Delta_{1}\right)$, if $\Delta_{1}>0$ or $f_{2}^{\prime}\left(\Delta_{2}\right)$ if $\Delta_{2}>0$. With this price impact function implemented, one can use bisection to compute the solution to (C.16). When price impact function has a closed form and is readily differentiable by hand, it is possible to use a much faster Newton method to solve this problem. In the case where the function does not have a simple closed form, we can use automatic differentiation (e.g., using ForwardDiff.jl [RLP16]) to generate the gradients for this function.

Aggregate CFMMs. In the special case of aggregate, bounded liquidity CFMMs, the price impact function often does not have a closed form. On the other hand, whenever the active price intervals are disjoint, we can use the trick presented in §C.4.1 to quickly arbitrage an aggregate CFMM. For example, a number of Uniswap v3 markets are actually composed of many thousands of bounded liquidity CFMMs. Treating each of these as their own market, without any additional considerations, significantly increases the size and solution complexity of the problem.

In this special case, each aggregate market 'contains' $s$ trading sets, each of which has disjoint active price intervals with all others. We will write these intervals as $\left(p_{i}^{-}, p_{i}^{+}\right)$for each trading set $i=1, \ldots, s$, and assume that these are in sorted order $p_{i-1}^{+} \leq p_{i}^{-}<p_{i}^{+} \leq p_{i+1}^{+}$. Given some dual variables $\nu_{1}$ and $\nu_{2}$ for which to solve the arbitrage problem (C.8), we can then run binary search over the sorted intervals (taking $O(\log (s))$ time) to find which of the intervals the price $\nu_{1} / \nu_{2}$ lies in. We can compute the optimal arbitrage for this 'active' trading set, and note that the remaining trading sets all have a known optimal trade (from the discussion in §C.4.1) and require only constant time. For Uniswap v3 and other aggregate CFMMs, this algorithm is much more efficient from both a computational and memory perspective when compared with a direct approach that considers all $s$ trading sets separately.

Other functions. If one is solving the arbitrage problem multiple times in a row, it may be helpful to implement the following additional functions:
1. swap! (cfmm, $\Delta$ ) : updates cfmm's state following a trade $\Delta$.
2. update_liquidity! (cfmm, [range,] L): adds some amount of liquidity $\mathrm{L} \in \mathbf{R}_{+}^{2}$, optionally includes some interval range $=(p 1, p 2)$.

\section*{C.6.2 Utility functions.}

Recall that the dual problem relies on a slightly transformed version of the Fenchel conjugate, which is the optimal value of problem (C.7). To use LBFGS-B (and most other optimization methods), we need to be able to evaluate this function $\bar{U}(\nu)$ and its gradient $\nabla \bar{U}(\nu)$, which is the solution $\Psi^{\star}$ to (C.7) with parameter $\nu$. Thus, utility functions are implemented as objects that implement the following interface:
- f(objective, v) evaluates $\bar{U}$ at v.
- grad! (g, objective, v) evaluates $\nabla \bar{U}$ at v and stores it in $g$.
- lower_limit (objective) returns the lower bound of the objective.
- upper_limit(objective) returns the upper bound of the objective.

The lower and upper bounds can be found by deriving the conjugate function. For example, for the 'total arbitrage' objective $U(\Psi)=c^{T} \Psi-I(\Psi \geq 0)$, where a trader wants to tender no tokens to the network, but receive any positive amounts out with value proportional to some nonnegative vector $c \in \mathbf{R}_{+}^{n}$, has $\bar{U}(\nu)=0$ if $\nu \geq c$ and $\infty$ otherwise. Thus, we have the bounds $c \leq \nu<\infty$, and gradient $\nabla \bar{U}(\nu)=0$. We provide implementations for arbitrage and for basket liquidations in our Julia package. (See [Ang+22a, §3] for definitions.)

\section*{C. 7 Numerical results}

We compare the performance of our solver against the commercial, off-the-shelf convex optimization solver Mosek, accessed through JuMP [DHL17; Leg+21]. In addition, we use our solver with real, on-chain data to illustrate the benefit of routing an order through multiple markets rather than trading with a single market. Our code is available at
https://github.com/bcc-research/router-experiments.

Performance. We first compare the performance of our solver against Mosek [ApS24a], a widely-used, performant commercial convex optimization solver. We generate $m$ swap markets over a global universe of $2 \sqrt{m}$ assets. Each market is randomly generated with reserves uniformly sampled from the interval between 1000 and 2000 , denoted $R_{i} \sim \mathcal{U}(1000,2000)$, and is a constant product market with probability 0.5 and a weighted geometric mean market with weights $(0.8,0.2)$ otherwise. (These types of swap markets are common in protocols such as Balancer [MM19a].) We run arbitrage over the set of markets, with 'true prices' for each asset randomly generated as $p_{i} \sim \mathcal{U}(0,1)$. For each $m$, we use the same parameters (markets and price) for both our solver and Mosek. Mosek is configured with default parameters. All experiments are run on a MacBook Pro with a 2.3 GHz 8-Core Intel i9 processor. In figure C.1, we see that as the number of pools (and tokens) grow, our method begins to

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-113.jpg?height=561&width=1532&top_left_y=226&top_left_x=263}
\captionsetup{labelformat=empty}
\caption{Figure C.1: Solve time of Mosek vs. CFMMRouter.jl (left) and the resulting objective values for the arbitrage problem, with the dashed line indicating the relative increase in objective provided by our method (right).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-113.jpg?height=547&width=1610&top_left_y=981&top_left_x=257}
\captionsetup{labelformat=empty}
\caption{Figure C.2: Average price of market sold ETH in routed vs. single-pool (left) and routed vs. single-pool surplus liquidation value (right).}
\end{figure}
dramatically outperform Mosek and scales quite a bit better. We note that the weighted geometric mean markets are especially hard for Mosek, as they must be solved as power cone constraints. Constant product markets may be represented as second order cone constraints, which are quite a bit more efficient for many solvers. Furthermore, our method gives a higher objective value, often by over $50 \%$. We believe this increase stems from Mosek's use of an interior point method and numerical tolerances. The solution returned by Mosek for each market will be strictly inside the associated trading set, but we know that any rational trader will choose a trade on the boundary.

Real data: trading on chain. We show the efficacy of routing by considering a swap from WETH to USDC (i.e., using the basket liquidation objective to sell WETH for USDC). Using on-chain data from the end of a recent block, we show in figure C. 2 that as the trade size increases, routing through multiple pools gives an increasingly better average price than using the Uniswap v3 USDC-WETH $.3 \%$ fee tier pool alone. Specifically, we route orders through the USDC-WETH $.3 \%$, WETH-USDT $.3 \%$, and USDC-USDT $.01 \%$ pools. This is
the simplest example in which we can hope to achieve improvements from routing, since two possible routes are available to the seller: a direct route through the USDC-WETH pool; and an indirect route that uses both the WETH-USDT pool and the USDC-USDT pool.

\section*{C. 8 Conclusion}

We constructed an efficient algorithm to solve the optimal routing problem. Our algorithm parallelizes across markets and involves solving a series of optimal arbitrage problems at each iteration. To facilitate efficient subproblem solutions, we introduced an interface for swap markets, which includes aggregate CFMMs.

We note that we implicitly assume that the trading sets are known exactly when the routing problem is solved. This assumption, however, ignores the realities of trading on chain: unless our trades execute first in the next block, we are not guaranteed that the trading sets for each market are the same as those in the last block. Transactions before ours in the new block may have changed prices (and reserves) of some of the markets we are routing through. This observation naturally suggests robust routing as a natural direction for future research. Furthermore, efficient algorithms for routing with fixed transaction costs (e.g., gas costs) are another interesting direction for future work (see [Ang+22a, §5] for the problem formulation).

\section*{Appendix D}

\section*{The Geometry of Constant Function Market Makers}

\section*{D. 1 Introduction}

The study of automated market makers has existed for many decades, with roots in the scoring rule literature dating back to at least the 1950s [McC56]. However, these mechanisms only reached mass adoption after being implemented as decentralized exchanges (DEXs) on blockchains. These exchanges (including Curve Finance [Ego19], Uniswap [AZR20], and Balancer [MM19b], among others) have facilitated trillions of dollars in cumulative trading volume and maintain a collective daily trading volume of several billion dollars. Surprisingly, the types of automated market maker that are most popular in practice bear little resemblance to those proposed prior to the invention of blockchains. Instead, the most popular blockchain-based automated market makers are what are known as the constant function market makers, or CFMMs. These market makers are generally simpler than earlier market maker designs, such as those based on the logarithmic scoring rule, and provide a means for efficient liquidity aggregation and order routing. But why have these mechanisms succeeded?

One of the main reasons that these mechanisms have been so popular in the cryptocurrency space is that solving the optimal arbitrage problem - the problem of how much to trade in order to equalize prices between CFMMs and other venues-is generally (computationally) 'easy' [Ang+22a]. This ease comes directly from the fact that CFMMs satisfy a general, very geometric, notion of convexity. Though the initial line of work, which defined CFMMs as a useful class, focused on their geometric properties [AC20b], the majority of research on CFMMs has focused on analytic properties of CFMMs that depend on explicit parameterizations [LP21; WM22; MMR23a; SKM23; FPW23; MMR23b; Goy+23; FKP23; FP23].

There are important reasons to examine the geometry of CFMMs directly. First, a geometric lens leads to very natural statements for many of the properties of, and operations one can perform on, CFMMs. Second, many 'surprising' decisions made by developers that 'worked in practice' can be explained by understanding the geometry of CFMMs; for example, intuitively, the 'curvature' of a CFMM corresponds to a notion of liquidity [ACE22b], which was known by practitioners well before its formalization. Third, the geometric setting for

CFMMs is very general and rarely requires notions of differentiability, homogeneity, or other similar properties. Finally, the geometric view of CFMMs allows for reasoning about CFMMs without regards to its particular trading function and/or its representation.

\section*{D.1.1 What is a CFMM?}

A constant function market maker, originally introduced in [AC20b], is a type of automated market maker typically defined by a trading function $\varphi: \mathbf{R}^{n} \rightarrow \mathbf{R}$. The state of the CFMM is defined by the quantity of assets it holds, which we represent as a nonnegative vector $R \in \mathbf{R}_{+}^{n}$. Traders can interact with the CFMM by proposing some trade $\Delta \in \mathbf{R}^{n}$, where a positive entry, say, $\Delta_{i}>0$, denotes that the trader wishes to receive some amount $\Delta_{i}$ of asset $i$, while a negative entry, say $\Delta_{j}<0$, denotes that the trader wishes to tender asset $j$ to the market. The CFMM accepts the trade $\Delta$, given its current reserves $R$, only if
$$
\begin{equation*}
\varphi(R-\Delta) \geq \varphi(R) \tag{D.1}
\end{equation*}
$$
(If this inequality is not satisfied, the trade is ignored and no transaction occurs.) In other words, the trading function evaluated at the new reserves $R-\Delta$, if the trade were to be accepted, must be at least as large as the trading function evaluated at the current reserves, $R$. Assuming that $\varphi$ is continuous, a trader will always prefer more of a token than less, all else being equal, so the proposed trade should always make the inequality (D.1) hold at equality. (Hence the name constant function market maker.)

The fact that this type of automated market maker is simple to describe and implement, along with having many strong theoretical guarantees, has been part of its reason for success, especially within difficult-to-secure environments such as public blockchains. Despite their simple description, CFMMs have spawned a large amount of research into their financial, arbitrage, and routing properties (e.g., [Ang+22a; DKP21b; Dia+23; FMW23; MD23], among many others). For a recent survey see [Ang+22c].

Example. Perhaps the most famous CFMM is the constant product market maker, first implemented by Uniswap [AZR20] which has facilitated almost half a trillion USD of volume, as of October 2023. ${ }^{1}$ This CFMM trades two tokens and is defined by the trading function
$$
\varphi(R)=R_{1} R_{2}
$$

As any 'reasonable' trades will leave this function unchanged, we will set $k=R_{1} R_{2}$, where $R$ denote the current reserves of the CFMM. Of course, there are many other trading functions which would provide equivalent behavior under (D.1). For example, any monotonically increasing transformation applied to $\varphi$ would yield the same behavior. One example is the trading function
$$
\tilde{\varphi}(R)=\sqrt{R_{1} R_{2}},
$$
which we note is concave, homogeneous, and nondecreasing. It is 'equivalent' to the original trading function in that, for any trade $\Delta \in \mathbf{R}^{n}$
$$
\varphi(R-\Delta) \geq k, \quad \text { if, and only if, } \quad \tilde{\varphi}(R-\Delta) \geq \sqrt{k}
$$

\footnotetext{
${ }^{1}$ For the most up-to-date on-chain information, see https://defillama.com/protocol/ uniswap-v2.
}

Another possible definition of the constant product market maker is via its portfolio value function. Given a price vector $c \in \mathbf{R}_{+}^{n}$, where $c_{i}$ denotes the price of asset $i$ in some common numéraire, the portfolio value function $V: \mathbf{R}_{+}^{n} \rightarrow \mathbf{R}$ maps these prices $c$ to the value held by the CFMM, under no-arbitrage. (We define this more carefully in §D.2.4.) In this case, the portfolio value function for the constant product market maker is
$$
V(c)=2 \sqrt{k c_{1} c_{2}} .
$$

This representation of the constant product market maker corresponds to the same underlying object and can be shown to be unique; i.e., any representation of the constant product market maker's trading function corresponds to exactly this portfolio value function.

A geometric viewpoint. In some sense, these representations, via the trading functions or the portfolio value functions, all point to the same underlying object. A useful set of definitions would be agnostic to the particular representation of the object.

Nearly by definition, geometric descriptions of CFMMs turn out to be unique and relatively simple to handle. For instance, there is a natural 'addition' operator for CFMMs using a geometric representation, which we present in §D.2.1. Describing the corresponding operation on trading functions is not obvious and likely has no natural analogue. This idea that certain operations such as addition, are 'easy' to perform on CFMMs, when defined geometrically, is one of the reasons that proofs using geometry can be significantly more succinct when compared to more functional approaches.

This paper. In this paper, we focus on representing CFMMs via classical geometric objects such as convex sets and cones, assuming a bare minimum of requirements. Using these objects, we replicate the results of a number of papers for CFMMs without fees (also known as the 'path-independent' CFMMs) and many results in the case of a single trade with no restrictions on the CFMM. The key objects we look at are particular cones which we call the liquidity cones, and their corresponding conic duals. We construct many of the 'usual objects' such as trading functions, portfolio value functions, no arbitrage intervals, and so on, directly from these objects. This leads to a number of interesting results: for example, that every (path-independent) CFMM has a canonical trading function that is concave, homogeneous, and nondecreasing (a so-called consistent function), along with new proofs for older, previously known results. More broadly, this paper presents a general set of duality results and equivalences for the class of consistent functions, which is of independent interest. We assume a reasonable amount of familiarity with convex optimization and provide a very short primer on conic duality in appendix D.5.1 as a refresher.

\section*{D. 2 Fee-free constant function market makers}

In this section, we consider the general case of constant function market makers that are pathindependent. We show the connection between these 'path-independent' or 'fee-free' constant function market makers and 'general' constant function market makers later, in §D.3.4. We consider this case first as this is the most common case in the literature [AC20b; FPW23; Goy+23], and is a good starting point for the more general case.

Section layout. The section begins with a basic set of requirements (sometimes called 'axioms') which are of a different form than the standard assumptions made in many texts. We will show that from these requirements, which are mostly geometric in origin, we can derive many known results and a number of generalizations that, to our knowledge, are not known in the overall literature. For example, one important case is that any CFMM has a canonical trading function that is homogeneous, nondecreasing, and concave. This is usually taken as an assumption in some form (see, e.g., [AEC21; FPW23; SKM23]), but we show here that it is true of any CFMM satisfying some basic properties that are essentially necessary for a CFMM to be reasonable. (Indeed, these properties are almost always part of a much longer list, or are easy consequences of a subset of assumptions generally made in the literature.) This geometric set up also simplifies a number of known statements in the literature, such as those of [AEC23], by showing that the equivalence of a portfolio value function and a trading function is a special case of conic duality.

\section*{D.2.1 Reachable set}

We will define the reachable set of reserves as a set $S \subseteq \mathbf{R}^{n}$ satisfying certain requirements. This set will represent the valid holdings of a constant function market maker (CFMM). In general, if $R \in S$ are the current reserves of the constant function market maker, then any trader may change the reserves to $R^{\prime} \in S$ by selling $R^{\prime}-R$ to the CFMM. The trader would then receive the entries of $R^{\prime}-R$ which are negative, and tender the entries which are positive to the CFMM. In a certain sense, we may view the reachable set $S$ as the set of valid states available to the CFMM.

Definition. We say a set $S$ is a reachable set (which defines a fee-free, or 'path independent' CFMM) if it satisfies these rules or 'axioms':
1. All reserves are nonnegative; that is, $S \subseteq \mathbf{R}_{+}^{n}$
2. The set $S$ is nonempty, closed, and convex
3. The set $S$ is upward closed; i.e., if $R \in S$, then any $R^{\prime} \geq R$ has $R^{\prime} \in S$

From these three rules, we will recover (and generalize) many of the results known in the literature. In general, while we do not assume that $0 \notin S$, we note that this is a silly case as we would then have $S=\mathbf{R}_{+}^{n}$, so this case is often excluded from many of the proofs presented.

High-level interpretation. The first requirement means that a constant function market maker cannot take on debt, or that the position is always solvent. Many, but not all, results hold with some slight modifications, even in the case where this condition is relaxed. The convexity requirement roughly corresponds to the fact that increasing the size of a trade does not result in a better exchange rate for the trader. The nonemptyness of $S$ just means that $S$ is nontrivial, while the closedness is a technical condition. Finally, the 'upwards closed' condition means that, if a CFMM accepts some trade, then it would always accept a different trade that tenders more of any asset. (This condition is not technically necessary:

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-119.jpg?height=609&width=1617&top_left_y=247&top_left_x=244}
\captionsetup{labelformat=empty}
\caption{Figure D.1: The set of reachable reserves for Uniswap (left) and Uniswap v3 (right).}
\end{figure}
it suffices that, given a nonempty set $S$ satisfying the first condition, the set $\tilde{S}=S+\mathbf{R}_{+}^{n}$ satisfies the second condition above. Almost all results shown below hold in this case.) The last condition also lets us interpret the boundary of $S$ as a Pareto-optimal frontier for the possible reserves in the sense that no rational trader would ever trade on the interior of $S$.

Examples. One of the canonical examples of a reachable set is that of Uniswap [AZR20], defined
$$
S=\left\{R \in \mathbf{R}_{+}^{2} \mid R_{1} R_{2} \geq k\right\},
$$
where $k>0$ is a constant. See figure D. 1 for an example. Another example is that of a 'tick' in Uniswap v3 [Ada+21b], which is defined
$$
S=\left\{R \in \mathbf{R}_{+}^{2} \mid\left(R_{1}+\alpha\right)\left(R_{2}+\beta\right) \geq k\right\},
$$
where, again, $\alpha, \beta, k>0$ are some provided constants.

Quasiconcavity. Note that, in these examples, $S$ is the superlevel set of some quasiconcave, nondecreasing function. In fact, we can show that any nonempty set $S$ defined by
$$
\begin{equation*}
S=\left\{R \in \mathbf{R}_{+}^{n} \mid \psi(R) \geq \alpha\right\} \tag{D.2}
\end{equation*}
$$
with quasiconcave, nondecreasing $\psi: \mathbf{R}_{+}^{n} \rightarrow \mathbf{R} \cup\{-\infty\}$, generates a reachable set satisfying the required conditions. This includes [SKM23] and [FPW23] as a special case, though we do not require homogeneity. (Indeed, homogenity is not needed as an assumption as we will later show that one can always choose $\psi$ to be concave, nondecreasing, and homogeneous for any set $S$ satisfying the reachable set conditions, even when the 'original' function $\psi$ is not.) We may also replace the inequality with an equality and define the set
$$
S=\left\{R^{\prime} \in \mathbf{R}_{+}^{n} \mid R^{\prime} \geq R, \text { for some } \psi(R)=\alpha\right\} .
$$

Note that these two definitions are equivalent if $\psi$ is continuous in some neighborhood $\psi^{-1}(N)$ where $N$ is a neighborhood around $\alpha$.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-120.jpg?height=435&width=1544&top_left_y=242&top_left_x=283}
\captionsetup{labelformat=empty}
\caption{Figure D.2: Adding two Uniswap v3 bounded liquidity pools (left, middle) gives us another CFMM (right).}
\end{figure}

\section*{D.2.2 Composition rules}

An interesting consequence of the definition of reachable sets is that these sets, and therefore CFMMs, satisfy certain composition rules, some of which were known in the literature under additional assumptions [EH21]. These rules follow directly from the calculus of convex sets [BV04, §2.3] and require no additional assumptions than those given in §D.2.1.

Nonnegative scaling. Given a reachable set $S$, we may scale the set by $\alpha \geq 0$ to get $\alpha S$, which is another reasonable reachable set satisfying the conditions. (We will see later in §D.2.6 that this scaling corresponds to adding or removing liquidity to the CFMM.)

Set addition. We may also add any two reachable sets $S$ and $S^{\prime}$, which gives another reachable set
$$
S+S^{\prime}=\left\{R+R^{\prime} \mid R \in S, R^{\prime} \in S^{\prime}\right\}
$$

This set is convex and nonempty, and it is not hard to prove the set is closed since $S$ and $S^{\prime}$ are both contained in the positive orthant. It is also clear that $S+S^{\prime}$ is upward closed since each of $S$ and $S^{\prime}$ are upward closed. These sums have the 'simple' interpretation that $S+S^{\prime}$ are the possible combined holdings of the two CFMMs. Additionally, this, combined with nonnegative scaling, means that taking nonnegative linear combinations of trading sets always yields another trading set. We provide an example in figure D.2.

Nonnegative matrix multiplication. Another important rule is that multiplication by a nonnegative matrix $A \in \mathbf{R}_{+}^{n \times p}$ and 'upwards closure' of the resulting set gives another reachable set; i.e., the set
$$
A S+\mathbf{R}_{+}^{n}=\left\{R^{\prime} \in \mathbf{R}_{+}^{n} \mid R^{\prime} \geq A R \text { for some } R \in S\right\}
$$
is a reachable set. This operation can be interpreted when looking at each row $j=1, \ldots, n$ of $A$, which we write as $\tilde{a}_{j}^{T}$. Given some vector $R \in S$, then $\tilde{a}_{j}^{T} R=(A R)_{j}$. This entry, $(A R)_{j}$, can then be seen as a type of 'meta-asset', whose value is equal to a weighted basket of assets, where the weights are the entries of $\tilde{a}_{j}$. This is a reachable set since $A R \in \mathbf{R}_{+}^{n}$ for any $R \in S \subseteq \mathbf{R}_{+}^{n}$ and $A S$ is a convex set if $S$ is convex. (The set is clearly upward closed by definition.)

Special case: projection. An important special case is when the matrix $A$ projects all components of a trading set into a larger space. More specifically, let $A$ be a matrix of the form
$$
A=\left[\begin{array}{llll}
a_{1} & a_{2} & \cdots & a_{k}
\end{array}\right]
$$
with $a_{i} \in \mathbf{R}^{n}$ and $k \leq n$ being all distinct unit basis vectors (i.e., $a_{i}$ is 0 everywhere except at exactly one entry, where it is 1 ). We can interpret $A S+\mathbf{R}_{+}^{k}$ in the following way: if there is a 'list' of $n$ assets, and the CFMM defined by $S$ trades only $k$ of those assets, then $A S$ is a CFMM which trades these $k$ assets and zero of the remaining possible $n-k$ assets. (The CFMM will happily accept any of the remaining $n-k$ assets, but tender nothing for them: a trade no rational user would want.)

Intersection. Finally, we can take the intersection of reachable sets, which yields another reachable set; i.e., if $S$ and $S^{\prime}$ are reachable sets then $S \cap S^{\prime}$ is similarly a reachable set. This corresponds to a CFMM whose reachable reserves can only be those which the individual CFMMs have in common. Though this is not a natural operation for CFMMs which already exist on chain, it is a useful theoretical operation for constructing CFMMs with particular properties. (Indeed, we will see that constructing a CFMM from a portfolio value function, to be presented later, is possible only due to this intersection property.)

Aggregate CFMMs. Combining the previous two rules gives us a very general way of 'combining' CFMMs which trade different (but potentially overlapping) baskets of assets. Assume we have $m$ constant function market makers and a universe of $n$ assets. We will have CFMMs $i=1, \ldots, m$ with reachable sets $S_{i} \subseteq \mathbf{R}_{+}^{n_{i}}$, each trading a subset tokens of $n_{i}$ tokens. We introduce matrices $A_{i} \in \mathbf{R}_{+}^{n \times n_{i}}$ which map the 'local' basket of $n_{i}$ tokens for CFMM $i$ to the global universe of $n$ tokens. We have $\left(A_{i}\right)_{j k}=1$ if token $k$ in market $i$ 's local index corresponds to global token index $j$, and $\left(A_{i}\right)_{j k}=0$ otherwise. We note that the ordering of tokens in the local index does not need to be the same as the global ordering. Then the set
$$
\tilde{S}=\sum_{i=1}^{m} A_{i} S_{i}
$$
is a aggregate $C F M M$ which corresponds exactly to the set of all possible holdings for every CFMM in the network. Such CFMMs were first implicitly defined for Uniswap v3 [Ada+21b], and later used in [CAE21] to prove some basic approximation bounds, while [MMR23a] defined a notion of 'complexity' based on similar ideas, and, finally [Dia +23 ] defined them as part of the solution method for optimal routing.

Extensions to negative reserves. There are some basic generalizations of some of these conditions in the case where the set $S$ is not contained in the positive orthant. In this case, the CFMM can take on debt. If the debt is unbounded, it is possible to create sets $S$ and $S^{\prime}$ such that $S+S^{\prime}$ is not closed, so the resulting set would not be a reachable set. On the other hand, it is not hard to show that allowing bounded debt (i.e., there exists some $x \in \mathbf{R}_{+}^{n}$ such that $x+S \subseteq \mathbf{R}_{+}^{n}$ ) means that an analogous statement does still hold by a nearly identical proof.

\section*{D.2.3 Liquidity cone and canonical trading function}

In this subsection we introduce the liquidity cone for a reachable set $S$. The liquidity cone is a kind of 'homogenized' version of the reachable set defined previously that simplifies a number of later derivations. Its definition will also suggest a canonical trading function: a trading function that corresponds to the reachable set $S$ and is nondecreasing, homogeneous, and concave.

\section*{Liquidity cone}

The liquidity cone for reachable set $S$ is defined as
$$
\begin{equation*}
K=\mathbf{c l}\left\{(R, \lambda) \in \mathbf{R}^{n+1} \mid R / \lambda \in S, \lambda>0\right\} \tag{D.3}
\end{equation*}
$$
where $\mathbf{c l}$ is the closure of the set. The set $K$ is a cone as $(R, \lambda) \in K$ implies that $(\alpha R, \alpha \lambda) \in K$ for any $\alpha \geq 0$. The name 'liquidity cone' comes from the fact that, if $(R, \lambda) \in K$ then the largest such $\lambda$ indicates, roughly speaking, the amount of liquidity available from reserves $R$. (We will see what this means in a later section.)

Basic properties. The liquidity cone $K$ has some important properties we use later in this section. First, the set $K$ is nonempty as $S$ is nonempty and $S \times\{1\} \subseteq K$. We also have that $0 \in K$ as $K$ is nonempty and closed. To see this, if $y \in K$ then $\alpha y \in K$, so $\alpha \downarrow 0$ gives the result. The cone $K$ is also a convex cone as it is the closure of the perspective transform on the convex set $S \times \mathbf{R}_{++}$(see, e.g., [BV04, §2.3.3]).

Upward closedness. The cone $K$ is not upward closed, but is 'almost upward closed' in the following sense: if $(R, \lambda) \in K$ and $R^{\prime} \geq R$ with $\lambda^{\prime} \leq \lambda$ then $\left(R^{\prime}, \lambda^{\prime}\right) \in K$. In particular, note that the inequality over $\lambda$ is reversed. Showing this fact is just a definitional exercise.

Positive reachability. We also have that,
$$
\begin{equation*}
\left(\mathbf{R}_{++}^{n}, 0\right) \subseteq K \tag{D.4}
\end{equation*}
$$

This follows from the fact that the set $S$ is nonempty. To see this, let $R \in S$ and note that, for any strictly positive vector $R^{\prime} \in \mathbf{R}_{++}^{n}$ we know that $R^{\prime} / \lambda \geq R$ for $\lambda$ small enough, so $\left(R^{\prime}, \lambda\right) \in K$. Finally, since $\left(R^{\prime}, \lambda\right) \in K$ implies that $\left(R^{\prime}, \lambda^{\prime}\right) \in K$ for any $\lambda^{\prime} \leq \lambda$, then we are done by setting $\lambda^{\prime}=0$. Roughly speaking, this corresponds to the intuitive fact that every nonnegative basket is a feasible set of reserves, at some 'large enough' multiple. This observation is taken as an assumption in [FPW23] and [SKM23], but is a direct consequence of the definition of the reachable set. Additionally, since $K$ is closed we have
$$
\left(\mathbf{R}_{+}^{n}, 0\right) \subseteq K
$$
though this construction is less useful than the previous.

Reachable set. We may, of course recover the reachable set from the liquidity cone in a variety of ways. Perhaps the simplest is to note that, for any $\lambda>0$ we have
$$
\begin{equation*}
S=\{R / \lambda \mid(R, \lambda) \in K\} \tag{D.5}
\end{equation*}
$$

This is easy to see as $(R, \lambda)=\lambda(R / \lambda, 1) \in K$, and, since $K$ is a cone, this is if, and only if, $(R / \lambda, 1) \in K$ which is also if, and only if, $R / \lambda \in S$. This will be useful in what follows.

\section*{Canonical trading function}

Given any liquidity cone $K$ for a reachable set $S$, we will define a canonical trading function,
$$
\begin{equation*}
\varphi(R)=\sup \{\lambda \mid(R, \lambda) \in K\} \tag{D.6}
\end{equation*}
$$
setting $\varphi(R)=0$ if the set is empty. (Since $K$ is closed, we may replace the sup with a max if $0 \notin S$, which we assume for the remainder of the section.) In terms of the trading set $S$, we may write this as
$$
\varphi(R)=\sup \{\lambda>0 \mid R / \lambda \in S\}
$$
using the definition of the liquidity cone $K$. If the reachable set $S$ is written using a nondecreasing, quasiconcave, but not necessarily concave, function as in (D.2), then we can 'canonicalize' this trading function by writing
$$
\begin{equation*}
\varphi(R)=\sup \{\lambda>0 \mid \psi(R / \lambda) \geq k\} \tag{D.7}
\end{equation*}
$$

Note that, if $\psi$ is continuous, this is the same as finding the largest positive root over $\lambda$ of $\psi(R / \lambda)=k$. If the function is strictly increasing (as is often the case) then the positive root is unique and it suffices only to find it. Figure D. 3 illustrates this definition for the case of Uniswap.

Computational considerations. It may be the case that the canonical trading function (D.7) has no closed form solution. From the previous, since we know that computing the value of the canonical trading function at some reserves $R$ corresponds to a root-finding problem, we may do this using efficiently by using bisection (as $\psi$ is assumed to be nondecreasing) or, if $\psi$ is differentiable, using Newton's method for finding the positive root. In either case, computing $\varphi(R)$ can be done efficiently in practice. (As a side note: if bisection is used, it suffices to run it only until the bracketing interval is either fully contained in $[0,1)$ or $[1, \infty)$. In the former, the reserves are guaranteed to be infeasible, while in the latter they are guaranteed to be feasible.)

Reachable set. From (D.5) we can recover the set $S$ from this canonical trading function since
$$
S=\left\{R \in \mathbf{R}_{+}^{n} \mid \varphi(R) \geq 1\right\}
$$
(Of course, the set of $R$ such that $\varphi(R)=1$ gives the boundary of $S$.) Additionally, note that if $\varphi(R)>0$, which is always true if $R \in \mathbf{R}_{++}^{n}$ is strictly positive, from positive reachability (D.4), then
$$
\begin{equation*}
\frac{R}{\varphi(R)} \in S \tag{D.8}
\end{equation*}
$$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-124.jpg?height=609&width=727&top_left_y=247&top_left_x=691}
\captionsetup{labelformat=empty}
\caption{Figure D.3: Another interpretation of the canonical trading function (D.6): we scale along the line segment defined by $\left(R_{1}, R_{2}\right)$ to $(0,0)$, with scale factor $1 / \lambda$, increasing $\lambda$ until we hit the reachable set boundary.}
\end{figure}

Concavity. This function is concave, as it is the partial maximization of the concave function
$$
f(R, \lambda)=\lambda-I(R, \lambda)
$$
over $\lambda$, where $I$ is the indicator function of the (convex) set $K$, defined $I(R, \lambda)=0$ if $(R, \lambda) \in K$ and $+\infty$ otherwise.

Homogeneity. The trading function $\varphi$ is homogeneous for $\alpha>0$ since
$$
\varphi(\alpha R)=\sup \{\lambda \mid(\alpha R, \lambda) \in K\}
$$

Since $K$ is a cone, then $(\alpha R, \lambda) \in K$ if, and only if, $(R, \lambda / \alpha) \in K$. Setting $\bar{\lambda}=\lambda / \alpha$, then we have
$$
\varphi(\alpha R)=\sup \{\alpha \bar{\lambda} \mid(R, \bar{\lambda}) \in K\}=\alpha \varphi(R)
$$

For $\alpha=0$ the result follows since $0 \in K$.

Monotonicity. The trading function is nondecreasing from the 'almost upward closed' property mentioned previously. For the remainder of the paper, we will call a function that is concave, homogeneous, and nondecreasing a consistent function.

Marginal prices. Given $R$ with $\varphi(R)=1$, i.e., the starting reserves are 'reasonable' and $\varphi$ differentiable at $R$, then, from concavity,
$$
\varphi(R+\Delta) \leq \varphi(R)+\nabla \varphi(R)^{T} \Delta
$$
(We may replace the gradient with a supergradient for a more general condition.) If the trade $\Delta$ is feasible in that $\varphi(R+\Delta) \geq \varphi(R)$ then
$$
\nabla \varphi(R)^{T} \Delta \geq 0
$$

Note that this means that $\nabla \varphi(R)$ is a supporting hyperplane of $S$ at $R$ if $\varphi(R)=1$. If the trader is trading some amount of asset $i$ for asset $j$, i.e., $\Delta_{i}>0$ and $\Delta_{j}<0$ with all other entries zero, we have
$$
(\nabla \varphi(R))_{j} \Delta_{j} \leq-(\nabla \varphi(R))_{i} \Delta_{i}
$$

Rewriting further gives,
$$
\Delta_{j} \leq \frac{(\nabla \varphi(R))_{i}}{(\nabla \varphi(R))_{j}}\left(-\Delta_{i}\right)
$$
where equality can be achieved in the limit as the trade becomes small. We can therefore interpret the quantity $(\nabla \varphi(R))_{i} /(\nabla \varphi(R))_{j}$ as the marginal price of token $j$ with respect to token $i$, and we can interpret the vector $\nabla \varphi(R)$ as a vector of prices, up to a scaling factor determined by the numéraire.

Discussion. This shows that a number of results which hold 'only' for homogeneous trading functions, such as those of [FPW23; AEC21], are fully general and hold for all CFMMs. Indeed, we do not need to assume homogeneity at all as it may always be derived for a trading set satisfying some basic conditions given above. Additionally, the direct connection to constant function market makers comes from the fact that any trader may change the reserves to some $R^{\prime} \in \mathbf{R}_{+}$so long as
$$
\varphi\left(R^{\prime}\right) \geq \varphi(R)=1
$$
where we assume that $\varphi(R)=1$ is a 'starting condition' on the level set. Of course, no trader would ever take $\varphi\left(R^{\prime}\right)>1$, since otherwise there exists some dominating trade $\tilde{R}^{\prime} \leq R^{\prime}$ with at least one inequality holding strictly; i.e., the trader would tender less (or get more) of at least one token and still have a feasible trade. So, in general, we have that, for any 'reasonable' action,
$$
\begin{equation*}
\varphi\left(R^{\prime}\right)=\varphi(R) \tag{D.9}
\end{equation*}
$$
where $R^{\prime}$ is the new set of reserves, after a trade has been made, and $R$ is the original set of reserves. Equation (D.9) is the defining equation for path-independent constant function market makers, explaining both their name and the direct connection to the reachable set defined here. (See $[\mathrm{Ang}+22 \mathrm{c}]$ for more.)

\section*{Uniqueness of canonical trading function}

We call this trading function canonical since it is unique up to a scaling constant. In fact, this function is unique if the function is scaled such that the reachable set corresponds to its 1 -superlevel set.

Proof. To see this, let $\varphi$ and $\tilde{\varphi}$ be two trading functions that are consistent and yield the same reachable set $S$; i.e.,
$$
S=\left\{R \in \mathbf{R}_{+}^{n} \mid \varphi(R) \geq \alpha\right\}=\left\{R \in \mathbf{R}_{+}^{n} \mid \tilde{\varphi}(R) \geq \beta\right\}
$$
where $\alpha, \beta>0$. (If $\alpha=0$ then, since $\varphi$ is homogeneous and nondecreasing, we have that $\varphi(R) \geq 0$, which would imply that its reachable set is all of $\mathbf{R}_{+}^{n}$, and similarly for $\tilde{\varphi}$.) This is the same as
$$
\{R \mid \varphi(R) / \alpha \geq 1\}=\{R \mid \tilde{\varphi}(R) / \beta \geq 1\}
$$
so we will overload notation by writing $\varphi$ for $\varphi / \alpha$ and $\tilde{\varphi}$ for $\tilde{\varphi} / \beta$, with the understanding that these differ by a proportionality constant. Now, we will show that $\varphi=\tilde{\varphi}$. To see this, start with the case that $R$ satisfies $\varphi(R)>0$ and $\tilde{\varphi}(R)>0$, then
$$
\varphi\left(\frac{R}{\varphi(R)}\right)=1
$$
so $R / \varphi(R) \in S$ and we then have, by definition of $\tilde{\varphi}$,
$$
\frac{\tilde{\varphi}(R)}{\varphi(R)}=\tilde{\varphi}\left(\frac{R}{\varphi(R)}\right) \geq 1
$$

Repeating the steps above with $\varphi$ and $\tilde{\varphi}$ swapped yields
$$
\varphi(R)=\tilde{\varphi}(R)
$$
when $\varphi(R)>0$ and $\tilde{\varphi}(R)>0$. Now, if $\varphi(R)=0$ then
$$
\varphi(t R)=t \varphi(R)=0
$$
so $t R \notin S$ for any $t>0$. This means that
$$
t \tilde{\varphi}(R)=\tilde{\varphi}(t R)<1
$$
again by definition of $\tilde{\varphi}$, or, that $\tilde{\varphi}(R)<1 / t$ for any $t>0$, so $\tilde{\varphi}(R)=0$. Repeating these steps where $\varphi$ is swapped with $\tilde{\varphi}$ implies that $\varphi(R)=0$ only when $\tilde{\varphi}(R)=0$. This gives the final result that $\varphi=\tilde{\varphi}$, or that the canonical function is unique up to scaling constants.

\section*{Examples}

In this subsection, we show the canonical trading function for Uniswap and Uniswap v3. We also derive the canonical trading function for Curve [Ego19] in appendix D.5.2. (That Curve is homogeneous in its 'standard' form seems to have been first noted in [FP23, §3.4].)

Uniswap. Starting with the usual example of Uniswap, we have that
$$
S=\left\{R \in \mathbf{R}_{+}^{2} \mid R_{1} R_{2} \geq k\right\}
$$

The liquidity cone for Uniswap is given by
$$
\begin{equation*}
K=\left\{(R, \lambda) \in \mathbf{R}^{3} \mid R_{1} R_{2} / \lambda^{2} \geq k \in S, \lambda>0\right\} \tag{D.10}
\end{equation*}
$$
so, the canonical trading function (D.7) can be written
$$
\varphi(R)=\sup \left\{\lambda>0 \mid R_{1} R_{2} / \lambda^{2} \geq k\right\}
$$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-127.jpg?height=649&width=1626&top_left_y=232&top_left_x=257}
\captionsetup{labelformat=empty}
\caption{Figure D.4: Left: the liquidity cone for Uniswap, with the level set defined by the trading function $\varphi(R)=\sqrt{R_{1} R_{2}}=1$ shown. Right: each $\lambda$-level set of the surface looks like the boundary of the set of reachable reserves (see figure D.1). The trading function $\varphi$ is highlighted.}
\end{figure}
when $R \in \mathbf{R}_{++}^{2}$ (and zero otherwise). This gives the canonical trading function
$$
\varphi(R)=\sqrt{\frac{R_{1} R_{2}}{k}},
$$
which is evidently concave, nondecreasing, and 1 -homogeneous, with
$$
\varphi(R) \geq 1 \quad \text { if, and only if } \quad R_{1} R_{2} \geq k,
$$
as required. The liquidity cone and canonical trading function are shown in figure D.4.
Uniswap v3. We can also do the same for Uniswap v3, which has a quasiconcave trading function given by
$$
\psi(R)=\left(R_{1}+\alpha\right)\left(R_{2}+\beta\right) .
$$

Since $\psi$ is strictly increasing in the positive orthant, it suffices only to find the (positive) root of
$$
\left(R_{1} / \lambda+\alpha\right)\left(R_{2} / \lambda+\beta\right)=k,
$$
which is a simple quadratic. The resulting canonical trading function is unfortunately more complicated:
$$
\begin{equation*}
\varphi(R)=\frac{1}{2}\left(\frac{\beta R_{1}+\alpha R_{2}+\sqrt{\left(\beta R_{1}+\alpha R_{2}\right)^{2}+4(k-\alpha \beta) R_{1} R_{2}}}{k-\alpha \beta}\right) . \tag{D.11}
\end{equation*}
$$

This function is evidently homogeneous and strictly increasing since $k>\alpha \beta$. Concavity is more difficult due to the square root term, but we show it directly in appendix D.5.3. A good exercise is to show that the canonical trading function $\varphi$ in (D.11) has $\varphi(R) \geq 1$, if, and only if, $\left(R_{1}+\alpha\right)\left(R_{2}+\beta\right) \geq k$.

\section*{D.2.4 Dual cone and portfolio value function}

In this section, we will look at an equivalent characterization of the liquidity cone $K$, called the dual cone. The characterizations are equivalent since the liquidity cone $K$ is convex. Indeed, we will show that this dual cone has a very tight relationship with the portfolio value function, and leads to a simple proof of the equivalence of (consistent) portfolio value functions and (canonical) trading functions in that every portfolio value function has a corresponding trading function, and vice versa, which was originally derived in [AEC23].

\section*{Dual cone}

The dual cone of a cone $K \subseteq \mathbf{R}^{n+1}$ is defined as
$$
K^{*}=\left\{(c, \eta) \in \mathbf{R}^{n+1} \mid c^{T} R+\eta \lambda \geq 0, \text { for all }(R, \lambda) \in K\right\}
$$

While this definition holds for any cone $K$, for the remainder of this section, we will be working with the case that $K$ is the liquidity cone of a CFMM with reachable set $S$, as defined the previous subsection.

Intuition. In a very general sense, the dual cone $K^{*}$ is simply another (dual) representation of the original liquidity cone, $K$, in that the dual of $K^{*}$, defined as $\left(K^{*}\right)^{*}=K$, as $K$ is closed and convex. (For more information on conic duality, we refer the reader to appendix D.5.1.) We will use this fact to give a simple proof that the trading function and the portfolio value function (to be introduced later in this section) are two views of the same underlying object.

Basic properties. First, note that $K^{*}$ is always a closed, convex cone as it can be written as the intersection of closed hyperplanes, and, by definition, we have $0 \in K^{*}$. Additionally, we have that
$$
\begin{equation*}
K^{*} \subseteq \mathbf{R}_{+}^{n} \times \mathbf{R} \tag{D.12}
\end{equation*}
$$
since $K \supseteq \mathbf{R}_{++}^{n} \times\{0\}$, from the previous section. (To see this, use the definition of $K^{*}$.) Finally, we may write the dual cone in terms of only the reachable set $S$. We have that $(c, \eta) \in K^{*}$ if, and only if,
$$
c^{T} R+\eta \lambda \geq 0, \text { for all }(R, \lambda) \in K,
$$
by definition. But this latter statement is true if, and only if it is true for all $\lambda>0$, since $K$ is the closure over the set defined in (D.3); i.e., $(c, \eta) \in K^{*}$ if, and only if,
$$
c^{T} R+\eta \lambda \geq 0, \text { for all }(R, \lambda) \in K, \lambda>0 .
$$

Rearranging the inequality gives that $c^{T}(R / \lambda)+\eta \geq 0$, and note that, by definition of $K$, we have that $(R, \lambda) \in K$ with $\lambda>0$ only when $R / \lambda \in S$. This means that $(c, \eta) \in K^{*}$ if, and only if,
$$
\begin{equation*}
c^{T} \tilde{R}+\eta \geq 0, \text { for all } \tilde{R} \in S \tag{D.13}
\end{equation*}
$$

This particular rewriting of $K^{*}$ will be useful in what follows.

\section*{Portfolio value function}

Much in the same way that we defined the trading function (D.6), we may define the portfolio value function as
$$
\begin{equation*}
V(c)=\sup \left\{-\eta \mid(c, \eta) \in K^{*}\right\} \tag{D.14}
\end{equation*}
$$

This function has the following interpretation: given an external market with prices $c \in \mathbf{R}_{+}^{n}$ (i.e., anyone may trade asset $i$ for asset $j$ at a fixed price $c_{i} / c_{j}$ ) then $V(c)$ corresponds to the total value of reserves after arbitrage has been performed. In particular, $V(c)$ is the optimal value of the problem,
$$
\begin{array}{ll}
\operatorname{minimize} & c^{T} R \\
\text { subject to } & R \in S, \tag{D.15}
\end{array}
$$
with variable $R \in \mathbf{R}^{n}$, where $S$ is the reachable set.
To see this, note that $(c, \eta) \in K^{*}$ if, and only if,
$$
c^{T} R \geq-\eta, \text { for all } R \in S
$$
from the previous characterization of $K^{*}$ given in (D.13). The claim follows by applying the definition of $V$ in (D.14).

Properties. From the optimization problem formulation (D.15), we see that $V$ is clearly nonnegative and nondecreasing since for any $R \in S$, we have that $R \geq 0$. The function $V$ is also concave because it is the partial maximization of the concave function
$$
f(c, \eta)=-\eta-I(c, \eta)
$$
over $\eta$, where $I$ is the indicator function of the convex set $K^{*}$, defined as $I(c, \eta)=0$ if $(c, \eta) \in K^{*}$ and $+\infty$, otherwise. Finally, we see that $V$ is homogeneous since for $\alpha>0$, $V(\alpha c)$ is the optimal value of the problem
$$
\begin{array}{ll}
\text { minimize } & \alpha c^{T} R \\
\text { subject to } & R \in S
\end{array}
$$

Since $\alpha$ is a constant, this value is clearly $\alpha V(c)$.
Consistency. We say a portfolio value function is consistent if it is concave, homogeneous, and nondecreasing, which we know is true for any function $V$ derived from a reachable set $S$. Of course, every consistent portfolio value function also defines a dual cone:
$$
K^{*}=\{(c, \eta) \mid V(c)+\eta \geq 0\}
$$
which can be easily verified to be a convex cone that is contained in $K^{*} \subseteq \mathbf{R}_{+}^{n} \times \mathbf{R}$ using the fact that $V$ is consistent, so we may convert from portfolio value functions to dual cones directly.

Examples. Using (D.10), we can write the dual cone for Uniswap
$$
K^{*}=\left\{(c, \eta) \mid c^{T} R+\eta \lambda \geq 0, \text { for all } R_{1} R_{2} \geq k \lambda^{2}, \lambda>0\right\}
$$

We can simplify this expression via a few observations. First, we must have $c \geq 0$, from (D.12). Second, because $c \geq 0$, if $\eta \geq 0$ then ( $c, \eta$ ) is clearly in $K^{*}$. The interesting case is then when $c \geq 0$ but $\eta<0$. In this case, we must have that
$$
c^{T} R \geq(-\eta) \sqrt{\frac{R_{1} R_{2}}{k}},
$$
since $\lambda$ can take any value between 0 and $\sqrt{R_{1} R_{2} / k}$. Rearranging, we have that
$$
c_{1} x+c_{2} x^{-1} \geq(-\eta) / \sqrt{k}
$$
where $x=\sqrt{R_{1} / R_{2}}$. Minimizing the left hand side over $x>0$ means that this inequality is true if, and only if,
$$
2 \sqrt{c_{1} c_{2}} \geq-\eta / \sqrt{k}
$$
so the dual cone for Uniswap is
$$
K^{*}=\left\{(c, \eta) \in \mathbf{R}_{+}^{2} \times \mathbf{R} \mid 2 \sqrt{k c_{1} c_{2}}+\eta \geq 0\right\}
$$

The portfolio value function can almost be read off from the definition:
$$
\begin{equation*}
V(c)=2 \sqrt{k c_{1} c_{2}}, \tag{D.16}
\end{equation*}
$$
which is evidently concave, homogeneous, and nondecreasing.
As a more complicated example, we'll derive the portfolio value function for a Uniswap v3 'tick'. In this case, it's easier to work directly from the optimization problem (D.15). For convenience, let $c=(p, 1)$ and note that we can recover the general case using the homogeneity of $V$, as $V(\tilde{c})=\tilde{c}_{2} V\left(\tilde{c}_{1} / \tilde{c}_{2}, 1\right)$. Then,
$$
V(p, 1)=\inf _{R \geq 0}\left\{p R_{1}+R_{2} \mid\left(R_{1}+\alpha\right)\left(R_{2}+\beta\right) \geq k\right\} .
$$

Any profit maximizing trader will ensure that the inequality holds with equality (i.e., the solution is at the boundary of the set $S$ ). After substitution, we have a simple convex function that is minimized either at a point $R>0$ with gradient zero or at the boundary. We can conclude that
$$
V(p, 1)= \begin{cases}p k / \alpha-\beta & p<\beta^{2} / k \\ 2 \sqrt{p k}-(\alpha+\beta) & \beta^{2} / k \leq p \leq k / \alpha^{2} \\ p k / \beta-\alpha & k / \alpha^{2}<p\end{cases}
$$

Note the similarity of this expression to the previous (D.16) when the price is within a particular range. This range corresponds exactly to the 'tick' interval in Uniswap v3 [Ada+21b].

\section*{Replicating market makers}

In this subsection we show how to convert directly between the portfolio value function and a canonical trading function (and vice versa). This shows that, indeed, every canonical trading function has an equivalent consistent portfolio value function, and, in a sense, each of these functions is a different 'view' of the same underlying object.

Trading function to portfolio value. Assuming $\varphi$ is a canonical trading function, as defined in (D.6), then we may write the portfolio value function as
$$
V(c)=\inf _{R>0}\left(\frac{c^{T} R}{\varphi(R)}\right)
$$

To see this, note that the definition of $K^{*}$ is that $(c, \eta) \in K^{*}$ when
$$
c^{T} R+\eta \lambda \geq 0, \text { for all }(R, \lambda) \in K .
$$

Minimizing the left hand side over $\lambda>0$ gives that $(c, \eta) \in K^{*}$ only when
$$
c^{T} R+\eta \varphi(R) \geq 0, \text { for all } R \in \mathbf{R}_{+}^{n}
$$
by definition of $\varphi(R)$. Using a basic limiting argument, we may replace $R \in \mathbf{R}_{+}^{n}$ with $R \in \mathbf{R}_{++}^{n}$, which implies that $\varphi(R)>0$ by positive reachability (D.4), so we have that
$$
-\eta \leq \frac{c^{T} R}{\varphi(R)}, \text { for all } R \in \mathbf{R}_{++}^{n}
$$
or, equivalently, that $(c, \eta) \in K^{*}$ if, and only if,
$$
-\eta \leq \inf _{R>0}\left(\frac{c^{T} R}{\varphi(R)}\right)
$$

Applying the definition of $V(c)$, given in (D.14), gives the final result.
Trading function from portfolio value. It is also possible to show that we can recover a canonical trading function from a given portfolio value function. To see this, note that
$$
\begin{equation*}
\varphi(R)=\inf _{c>0}\left(\frac{c^{T} R}{V(c)}\right) \tag{D.17}
\end{equation*}
$$
is a concave (as it is the minimization of a family of affine functions over $R$ ), homogeneous, and nondecreasing trading function. We can easily show that, if $K^{*}$ corresponds to the dual of a liquidity cone $K$, and $V$ is the corresponding portfolio value function, then $\varphi(R)$ corresponds to its canonical trading function.

From a nearly-identical argument to the previous, replacing the definition of $\varphi$ with that of $V$, we have that $(\tilde{R}, \tilde{\lambda}) \in\left(K^{*}\right)^{*}$ if, and only if,
$$
\tilde{\lambda} \leq \inf _{c>0}\left(\frac{c^{T} \tilde{R}}{V(c)}\right)
$$

Since $K$ is a liquidity cone (by assumption) it is therefore closed and convex, so we have that $\left(K^{*}\right)^{*}=K$; cf., appendix D.5.1. Finally, maximizing over $\tilde{\lambda}$ and using the definition of $\varphi$ given in (D.6):
$$
\varphi(\tilde{R})=\inf _{c>0}\left(\frac{c^{T} \tilde{R}}{V(c)}\right)
$$
where $\varphi$ is the canonical trading function for $K$.

Example. To complete the cycle, we convert the portfolio value function of Uniswap back to its canonical trading function. From above,
$$
\begin{aligned}
\varphi(R)=\inf _{c>0}\left(\frac{c^{T} R}{V(c)}\right) & =\frac{1}{2 \sqrt{k}} \inf _{c>0}\left(\sqrt{\frac{c_{1}}{c_{2}}} R_{1}+\sqrt{\frac{c_{2}}{c_{1}}} R_{2}\right) \\
& =\frac{1}{2 \sqrt{k}} \inf _{x>0}\left(x R_{1}+x^{-1} R_{2}\right) \\
& =\sqrt{\frac{R_{1} R_{2}}{k}},
\end{aligned}
$$
where we recognized $x R_{1}+x^{-1} R_{2}$ as a convex function and minimized by simply applying the first order optimality conditions.

Interpretation. There is a nice interpretation for equation (D.17) which is that the quotient $c^{T} R / V(c)$ denotes the leverage or the 'lambda' of the portfolio $R \in \mathbf{R}_{+}^{n}$ at price $c$, where $V(c)$ denotes the true value of the CFMM holdings at this price. We may then view the trading function $\varphi(R)$ as the lowest possible leverage over all possible prices. The inequality $\varphi(R) \geq 1$, which defines the reachable set, says that the leverage must be at least 1 in order for the reserves to lie in the set.

Connection to RMMs. There is a connection to the original result of [AEC23] by noting that the trading function presented there is defined, using the portfolio value function $V$, as
$$
\varphi^{0}(R)=\inf _{c>0}\left(c^{T} R-V(c)\right)=-I_{\left(K^{*}\right)^{*}}(R, 1)
$$
where $I_{\left(K^{*}\right)^{*}}$ is the indicator function for the dual cone of the dual cone, $\left(K^{*}\right)^{*}$. Since $\left(K^{*}\right)^{*}=K$ then $\varphi^{0}(R) \geq 0$ if, and only if, $(R, 1) \in K$, which happens if, and only if, $R \in S$, as required.

Discussion. From the above, we have that every consistent portfolio value leads to a canonical trading function. This method gives a general procedure for going from one to the other. Additionally, since we know $\left(\left(K^{*}\right)^{*}\right)^{*}=K^{*}$, then we know that, starting from any consistent portfolio value function $V$, converting it to a trading function $\varphi$, and then converting back results in the same $V$ we started with, which shows that the mapping is indeed invertible.

\section*{Composition rules}

We will denote $S_{V} \subseteq \mathbf{R}_{+}^{n}$ as the reachable set corresponding to the portfolio value function $V$. (We will see how to construct this explicitly in what follows.)

Composition rules for portfolio value. Given consistent portfolio value functions, there are a number of possible ways these could be 'combined'. The first is by scaling: if $V$ is consistent, then certainly $\alpha V$ is consistent for $\alpha \geq 0$. If both $V$ and $V^{\prime}$ are consistent, then $V+V^{\prime}$ is consistent, and, finally if $A$ is a nonnegative orthogonal matrix, and $V$ is consistent, then $V \circ A^{T}$ is consistent. We will show that these operations correspond to natural operations over the reachable sets corresponding to the portfolio value functions.

Recovering the reachable set. We may recover the reachable set from a given portfolio value function since we know that, for given portfolio value function, its liquidity cone, which we will denote $K_{V} \subseteq \mathbf{R}^{n+1}$ is
$$
K_{V}=\left\{(R, \lambda) \in \mathbf{R}^{n+1} \mid c^{T} R-\lambda V(c) \geq 0 \text { for all } c \geq 0\right\} .
$$

Clearly, this cone is convex, closed, and satisfies $K_{V} \subseteq \mathbf{R}_{+}^{n+1}$ from the fact that $V$ is consistent. Additionally, since we may define a reachable set from a liquidity cone as $S_{V}=\left\{R \mid(R, 1) \in K_{V}\right\}$, then this is the same as saying
$$
\begin{equation*}
S_{V}=\left\{R \mid c^{T} R \geq V(c) \text { for all } c \geq 0\right\} . \tag{D.18}
\end{equation*}
$$

It remains to be verified that $S_{V}$ is a valid reachable set, but this follows from the properties of $K_{V}$ outlined above. (Another way to see this is to note that $c^{T} R \geq V(c)$ if, and only if, $c^{T} R / V(c) \geq 1$ for all $c>0$, i.e., $\varphi(R) \geq 1$ using (D.17). Since $\varphi$ is a canonical trading function, then $S_{V}$ is a reachable set.)

Scaling. It is not hard to see that
$$
S_{\alpha V}=\alpha S_{V}
$$
for any $\alpha \geq 0$ by using the definition of $S_{V}$ and the fact that $V$ is homogeneous.
Addition. Similarly, addition over the portfolio value functions 'commutes' over the reachable sets; i.e.,
$$
S_{V+V^{\prime}}=S_{V}+S_{V^{\prime}}
$$

The direction $S_{V+V^{\prime}} \subseteq S_{V}+S_{V^{\prime}}$ is easy to show by definition. On the other hand, since $S_{V+V^{\prime}}$ is a closed convex set, if $R \notin S_{V+V^{\prime}}$ then there exists a strictly separating hyperplane $c \in \mathbf{R}_{+}^{n}$ with
$$
c^{T} R<c^{T} \tilde{R}, \text { for all } \tilde{R} \in S_{V+V^{\prime}}
$$

Taking the infimum of the right hand side and using (D.15) gives
$$
c^{T} R<\left(V+V^{\prime}\right)(c) \leq c^{T} \tilde{R}+c^{T} \tilde{R}^{\prime} \text { for all } \tilde{R} \in S_{V}, \tilde{R}^{\prime} \in S_{V^{\prime}}
$$
which means that $R \notin S_{V}+S_{V^{\prime}}$. Here, the last inequality follows from the fact that $V(c) \leq c^{T} \tilde{R}$ for all $\tilde{R} \in S_{V}$ and similarly for $S_{V^{\prime}}$. Putting both statements together gives that $S_{V+V^{\prime}}=S_{V}+S_{V^{\prime}}$.

Nonnegative projection. Given some nonnegative matrix $A \in \mathbf{R}_{+}^{m \times n}$ that is also an orthogonal matrix, i.e., $A^{T} A=I$, then
$$
S_{V \circ A^{T}}=A S_{V}+\mathbf{R}_{+}^{m}
$$
where $\left(V \circ A^{T}\right)(c)=V\left(A^{T} c\right)$. This follows nearly immediately from the definition of $A$ and (D.18).

Intersection. There is a natural question then, as to what the intersection of reachable sets corresponds to. Clearly, we have
$$
S_{V} \cap S_{V^{\prime}}=\left\{R \mid c^{T} R \geq V(c) \text { and } c^{T} R \geq V^{\prime}(c) \text { for all } c \geq 0\right\} .
$$

Of course, this implies that
$$
S_{V} \cap S_{V^{\prime}}=S_{\max \left\{V, V^{\prime}\right\}}
$$
where the max is taken pointwise. Note that this does not correspond to a natural operation on the portfolio value functions as the pointwise maximum of two concave functions is not necessarily concave. (Take, for example, $V(p, 1)=\sqrt{p}$ and $V^{\prime}(p, 1)=p$.) Let, on the other hand, $\tilde{V}$ be the (pointwise) smallest consistent function with $\tilde{V} \geq V$ and $\tilde{V} \geq V^{\prime}$, then indeed we have
$$
S_{V} \cap S_{V^{\prime}}=S_{\tilde{V}}
$$
(See appendix D.5.4 for a proof.)
Discussion. This also gives another proof of the composition rules presented for the reachable sets since we may always recover a consistent portfolio value from any reachable set. In this sense, we may think of the portfolio value function and the reachable set as two objects with a 'natural homomorphism' under which scaling, addition, nonnegative projection, and 'intersection' are all preserved.

\section*{D.2.5 Connection to prediction markets}

Prediction markets are a type of market which attempts to elicit the beliefs of players about the probability that certain events occur. These markets have a rich academic history, dating back to at least the 50s [McC56]; indeed, there have been many geometric approaches to these types of prediction markets that are similar in spirit to those presented here [Art +99 ; AFK15]. Until the relatively recent paper [FPW23], a connection between such markets and CFMMs was not formalized, except in some very special cases. This section restates and simplifies the results of [FPW23] in this framework. We differ in the notion of 'histories' for path-independent CFMMs which is implicitly included in this framework and discussed later in this paper, as a general result in §D.3.4.

Cost functions. A cost function is defined as a function $C: \mathbf{R}^{n} \rightarrow \mathbf{R} \cup\{+\infty\}$ such that
1. The function $C$ is convex, nondecreasing
2. It is translation invariant, $C(q+\alpha \mathbf{1})=C(q)+\alpha$
3. It is somewhere finite; i.e., there is at least one $q$ for which $C(q)$ is finite

Note that we do not require the function to be differentiable, only subdifferentiable in its domain. This means that there might be many probabilities which are consistent with the market's predictions, but includes differentiability as the special case where there is only one.

Example. One particular, important example is the logarithmic market scoring rule, or LMSR, which has cost function
$$
C(q)=b \log \left(\sum_{i=1}^{n} \exp \left(\frac{q_{i}}{b}\right)\right)
$$
where $b>0$ is some given constant. This function is clearly nondecreasing and finite. It is also convex as it is a $\log$-sum-exp [BV04, §3.1.5] function with affine precomposition. It is not hard to see that this function is also translation invariant using the definition, which means that this function is, indeed, a reasonable cost function.

Mechanics. The mechanics of a prediction market are that any player may buy any of $n$ possible mutually exclusive outcomes. Every player will be paid a dollar for each share of outcome $i$ they hold if outcome $i$ occurs at some future time. All other outcomes will have a value of zero. A player who wishes to buy $\delta \in \mathbf{R}^{n}$ shares and must pay a cost of $C(q+\delta)-C(q)$, where $q$ is the current set of outstanding shares that have been sold to all players. (Negative values of $\delta_{i}$ means that the player is selling back $\delta_{i}$ shares to the market.) The outstanding shares are then updated to $q \leftarrow q+\delta$.

Interpretation. Let's say the prediction market begins with some outstanding shares $q_{0}$, and a player has beliefs $p \in \mathbf{R}_{+}^{n}$ about the probability of each event such that $p_{i}$ corresponds to the probability of the $i$ th event occurring. The player can then maximize her expected profit (under her distribution of beliefs) by solving
$$
\operatorname{maximize} \quad p^{T} q-\left(C\left(q_{0}+q\right)-C\left(q_{0}\right)\right)
$$
with variable $q$. We note that the optimal value of this problem, call it $E(p)$ for expected payoff at probabilities $p$, is tightly related to the Fenchel conjugate of $C$, since
$$
E(p)=C^{*}(p)+C\left(q_{0}\right)-p^{T} q_{0} .
$$

The optimality conditions for this problem are that
$$
p \in \partial C\left(q_{0}+q^{\star}\right)
$$
where $q^{\star}$ is the solution to the optimization problem. This means that, if the market has some outstanding shares given by $q_{0}$ then we may interpret $\partial C\left(q_{0}\right)$ as the set of probabilities consistent with the market's belief about the event.

CFMM to cost function. Given a reachable set $S$, we can construct a cost function:
$$
\begin{equation*}
C(q)=\min \{\alpha \geq 0 \mid \alpha \mathbf{1}-q \in S\} . \tag{D.19}
\end{equation*}
$$
(This was first observed by [FPW23].) We may also define the cost function in terms of the liquidity cone as
$$
C(q)=\max \{\beta \geq 0 \mid(\mathbf{1}-\beta q, \beta) \in K\} .
$$

This function is a cost function since it is evidently translation invariant by definition, and is nondecreasing since $S$ is upward closed. The function is finite at 0 since $S$ is nonempty: if $R \in S$, then $0 \leq C(0) \leq \max _{i} R_{i}$. Additionally, this function is convex as it is the partial minimization of the convex function
$$
f(\alpha, q)=\alpha+I(\alpha, q)
$$
over $\alpha \in \mathbf{R}$, where $I(\alpha, q)=0$ if $\alpha \mathbf{1}-q \in S$ and $\alpha \geq 0$, and is $+\infty$ otherwise. (This set indicator is convex as it is the indicator function for the intersection of convex sets.)

Example. Recall that Uniswap has the trading set
$$
S=\left\{R \in \mathbf{R}_{+}^{2} \mid R_{1} R_{2} \geq k\right\}
$$

Using (D.19), we have the cost function
$$
C(q)=\min \left\{\alpha \geq 0 \mid\left(\alpha-q_{1}\right)\left(\alpha-q_{2}\right) \geq k\right\} .
$$

The cost function is the positive root of the quadratic over $\alpha$, the same as was found in [FPW23]:
$$
C(q)=\frac{q_{1}+q_{2}}{2}+\frac{1}{2} \sqrt{\left(q_{1}-q_{2}\right)^{2}+k}
$$

We can easily verify that this function is finite, translation invariant, and convex (by noting that the square root term can be expressed as the $\ell_{2}$ norm of the vector $\left(q_{1}-q_{2}, \sqrt{k}\right)$ ). The fact that it is nondecreasing can be seen by showing that its gradient is everywhere nonnegative.

Cost function to CFMM. Any cost function $C$ defines a CFMM by defining its reachable set as,
$$
\begin{equation*}
S=\left\{R \in \mathbf{R}_{+}^{n} \mid C(-R) \leq 0\right\} . \tag{D.20}
\end{equation*}
$$

This $S$ is indeed a reachable set as (a) the function $C$ is nondecreasing by assumption, so $S$ is upward closed, (b) it is convex as $C$ is convex, and (c) it is nonempty since $C(q)$ is finite for some $q \in \mathbf{R}^{n}$, so $C(q-C(q) \mathbf{1})=0$ by translation invariance, and therefore $C(q) \mathbf{1}-q \in S$. We may write its canonical trading function using (D.7):
$$
\varphi(R)=\sup \{\lambda>0 \mid C(-R / \lambda) \leq 0\} .
$$

Equivalence. If the cost function $C$ is constructed from a CFMM with reachable set $S$, as in (D.19), then it is not hard to show that (D.20) yields exactly this set $S$. To see this, note that, by definition (D.19), we have that $C(-R) \leq 0$, if, and only if, $\alpha \mathbf{1}+R \in S$ for all $\alpha \geq 0$; letting $\alpha=0$ gives that $R \in S$. On the other hand, if $R \in S$ then, $R+\alpha \mathbf{1} \in S$ for every $\alpha \geq 0$, by upward closedness, so $C(-R) \leq 0$ and the sets are equivalent.

Example. The logarithmic market scoring rule (LMSR) has the cost function
$$
C(q)=b \log \left(\sum_{i=1}^{n} \exp \left(\frac{q_{i}}{b}\right)\right) .
$$

We may define its trading set, using (D.20), as
$$
S=\left\{R \in \mathbf{R}_{+}^{n} \mid \sum_{i=1}^{n} \exp \left(-R_{i} / b\right) \leq 1\right\} .
$$

The corresponding canonical trading function is
$$
\varphi(R)=\sup \left\{\lambda>0 \mid \sum_{i=1}^{n} \exp \left(-R_{i} / \lambda b\right) \leq 1\right\}
$$

This function has no closed form solution but can be solved numerically as a univariate root-finding problem. Since $C$ is strictly increasing, the positive root is unique and can be found efficiently using the methods discussed in §D.2.3.

\section*{D.2.6 Liquidity provision}

As in $[\mathrm{Ang}+22 \mathrm{c}]$, we discuss liquidity provision in the case where the trading function $\varphi$ is homogeneous. This is, of course, fully general as we may assume that $\varphi$ is a consistent trading function.

Liquidity providers. An agent, called a liquidity provider can add or remove assets from the CFMM's reserves $R$. When an agent adds liquidity, she adds a basket $\Psi \in \mathbf{R}_{+}^{n}$ to the reserves, resulting in the updated reserves $R^{+}=R+\Psi$. When an agent removes liquidity, she removes a basket $\Psi \in \mathbf{R}_{+}^{n}$ with $\Psi \leq R$ from the reserves, resulting in the updated reserves $R^{+}=R-\Psi$. When adding (or removing) to reserves, the agent receives (tenders) an IOU. This IOU gives the agent a pro-rata share of the reserves based on the amount of value the agent added and the total amount of value in the pool. We describe the exact mechanism for liquidity addition (and removal) below.

Liquidity change condition. The main condition for adding and removing liquidity is that the asset prices must not change after the removal, or addition, of liquidity. More specifically, we must have that the prices, as given in §D.2.3, at the new reserves, $R^{+} \in \mathbf{R}_{+}^{n}$,
must be equal, up to a scalar multiple, to those at the original reserves, $R \in \mathbf{R}_{+}^{n}$. Written out, this gives the following condition:
$$
\nabla \varphi\left(R^{+}\right)=\alpha \nabla \varphi(R)
$$
where $\varphi$ is the canonical trading function and $\alpha>0$ is some positive constant. Since $\varphi$ is homogeneous, we have that, for any $\alpha>0, \nabla \varphi(\alpha R)=\alpha \nabla \varphi(R)$. We conclude that $\Psi=\nu R$ for $\nu>0$ is a valid liquidity change for any $\nu>0$ (where we must have $\nu \leq 1$ for liquidity removal). Note that scaling $R$ to $\alpha R$ corresponds exactly to scaling the reachable set by $\alpha$.

Liquidity provider share weights. The CFMM additionally maintains a table of all liquidity providers and their corresponding share weights, representing the fraction of the reserves that each liquidity provider owns. We denote these weights as $w \in \mathbf{R}_{+}^{N}$, where $N$ is the number of liquidity providers, and enforce that they sum to one, i.e., $\sum_{i=1}^{N} w_{i}=1$. These weights are updated whenever a liquidity provider adds or removes liquidity, or when the number of liquidity providers $N$ changes.

Value of the reserves. Let $V=p^{T} R$ be the value of the CFMM reserves at price $p$. After adding liquidity $\nu R$, the value of the reserves is now
$$
V^{+}=p^{T} R^{+}=(1+\nu) p^{T} R=(1+\nu) V
$$

For removing liquidity, we replace $\nu$ with $-\nu$. The fractional change in reserve value is
$$
\left(V^{+}-V\right) / V^{+}=\nu /(1+\nu)
$$

Liquidity provider share update. When liquidity provider $j$ adds or removes liquidity, all the share weights are adjusted pro-rata based on the change of value of the reserves, which is the value of the basket she adds or removes. The fractional change in reserve value is $\nu /(1+\nu)$. Thus, after adding liquidity, the change in share weights is
$$
w_{i}^{+}= \begin{cases}w_{i} /(1+\nu)+\nu /(1+\nu) & i=j \\ w_{i} /(1+\nu) & i \neq j\end{cases}
$$

For removing liquidity, we replace $\nu$ with $-\nu$ and add the constraint that $\nu \leq w_{j}$.
Portfolio value. We note that, since liquidity providers own a pro-rata share with weight $w_{i}$ of the total pool value, we may view each liquidity provider's position as 'independent'. In particular, there is no distinction between many liquidity providers pooling their assets together into a single CFMM versus every liquidity provider having their own CFMM instance and owning all of the assets of their particular instance. (There may be practical differences, however, owing to the fact that users may prefer to trade with a subset of these for a variety of reasons, such as gas fees.)

\section*{D. 3 Single trade}

We consider in this section the general CFMM case, which potentially includes fees and is therefore not necessarily path-independent. (We show the connection to the previous fee-free case later in the section.)

\section*{D.3.1 Trading set}

Much in the same way as the previously-defined reachable set, we will define the trading set $T \subseteq \mathbf{R}^{n}$, which is any set $T$ satisfying the following properties:
1. The set $T$ is closed and convex
2. The zero trade is included, $0 \in T$
3. The set $T$ is downward closed; i.e., if $\Delta \in T$ and $\Delta^{\prime} \leq \Delta$ then $\Delta^{\prime} \in T$

An additional requirement that will be useful in the composition rules presented later, but is not strictly required for most of the statements below is that: there exists $R \in \mathbf{R}_{+}^{n}$ such that
$$
\begin{equation*}
R-T=\{R-\Delta \mid \Delta \in T\} \subseteq \mathbf{R}_{+}^{n} . \tag{D.21}
\end{equation*}
$$

This corresponds to the statement that the CFMM can only tender a finite amount of some asset (in 'usual' CFMMs, this would be the available reserves) which is upper bounded by the quantity $R \geq 0$. One could imagine a mechanism that is allowed to mint unbounded amounts of tokens may violate (D.21).

Set up. In this set up, we have a trader who wishes to trade with the CFMM. This trader can suggest to trade a basket of tokens $\Delta \in \mathbf{R}^{n}$, where positive values denote what the trader receives from the CFMM and negative values denote what the trader tenders to the CFMM. The CFMM accepts this trade (and tenders or receives the stated amounts) only when $\Delta \in T$. (If $\Delta \notin T$, the trader receives and tenders nothing to the CFMM.) The state of the CFMM is then updated based on the accepted trade $\Delta$ (and a rejected trade does not change the state), but, for now, we will only consider the single-trade case and elide discussion of this state update until later. We assume only that the trading set at the current state is known and accessible to the trader.

Interpretation. The conditions imposed on the trading set all have relatively simple interpretations. The convexity of the trading set means that, as users trade more of a token, they receive marginally less (or, at least, no more) than they otherwise would by trading less. The fact that the zero trade is included means that a user is allowed to not trade. Finally, the downward-closedness of the set means that the CFMM will accept more of a given token, all else being equal; i.e., a trader is allowed to 'overpay' for a given trade, and this new trade is still valid. The final optional condition can be interpreted as: a CFMM has a finite amount of assets that it is allowed to tender. While not strictly a requirement, we will need it for a technical condition we present later.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-140.jpg?height=598&width=1641&top_left_y=232&top_left_x=244}
\captionsetup{labelformat=empty}
\caption{Figure D.5: Left: the trading set for Uniswap (without fees) for $R=(1,1)$ (light gray) and $R=(2,2)$ (dark gray). Right: the trading set for Uniswap v3.}
\end{figure}

Example. A basic example of a trading set is Uniswap with fees. In this particular case, the current state of the CFMM is given by some reserves $R \in \mathbf{R}_{+}^{2}$, and the trading set is
$$
T=\left\{\Delta^{+}-\Delta^{-} \mid\left(R_{1}+\gamma \Delta_{1}^{-}-\Delta_{1}^{+}\right)_{+}\left(R_{2}+\gamma \Delta_{2}^{-}-\Delta_{2}^{+}\right)_{+} \geq R_{1} R_{2}, \Delta^{-}, \Delta^{+} \in \mathbf{R}_{+}^{n}\right\},
$$
where $0 \leq \gamma \leq 1$ is the fee parameter and is set by the CFMM at creation time. We show this set in figure D.5. This writing is bit verbose and difficult to parse, but the construction is very similar to the original, given in the fee-free case above. Because of this, it is often nicer to work directly with a functional form, which we describe below.

Quasiconcavity and fees. Given any nondecreasing, quasiconcave function, with nonnegative domain $\psi: \mathbf{R}_{+}^{n} \rightarrow \mathbf{R}$ (much like the previous) we can define a trading set with fee $0 \leq \gamma \leq 1$ and reserves $R \in \mathbf{R}_{+}^{n}$,
$$
\begin{equation*}
T=\left\{\Delta^{+}-\Delta^{-} \mid \psi\left(R+\gamma \Delta^{-}-\Delta^{+}\right) \geq \psi(R), \Delta^{-}, \Delta^{+} \in \mathbf{R}_{+}^{n}\right\} . \tag{D.22}
\end{equation*}
$$
(We will, by convention, let the function $\psi$ take on $-\infty$ for values outside of its domain.) Clearly $0 \in T$ and $T$ is downward closed as $\psi$ is nondecreasing. The set is evidently convex as it is an affine transform of the set
$$
\left\{\left(\Delta^{+}, \Delta^{-}\right) \in \mathbf{R}_{+}^{n} \times \mathbf{R}_{+}^{n} \mid \psi\left(R+\gamma \Delta^{-}-\Delta^{+}\right) \geq \psi(R)\right\},
$$
which is a convex set by the quasiconcavity of $\psi$. Closedness is trickier, but follows from the fact that the valid choices of $\Delta^{+}$are in some compact set, $0 \leq \Delta^{+} \leq R$.

Composition rules. The composition rules are nearly identical in both statement and proof to those of the reachable set. Given trading sets $T, T^{\prime} \subseteq \mathbf{R}_{+}^{n}$
1. Trading sets may be added; i.e., $T+T^{\prime}$ yields a trading set
2. Trading sets may be scaled, so $\alpha T$ is a trading set for any $\alpha>0$
3. Taking the intersection $T \cap T^{\prime}$ preserves the trading set property
4. Applying a nonnegative linear transformation $A \in \mathbf{R}_{+}^{k \times n}$ and adding all dominated trades,
$$
A T-\mathbf{R}_{+}^{k},
$$
preserves the trading set property
These composition rules similarly lead to the notion of an aggregate CFMM mentioned previously, in the single-trade case, which is especially useful in the case of Uni v3 as we will show later. The only technical condition appears when adding trading sets: to ensure that the resulting trading set is closed, it suffices to ensure that all CFMMs can tender only a finite amount of assets, as in condition (D.21).

\section*{D.3.2 Trading cone and dual}

Much in the same way as we have defined the liquidity cone, we define the trading cone as
$$
K=\operatorname{cl}\left\{(\Delta, \lambda) \in \mathbf{R}^{n+1} \mid \Delta / \lambda \in T, \lambda>0\right\} .
$$

This cone plays a similar role to the liquidity cone, except in the single trade case. Indeed, many of the constructions we have shown previously for the liquidity cone will apply in a similar form to the trading cone. (We have overloaded notation as we will make no further reference to the liquidity cone.)

Trading function. From a nearly identical argument to the previous we may define a homogeneous, nondecreasing, but convex (instead of concave) trading function
$$
\varphi(\Delta)=\min \{\lambda \geq 0 \mid(\Delta, \lambda) \in K\}
$$
or, equivalently,
$$
\begin{equation*}
\varphi(\Delta)=\inf \{\lambda>0 \mid \Delta / \lambda \in T\} . \tag{D.23}
\end{equation*}
$$
such that
$$
T=\left\{\Delta \in \mathbf{R}^{n} \mid \varphi(\Delta) \leq 1\right\} .
$$
(Here, we define the min and inf of an empty set to be $+\infty$ for convenience.) The difference in sign from the definition in the path independent case in §D.2.3 comes from the fact that the set $T$ is downward (rather than upward) closed, since we are taking the perspective of the trader, rather than the CFMM or its liquidity providers. In this case, the function $\varphi$ is similarly canonical and rational traders will always tender trades $\Delta$ such that
$$
\varphi(\Delta)=1
$$
hence, again, the name 'constant function market maker.'

Example. Perhaps the simplest example of this type of function is, unsurprisingly, Uniswap. Using the quasiconcave function definition (D.22) of the trading set, we have, for $R \in \mathbf{R}_{+}^{2}$,
$$
\psi\left(R_{1}, R_{2}\right)=R_{1} R_{2}
$$
with some fee $0 \leq \gamma \leq 1$. For a given proposed trade, $\Delta \in \mathbf{R}^{n}$, we can decompose $\Delta$ into its positive and negative parts $\Delta=\Delta^{+}-\Delta^{-}$with $\Delta^{-}, \Delta^{+} \geq 0$ and disjoint support $\Delta_{i}^{-} \Delta_{i}^{+}=0$ for each $i=1,2$. Using the definition of the trading function, we look for the smallest $\lambda \geq 0$ such that
$$
\left(R_{1}+\gamma \frac{\Delta_{1}^{-}}{\lambda}-\frac{\Delta_{1}^{+}}{\lambda}\right)\left(R_{2}+\gamma \frac{\Delta_{2}^{-}}{\lambda}-\frac{\Delta_{2}^{+}}{\lambda}\right) \geq R_{1} R_{2} .
$$

With some basic rearrangements, we find
$$
\varphi(\Delta)=\frac{\left(\Delta_{1}^{+}-\gamma \Delta_{1}^{-}\right)\left(\Delta_{2}^{+}-\gamma \Delta_{2}^{-}\right)}{R_{1}\left(\gamma \Delta_{2}^{-}-\Delta_{2}^{+}\right)+R_{2}\left(\gamma \Delta_{1}^{-}-\Delta_{1}^{+}\right)}
$$

This trading function is homogeneous since the numerator is a homogeneous quadratic while the denominator is homogeneous. We can similarly verify that, as expected from the previous discussion, it is convex and nondecreasing by writing it in the following form:
$$
\varphi(\Delta)=\frac{1}{-\left(R_{1} /\left(\gamma \Delta_{1}^{-}-\Delta_{1}^{+}\right)+R_{2} /\left(\gamma \Delta_{2}^{-}-\Delta_{2}^{+}\right)\right)}
$$

Since the denominator is nonnegative, concave, and nonincreasing (in $\Delta^{+}$), then $\varphi$ must be nonnegative, convex, and nondecreasing (in $\Delta^{+}$). Since $\Delta=\Delta^{+}-\Delta^{-}$, directly verifying fact that $\varphi$ is nondecreasing and convex in $\Delta$ requires one more step, which we leave to the reader as a useful exercise. (Of course, we know that both of these already follow from the construction of the trading function in (D.23) and the fact that $\psi$ is quasiconcave and nondecreasing, as is the case for all such trading functions.)

Bounded liquidity. in a similar way to the previous section, we know that, since $0 \in T$, then
$$
\left(-\mathbf{R}_{+}^{n}, 0\right) \subseteq K
$$

We say the trading set has bounded liquidity in asset $i$ if the supremum
$$
\sup \left\{\Delta_{i} \mid \Delta \in T\right\}=\Delta_{i}^{\star}
$$
is achieved at some $\Delta^{\star} \in T$. This has the interpretation that there is a finite basket of assets such that we receive all possible amount of asset $i$ from the CFMM. We say a trading set has bounded liquidity if it has bounded liquidity for each asset $i=1, \ldots, n$. Examples of bounded liquidity CFMMs include Uniswap v3 (see figure D.5) and those with linear trading functions. These bounded liquidity CFMMs are useful since arbitrage can be easily computed in many important practical cases; see $[\mathrm{Dia}+23, § 3]$ for more.

Arbitrage cone. In a similar way to the previous, we will define the dual cone for the trading cone $K \subseteq \mathbf{R}^{n+1}$ as
$$
K^{*}=\left\{(c, \eta) \mid c^{T} \Delta+\eta \lambda \geq 0, \text { for all }(\Delta, \lambda) \in K\right\}
$$

By downward closedness and the fact that $0 \in T$, it is not hard to show that $K^{*} \subseteq\left(-\mathbf{R}_{+}\right)^{n} \times \mathbf{R}_{+}$. Minimizing over the left hand side of the inequality gives another definition, based on the trading function:
$$
K^{*}=\left\{(c, \eta) \mid c^{T} \Delta+\eta \varphi(\Delta) \geq 0, \text { for all } \Delta \in \mathbf{R}^{n}\right\}
$$

Some care has to be taken when interpreting this expression if $\varphi(\Delta)=\infty$ when $\eta=0$, based on the original definition of $K$, but this is an informative exercise for the reader.

Relation to arbitrage. Much like the portfolio value function, we write the arbitrage function, $\operatorname{arb}: \mathbf{R}^{n} \rightarrow \mathbf{R}$, for the trading set $T$ as
$$
\begin{equation*}
\operatorname{arb}(c)=\sup \left\{c^{T} \Delta \mid \Delta \in T\right\} . \tag{D.24}
\end{equation*}
$$

Note that if $c_{i}<0$ for any $i$ then $\mathbf{a r b}(c)=+\infty$ by the downward-closedness of $T$, so we may generally assume that $c \geq 0$. This function has the following interpretation: if there is an external market with prices $c \in \mathbf{R}_{+}^{n}$, this is the maximum profit that an arbitrageur could derive by trading between the external market and the CFMM. This function is convex (as it is the supremum of a family of functions that are affine in $c$ ), nondecreasing over $c \geq 0$, and homogeneous. We may write this function in terms of the dual cone as
$$
\operatorname{arb}(c)=\inf \left\{\eta \mid(-c, \eta) \in K^{*}\right\}
$$
by a nearly-identical argument to that of the portfolio value function in §D.2.4. This function will be very useful in the routing problem that follows. Additionally, from a very similar argument to §D.2.4, the arbitrage function and the trading function are equivalent representations in that we may derive one from the other by setting
$$
\varphi(\Delta)=\sup _{\operatorname{arb}(c)>0}\left(\frac{c^{T} \Delta}{\operatorname{arb}(c)}\right)
$$
and
$$
\operatorname{arb}(c)=\sup _{\varphi(\Delta)>0}\left(\frac{c^{T} \Delta}{\varphi(\Delta)}\right)
$$

From before, note the suprema in this equation, versus the infima in the previous. For examples of such arbitrage functions for some common constant function market makers, see $[\mathrm{Dia}+23$, app. A].

Marginal prices. We can view the supporting hyperplanes of $T$ at some $\Delta$ as the set of marginal prices at trade $\Delta$. We write this set as
$$
\begin{equation*}
C(\Delta)=\bigcap_{\Delta^{\prime} \in T}\left\{\nu \in \mathbf{R}^{n} \mid \nu^{T}\left(\Delta^{\prime}-\Delta\right) \leq 0\right\} . \tag{D.25}
\end{equation*}
$$

Note that this set is a closed convex cone as it is the intersection of closed convex cones and is always nonempty as $0 \in C(\Delta)$. We can write the cone $C(\Delta)$ using the trading function as
$$
\begin{equation*}
C(\Delta)=\bigcup_{\lambda \geq 0} \lambda \partial(\varphi(\Delta)) \tag{D.26}
\end{equation*}
$$
whenever $\varphi(\Delta)=1$ and the subdifferential is defined. As we will soon see, the cone $C(0)$ will be called the no-trade cone. This is a generalization of the no-trade interval [Dia+23] in the case where $n \geq 2$. We show this cone for Uniswap in figure D.6.

The proof of the equivalence (D.26) can be shown in two steps, one for the forward inclusion, and one for the reverse. From the statement, we have $\Delta$ with $\varphi(\Delta)=1$. Now let $g \in \partial \varphi(\Delta)$, then
$$
\varphi(\Delta)+g^{T}\left(\Delta^{\prime}-\Delta\right) \leq \varphi\left(\Delta^{\prime}\right)
$$
by definition of the subgradient $g$. Letting $\Delta^{\prime} \in T$ means that $\varphi\left(\Delta^{\prime}\right) \leq 1$ by definition and $\varphi(\Delta)=1$ by the previous, so
$$
g^{T}\left(\Delta^{\prime}-\Delta\right) \leq \varphi\left(\Delta^{\prime}\right)-1 \leq 0
$$
for every $\Delta^{\prime} \in T$, which means that $g \in C(\Delta)$. Multiplying both sides of this inequality by $\lambda \geq 0$ then means that $\lambda g \in C(\Delta)$, or that
$$
\bigcup_{\lambda \geq 0} \lambda \partial(\varphi(\Delta)) \subseteq C(\Delta)
$$

For the other direction, using the definition of $C$ in (D.25), we can see that $g \in C(\Delta)$ if, and only if, $\Delta$ is a maximizer of the following optimization problem:
$$
\begin{array}{ll}
\operatorname{maximize} & g^{T} \Delta^{\prime} \\
\text { subject to } & \varphi\left(\Delta^{\prime}\right) \leq 1
\end{array}
$$
with variable $\Delta^{\prime} \in \mathbf{R}^{n}$. Using the optimality conditions of this problem, we know that $\Delta$ is a maximizer if, and only if, there exists some $\lambda \geq 0$ such that
$$
0 \in-g+\lambda \partial \varphi(\Delta)
$$
or, equivalently, if, and only if, $g \in \lambda \partial \varphi(\Delta)$ for some $\lambda \geq 0$, which shows the reverse inclusion. It also shows that
$$
C(\Delta) \subseteq \mathbf{R}_{+}^{n}
$$
since $\varphi$ is nondecreasing, so it subgradients must be nonnegative. The optimality conditions are necessary and sufficient by Slater's condition [BV04, §5.2.3], since $\varphi(\mathbf{- 1})=0<1$ and $-\mathbf{1}$ is in the interior of the domain of $\varphi$.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-145.jpg?height=577&width=695&top_left_y=242&top_left_x=713}
\captionsetup{labelformat=empty}
\caption{Figure D.6: The trading set for Uniswap with fees (notice that the set is kinked at 0 ) and the corresponding no-trade cone.}
\end{figure}

Marginal price composition. Given $\Delta_{i} \in T_{i}$ for $i=1, \ldots, m$, we have that
$$
\begin{equation*}
C\left(\sum_{i=1}^{m} \Delta_{i}\right)=\bigcap_{i=1}^{m} C_{i}\left(\Delta_{i}\right), \tag{D.27}
\end{equation*}
$$
where $C_{i}$ is the cone of marginal prices for CFMM $i$ while $C$ is the cone of marginal prices for the aggregate CFMM
$$
\tilde{T}=\sum_{i=1}^{m} T_{i} .
$$

This is easy to see from the definitions of $C$ and $\tilde{T}$.
Connection to arbitrage. Note that $\Delta$ is a solution to the arbitrage problem at price $c$, i.e., $c^{T} \Delta=\operatorname{arb}(c)$ if, and only if,
$$
c \in C(\Delta)
$$
which follows by using the definition of $\mathbf{a r b}$ and $C$. In other words, the arbitrage problem is solved at any trade which changes the prices to match those of the external market with prices $c \in \mathbf{R}_{+}^{n}$. We say there is no arbitrage at price $c$ if the zero trade is a solution, i.e.,
$$
c \in C(0) .
$$

Equivalently, we may view this as the case where the CFMM's prices are consistent with those of the external market. Alternatively, there is a direct connection between the marginal prices, arbitrage, and the dual cone $K^{*}$ :
$$
c \in C(\Delta), \text { if, and only if, }\left(-c, c^{T} \Delta\right) \in K^{*} .
$$

We can see this since $c \in C(\Delta)$ if, and only if, for all $\Delta^{\prime} \in T$, we have
$$
c^{T} \Delta^{\prime} \leq c^{T} \Delta
$$

But $\left(\Delta^{\prime}, \lambda^{\prime}\right) \in K$ with $\lambda^{\prime}>0$ if and only if $\Delta^{\prime} / \lambda^{\prime} \in T$ so
$$
\frac{c^{T} \Delta^{\prime}}{\lambda^{\prime}} \leq c^{T} \Delta
$$

Multiplying both sides by $\lambda^{\prime}>0$ and using a limiting argument shows that this is true, if, and only if, for all $\left(\Delta^{\prime}, \lambda^{\prime}\right) \in K$ we have
$$
c^{T} \Delta^{\prime} \leq \lambda^{\prime} c^{T} \Delta
$$
which is the same as saying $\left(-c, c^{T} \Delta\right) \in K^{*}$.

\section*{D.3.3 Routing problem}

The routing problem takes a number of possible CFMMs $i=1, \ldots, m$, each trading a subset of $n_{i}$ tokens out of the universe of $n$ tokens, and seeks to find the best possible set of trades, i.e., those maximizing a given utility function $U: \mathbf{R}^{n} \rightarrow \mathbf{R} \cup\{-\infty\}$. We assume that $U$ is concave and increasing (i.e., we assume all assets have value with potentially diminishing marginal returns). We use infinite values of $U$ to encode constraints; a trade $\Psi$ such that $U(\Psi)=-\infty$ is unacceptable to the trader. See [Ang+22a, §5.2] for examples, including liquidating or purchasing a basket of tokens and finding arbitrage.

We denote the trade we make with the $i$ th CFMM by $\Delta_{i}$ and this CFMM's trading cone by $K_{i} \subseteq \mathbf{R}^{n_{i}+1}$. We also introduce matrices $A_{i} \in \mathbf{R}^{n \times n_{i}}$ which map the 'local' basket of $n_{i}$ tokens for CFMM $i$ to the global universe of $n$ tokens. This construction is similar to the construction of aggregate CFMMs in §D.2.2, but here we focus on the trade vectors and not the trading sets. The net trade is simply
$$
\Psi=\sum_{i=1}^{m} A_{i} \Delta_{i}
$$

The optimal routing problem is then the problem of finding a set of valid trades with each market that maximize the trader's utility:
$$
\begin{array}{ll}
\operatorname{maximize} & U(\Psi) \\
\text { subject to } & \Psi=\sum_{i=1}^{m} A_{i} \Delta_{i} \\
& \left(\Delta_{i}, 1\right) \in K_{i}, \quad i=1, \ldots, m
\end{array}
$$

The variables here are the net trade $\Psi \in \mathbf{R}^{n}$ and the trades $\Delta_{i} \in \mathbf{R}^{n_{i}}$. Note that, by definition of the trading cone $K_{i}$, we have that $\Delta_{i} \in T_{i}$ if, and only if, $\left(\Delta_{i}, 1\right) \in K_{i}$.

Other interpretations. If $A_{i}=I$, i.e., if all CFMMs trade the same tokens, then this problem is equivalent to
$$
\begin{array}{ll}
\operatorname{maximize} & U(\tilde{\Delta}) \\
\text { subject to } & \tilde{\Delta} \in \tilde{T}
\end{array}
$$
where $\tilde{T}=\sum_{i=1}^{m} T_{i}$, which is another trading set, by the composition rules given in §D.3.1. While this rewriting seems silly, it tells us that we may consider routing through a network of CFMMs as trading with one 'large' CFMM. The optimality conditions for this problem are that
$$
0 \in \partial(-U)\left(\tilde{\Delta}^{\star}\right)+\tilde{C}\left(\Delta^{\star}\right)
$$
and $\tilde{\Delta}^{\star} \in \tilde{T}$. From (D.27) we know that $\tilde{C}$ is the intersection of each individual price cone, so, using the definition of $\tilde{T}$, we get
$$
0 \in \partial(-U)\left(\sum_{i=1}^{m} A_{i} \Delta_{i}^{\star}\right)+\bigcap_{i=1}^{m} C_{i}\left(\Delta_{i}^{\star}\right)
$$
and $\Delta_{i}^{\star} \in T_{i}$, which are exactly the optimality conditions we would get from considering the original routing problem. The case where $A_{i}$ are general nonnegative orthogonal matrices is slightly more involved, but is ultimately very similar.

Dual problem. From conic duality (cf., appendix D.5.1), we know that the dual problem can be written as
$$
\begin{array}{ll}
\operatorname{minimize} & \bar{U}(\nu)+\mathbf{1}^{T} \eta \\
\text { subject to } & \left(-A_{i}^{T} \nu, \eta_{i}\right) \in K_{i}^{*}, \quad i=1, \ldots, m
\end{array}
$$
where the variables are $\nu \in \mathbf{R}^{n}$ and $\eta \in \mathbf{R}^{m}$. Partially minimizing over each $\eta_{i}$ and using the definition of the optimal arbitrage function, we have that this problem is equivalent to
$$
\operatorname{minimize} \bar{U}(\nu)+\sum_{i=1}^{m} \operatorname{arb}_{i}\left(A_{i}^{T} \nu\right)
$$
where $\mathbf{a r b}_{i}$ is the optimal arbitrage function for the $i$ th trading set. This is exactly the dual problem used in the decomposition method of [Dia +23 ]. This problem has a beautiful interpretation: the optimal trades are exactly those which result in a price vector $\nu$ that minimizes the total arbitrage profits that the user would receive if we interpret $\bar{U}(\nu)$ as the maximum utility that could be received by trading with an external market with price $\nu$.

\section*{D.3.4 Path independence}

In this subsection, we show the connection between the path-independent CFMMs, presented in the previous section, and the 'general' CFMMs presented in this one.

Mechanics of trading. In a CFMM, as stated previously, we have some state, which is given by the reserves $R \in \mathbf{R}_{+}^{n}$. The current trading set, defined as $T(R) \subseteq \mathbf{R}^{n}$ has the same properties given at the beginning of this section. (We implicitly included the relationship between the trading set and the reserves in the previous section as the reserves could be considered fixed for a single trade.) The CFMM then accepts or rejects any proposed trade $\Delta \in \mathbf{R}^{n}$ based on whether $\Delta \in T(R)$. If this is the case, then the CFMM accepts the trade, updating its reserves to $R \rightarrow R-\Delta$ (as it pays out $\Delta_{i}$ to the trader from its reserves for $\Delta_{i}>0$ and vice versa) and making the new trading set $T(R-\Delta)$. If the trade is rejected then the reserves are not updated and the trading set remains as-is.

Sequential feasibility. From before, we say a trade $\Delta$ is feasible if $\Delta \in T(R)$. We say a sequence of trades $\Delta_{i} \in \mathbf{R}^{n}$ for $i=1, \ldots, m$ is (sequentially) feasible if
$$
\Delta_{i} \in T\left(R-\left(\Delta_{1}+\cdots+\Delta_{i-1}\right)\right) .
$$
for each $i=1, \ldots, m$.
Reachability. We say some reserves $R^{\prime}$ are reachable from some initial set of reserves $R$ if there is a sequence of feasible trades $\Delta_{i} \in \mathbf{R}^{n}$, for $i=1, \ldots, m$, such that
$$
R^{\prime}=R-\left(\Delta_{1}+\cdots+\Delta_{m}\right) .
$$

In other words, $R^{\prime}$ is reachable from $R$ if there is a sequence of feasible trades that takes us from reserves $R$ to reserves $R^{\prime}$

Path independence. We say a CFMM is path independent if, for any reserves $R$ and for any trade $\Delta$ satisfying $\Delta \in T(R)$, we have
$$
\begin{equation*}
\Delta^{\prime} \in T(R-\Delta) \quad \text { if, and only if, } \quad \Delta+\Delta^{\prime} \in T(R) . \tag{D.28}
\end{equation*}
$$

In English: a CFMM is path independent if there is no difference between performing trades sequentially versus in aggregate, if the trades are sequentially feasible. (We may apply induction to this definition to get the more 'general-seeming' case that applies to any finite sequence of feasible trades.)

Reachable set. If the CFMM is path independent, there exists a fixed set $S \subseteq \mathbf{R}^{n}$ (which, as we will soon see, corresponds exactly to the reachable set of §D.2.1) such that every trading set $T\left(R^{\prime}\right)$ can be written as
$$
\begin{equation*}
T\left(R^{\prime}\right)=R^{\prime}-S \tag{D.29}
\end{equation*}
$$
for any reachable $R^{\prime}$, starting from some reserves $R$. (Here, $R^{\prime}-S=\left\{R^{\prime}-\tilde{R} \mid \tilde{R} \in S\right\}$.) Figure D. 7 illustrates these sets for Uniswap.

Proof. We will show that, whenever the CFMM is path independent, then we will have that, for any reachable $R^{\prime}, R-T(R)=R^{\prime}-T\left(R^{\prime}\right)$. Setting $S=R-T(R)$ will then suffice to satisfy (D.29). Note that it suffices to consider only $R^{\prime}$ which are reachable in 1 step, since the result follows by induction. That is, we will consider the case where $R^{\prime}=R-\Delta$ for $\Delta \in T(R)$ and the general case follows by induction.

We can rewrite the path independence condition (D.28) as
$$
\Delta^{\prime} \in T(R-\Delta) \quad \text { if, and only if, } \quad \Delta^{\prime} \in T(R)-\Delta,
$$
or, equivalently,
$$
T(R-\Delta)=T(R)-\Delta .
$$

The proof is then nearly obvious after this:
$$
R^{\prime}-T\left(R^{\prime}\right)=(R-\Delta)-T(R-\Delta)=R-\Delta-T(R)+\Delta=R-T(R) .
$$

We may then set $S=R^{\prime}-T\left(R^{\prime}\right)=R-T(R)$, such that (D.29) is satisfied for any $R^{\prime}$ reachable from $R$.
![](https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-149.jpg?height=519&width=622&top_left_y=330&top_left_x=244)

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-149.jpg?height=598&width=714&top_left_y=251&top_left_x=1153}
\captionsetup{labelformat=empty}
\caption{Figure D.7: The trading set $T\left(R^{\prime}\right)$ for Uniswap (left) and the corresponding reachable set $S$ (right).}
\end{figure}

Conditions. Note that the conditions on $T(R)$ will imply some conditions on $S$. Indeed, since $T(R)$ is a closed convex set, then $S$ must also be. Similarly, since $0 \in T(R)$ then $R \in S$, so $S$ is nonempty, and, since we must have $R-\Delta \geq 0$ for any $\Delta \in T(R)$ then $S=R-T(R) \subseteq \mathbf{R}_{+}^{n}$. Finally, since $T(R)$ is downward closed, then $S$ must be upward closed, hence $S$ must be a reachable set as defined in §D.2.1.

Equivalence. This is, of course, a bijection. We know that any path independent CFMM with trading set $T(R)$ may be written as
$$
T(R)=R-S,
$$
so long as $R \in S$. Additionally, if $S$ is a reachable set, then we must have that $0 \in T(R)$, that $T(R)$ is closed and convex, and that $T(R)$ is downward closed, making it a reasonable trading set. It is also bounded (D.21) since $S \subseteq \mathbf{R}_{+}^{n}$ so $R-T(R) \subseteq \mathbf{R}_{+}^{n}$.

Discussion. In general, it is tempting to deal with histories of trades among other objects when discussing CFMMs as these are dynamic systems with some internal state that changes as trades are performed. The above shows that, in the special case that the CFMM is path independent, we only need to consider the reachable set, as this contains all properties needed to completely describe the object in question. Indeed, the proof above shows that a CFMM is path-independent if, and only if, it is completely described by a reachable set meeting the conditions outlined in §D.2.1.

\section*{D. 4 Conclusion}

In this paper, we have shown that a general geometric perspective on constant function market makers is relatively fruitful. Indeed, assuming only a small number of 'intuitive' conditions on sets, we have derived a number of results-some known already, some not-which
follow from almost purely geometrical considerations. In some cases, we show that assumptions made in the literature actually are unnecessary, and indeed are consequences of a subset of the assumptions made. Examples of these include the homogeneity of [AEC21], [SKM23], and [FPW23]. In others, we derive a new form of a known result such as [AEC23] or [FPW23]. In addition, we showed that the objects typically studied in the CFMM literature - the reachable set of reserves, the trading function, and the portfolio value function - all come from a single geometric object: the liquidity cone (and its dual cone). Given one of these objects, the liquidity cone allows us to easily derive the others, illustrated in figure D.8. We suspect that there are a number of useful 'geometric' interpretations to other known results in the literature, but leave these for future work. As an interesting side note, after a preprint of this paper appeared online, Jack Mayo pointed out that some of the work found here is similar in spirit to recent literature in statistical learning theory [Wil14; WC23], serving a similar purpose to this work, except substituting the central object of CFMMs (the trading function) with the central object of statistical learning theory (losses). It is very likely that CFMMs have simple interpretations in terms of objects in statistical learning theory, and vice versa, which would also make for very interesting future work.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/d39c761d-2cf1-4caa-ba5e-87920c10f862-150.jpg?height=405&width=779&top_left_y=1095&top_left_x=672}
\captionsetup{labelformat=empty}
\caption{Figure D.8: Equivalence between CFMM representations. Arrows represent easy transformations between objects that were introduced in this work.}
\end{figure}

\section*{D. 5 Appendix}

\section*{D.5.1 Primer on conic duality}

This appendix is intended as a (very short) primer on conic duality. We assume basic familiarity with convex sets and the separating hyperplane theorem. For far more, see [BV04, §2.6].

Cones. A cone is a set $K \subseteq \mathbf{R}^{n}$ such that, if $x \in K$ then, for any $\alpha \geq 0$, we have $\alpha x \in K$. A convex cone is, as one would expect, a cone that is convex. More generally, convex cones are closed under nonnegative scalar multiplication, i.e., if $x, y \in K$ and $\alpha, \beta \geq 0$, then
$$
\alpha x+\beta y \in K .
$$

Basic examples of convex cones include the nonnegative real elements $\mathbf{R}_{+}^{n}$ and the norm cones, given by
$$
K_{\|\cdot\|}=\left\{(x, t) \in \mathbf{R}^{n+1} \mid\|x\| \leq t\right\} .
$$

Properties. If $K$ is closed and nonempty then $0 \in K$. The intersection of convex cones is a convex cone and scaling a convex cone results in a convex cone. Finally, the (Cartesian) product of two convex cones is again a convex cone.

Dual cone. The dual cone $K^{*}$ of a cone $K$ is defined as
$$
K^{*}=\left\{y \in \mathbf{R}^{n} \mid y^{T} x \geq 0, \text { for all } x \in K\right\} .
$$

In other words, the dual cone of $K$ is the set of all vectors which have nonnegative inner product with every element in $K$. Since we can write
$$
K^{*}=\bigcap_{x \in K}\left\{y \in \mathbf{R}^{n} \mid y^{T} x \geq 0\right\},
$$
then we can see that $K^{*}$ is a closed convex cone (even when $K$ is not). For example, the dual cone of $\mathbf{R}_{+}^{n}$ is $\mathbf{R}_{+}^{n}$, while the norm cone is
$$
K_{\|\cdot\|}^{*}=\left\{(y, r) \in \mathbf{R}^{n+1} \mid\|y\|_{*} \leq r\right\},
$$
where $\|y\|_{*}$ is the dual norm of $y$, defined
$$
\|y\|_{*}=\sup \left\{y^{T} x \mid\|x\| \leq 1\right\} .
$$

Properties. By definition $0 \in K$, and, since $K^{*}$ can be written as an intersection over $K$, then if $K^{\prime} \subseteq K$, we have
$$
K^{*} \subseteq K^{\prime *}
$$

Additionally note that
$$
\left(K+K^{\prime}\right)^{*}=K^{*} \cap K^{\prime *},
$$
and
$$
\left(K \times K^{\prime}\right)^{*}=K^{*} \times K^{\prime *},
$$
all of which are simple exercises and follow from the definition of the dual cone above.

Duality. In a certain sense, we may view the dual cone $K^{*}$ as a collection of certificates that an element is not in $K$. More specifically, if we have any $x \in \mathbf{R}^{n}$ and we are given some $y \in K^{*}$ such that $y^{T} x<0$, then we are guaranteed that, indeed $x \notin K$, by definition of $K^{*}$. Conic duality gives the following guarantee for a nonempty, closed convex cone $K$ : for any $x \in \mathbf{R}^{n}$, either $x \in K$, or there exists $y \in K^{*}$ with $y^{T} x<0$, but not both. In other words, given some point $x$, it either belongs in the cone, or we can furnish a certificate, using the dual cone, that it does not.

The reverse implication follows from the previous argument. (Note that this implication requires no assumptions on $K$.) The forward implication will make use of the convexity of $K$ to furnish a certificate. To see this, let $x \notin K$, then, since $K$ is convex closed, there exists a strict separating hyperplane with slope $y \in \mathbf{R}^{n}$ such that
$$
x^{T} y<z^{T} y, \text { for all } z \in K
$$

Since, for any $t \geq 0$ and $z \in K$, we have $t z \in K$, we therefore know that, for any $z \in K$,
$$
x^{T} y / t<z^{T} y
$$
and sending $t \rightarrow \infty$ we then know
$$
z^{T} y \geq 0 \text { for any } z \in K
$$
so $y \in K^{*}$. Finally, since $K$ is closed, then $0 \in K$ so
$$
x^{T} y<0
$$
completing the proof.
Dual of the dual. Because of the previous, we now have the following result: $K$ is exactly the set of vectors $x \in \mathbf{R}^{n}$ such that $x$ has nonnegative inner product with every element of $K^{*}$; i.e., for which we cannot furnish a certificate that $x \notin K$. But, the set of vectors which have nonnegative inner product with every element of $K^{*}$ is exactly the dual cone of $K^{*}$, written $\left(K^{*}\right)^{*}=K^{* *}$. This gives the following beautiful relation for a nonempty, closed, and convex cone $K$ :
$$
K^{* *}=K
$$

Conic duality in optimization. Most convex optimization problems can be cast as conic optimization problems. The general form of such a problem is, for some convex objective function $f: \mathbf{R}^{n} \rightarrow \mathbf{R} \cup\{\infty\}$
$$
\begin{array}{ll}
\operatorname{minimize} & f(x) \\
\text { subject to } & A x=b \\
& x \in K,
\end{array}
$$
where the variable is $x \in \mathbf{R}^{n}$, and the problem data are the closed nonempty convex cone $K \subseteq \mathbf{R}^{n}$, the matrix $A \in \mathbf{R}^{m \times n}$, and the constraint vector $b \in \mathbf{R}^{m}$.

Conic duality tells us that, if there exists any point in the interior of $K$, i.e., int $K \neq \varnothing$, then this problem and the following problem, called the dual problem, have the same optimal value
$$
\begin{array}{ll}
\text { maximize } & \bar{f}\left(A^{T} y\right)+b^{T} y \\
\text { subject to } & -A^{T} y \in K^{*}
\end{array}
$$
with variable $y \in \mathbf{R}^{n}$, where
$$
\bar{f}(z)=\inf _{x}\left(f(x)-x^{T} z\right),
$$
is sometimes known as the concave conjugate. As we only use this fact once in the main text, we do not derive it in detail, but see [Ber09, §5.3.6] for reference.

\section*{D.5.2 Curve}

In this section, we derive the canonical trading function for a two-asset Curve pool. Recall that the trading set for this market is given by [AC20b]
$$
S=\left\{R \left\lvert\, R_{1}+R_{2}-\frac{\alpha}{R_{1} R_{2}} \geq k\right.\right\}
$$

From (D.7), we can write the trading function as
$$
\varphi(R)=\sup \left\{\lambda>0 \left\lvert\, \frac{R_{1}+R_{2}}{\lambda}-\frac{\alpha \lambda^{2}}{R_{1} R_{2}} \geq k\right.\right\} .
$$

Rewriting, we have that
$$
\varphi(R)=\sup \left\{\lambda>0 \mid-\alpha \lambda^{3}-k R_{1} R_{2} \lambda+R_{1} R_{2}\left(R_{1}+R_{2}\right) \geq 0\right\} .
$$

The solution is given by the largest positive root of the cubic polynomial in $\lambda$ :
$$
\lambda^{\star}=\frac{\sqrt[3]{c_{1}(R)+\sqrt{c_{2}(R)}}}{3 \sqrt[3]{2} \alpha}-\frac{\sqrt[3]{2} k R_{1} R_{2}}{\sqrt[3]{c_{1}(R)+\sqrt{c_{2}(R)}}}
$$
where $c_{1}(R)=27 \alpha^{2} R_{1}^{2} R_{2}+27 \alpha^{2} R_{1} R_{2}^{2}$ and $c_{2}(R)=108 \alpha^{3} k^{3} R_{1}^{3} R_{2}^{3}+c_{1}^{2}$. Plugging this back in, we have the canonical trading function
$$
\varphi(R)=\frac{R_{1}+R_{2}}{k \lambda^{\star}}-\frac{\alpha\left(\lambda^{\star}\right)^{2}}{k R_{1} R_{2}},
$$
which can (painfully) be verified to be homogeneous.

\section*{D.5.3 Proof of concavity of Uniswap v3}

The main difficulty in showing that (D.11) is concave is the square root term
$$
\sqrt{\left(\beta R_{1}+\alpha R_{2}\right)^{2}+4(k-\alpha \beta) R_{1} R_{2}} .
$$

Its concavity follows from the fact that the set
$$
Q=\left\{(x, y, t) \in \mathbf{R}_{+}^{3} \mid\|(\sqrt{\eta}(x-y), t)\|_{2} \leq \sqrt{1+\eta}(x+y)\right\}
$$
is convex when $\eta \geq 0$, where $\|\cdot\|_{2}$ denotes the Euclidean norm. (To see this, note that norms are convex and affine functions are convex. Sets of the form $\{z \mid f(z) \leq 0\}$ are convex when $f$ is convex, and affine precomposition preserves convexity.) Expanding the inequality gives the following equivalent characterization of the set:
$$
Q=\left\{(x, y, t) \in \mathbf{R}_{+}^{3} \mid t \leq \sqrt{(x+y)^{2}+4 \eta x y}\right\}
$$
which means that the function
$$
\sqrt{(x+y)^{2}+4 \eta x y}=\sup \{t \geq 0 \mid(x, y, t) \in Q\}
$$
is concave in ( $x, y$ ). Finally, setting $\eta=(k-\alpha \beta) / \alpha \beta, x=\beta R_{1}$ and $y=\alpha R_{2}$ shows that the function
$$
\sqrt{\left(\beta R_{1}+\alpha R_{2}\right)^{2}+4(k-\alpha \beta) R_{1} R_{2}},
$$
is concave in $R_{1}$ and $R_{2}$.

\section*{D.5.4 Proof of consistency}

We are going to prove that the portfolio value function corresponding to the set $S_{V} \cap S_{V^{\prime}}$, which we write as
$$
\begin{equation*}
\tilde{V}(c)=\inf \left\{c^{T} R \mid R \in S_{V} \cap S_{V^{\prime}}\right\} \tag{D.30}
\end{equation*}
$$
for $c \geq 0$, is the pointwise smallest consistent function such that $\tilde{V} \geq \max \left\{V, V^{\prime}\right\}$. (This uses the same definitions as §2.2.1 We will show this in two steps, exploiting the equivalence between a portfolio value function $\hat{V}$ and its corresponding reachable set $S_{\hat{V}}$.

First, we show that $\tilde{V}$ is indeed larger than either $V$ or $V^{\prime}$. This is easy to see: let $R^{\star} \in S_{V} \cap S_{V^{\prime}}$ be the minimizer of (D.30) for a given price $c \geq 0$, then
$$
\tilde{V}(c)=c^{T} R^{\star} \geq V(c)
$$
by definition of $S_{V}$ and $S_{V^{\prime}}$. Since this is true for any $c$, then $\tilde{V} \geq V$ and $\tilde{V} \geq V^{\prime}$, as required.
Now, let $\hat{V}$ be any consistent portfolio value function which is at least (pointwise) as large as $V$ and $V^{\prime}$, then we will show that
$$
\begin{equation*}
S_{\hat{V}} \subseteq S_{V} \cap S_{V^{\prime}} \tag{D.31}
\end{equation*}
$$

This is also just definition chasing: since
$$
S_{\hat{V}}=\left\{R \in \mathbf{R}^{n} \mid c^{T} R \geq \hat{V}(c) \text { for all } c \geq 0\right\}
$$
but $\hat{V} \geq V$ and $\hat{V} \geq V^{\prime}$, then
$$
\left\{R \in \mathbf{R}^{n} \mid c^{T} R \geq \hat{V}(c) \text { for all } c \geq 0\right\} \subseteq S_{V} \cap S_{V^{\prime}}
$$

Since $S_{\tilde{V}}=S_{V} \cap S_{V^{\prime}}$, then this means that $S_{\hat{V}} \subseteq S_{\tilde{V}}$ for any consistent $\hat{V}$ that is pointwise at least as large as $V$ and $V^{\prime}$. But, since $S_{\hat{V}} \subseteq S_{\tilde{V}}$, then
$$
\hat{V}(c)=\inf \left\{c^{T} R \mid R \in S_{\hat{V}}\right\} \geq \inf \left\{c^{T} R \mid R \in S_{\tilde{V}}\right\}=\tilde{V}(c)
$$
for any $c \geq 0$, making $\tilde{V}$ the pointwise smallest consistent function larger than either $V$ or $V^{\prime}$.

\section*{References}
[AMO88] Ravindra K Ahuja, Thomas L Magnanti, and James B Orlin. Network flows. Cambridge, Mass.: Alfred P. Sloan School of Management, Massachusetts, 1988.
[Wil19] David P Williamson. Network flow algorithms. Cambridge University Press, 2019.
[Shi06] Maiko Shigeno. "Maximum network flows with concave gains". In: Mathematical programming 107.3 (2006), pp. 439-459.
[Vég14] László A Végh. "Concave generalized flows with applications to market equilibria". In: Mathematics of Operations Research 39.2 (2014), pp. 573-596.
[Ber98] Dimitri Bertsekas. Network optimization: continuous and discrete models. Vol. 8. Athena Scientific, 1998.
[Tru78] Klaus Truemper. "Optimal flows in nonlinear gain networks". In: Networks 8.1 (1978), pp. 17-36.
[Kuh55] Harold W Kuhn. "The Hungarian method for the assignment problem". In: Naval research logistics quarterly 2.1-2 (1955), pp. 83-97.
[Ber16] Dimitri Bertsekas. Nonlinear Programming. Third edition. Belmont, Massachusetts: Athena Scientific, 2016. 861 pp. ISBN: 978-1-886529-05-2.
[Chi+07] Mung Chiang, Steven H Low, A Robert Calderbank, and John C Doyle. "Layering as optimization decomposition: A mathematical theory of network architectures". In: Proceedings of the IEEE 95.1 (2007), pp. 255-312.
[Ber08] Dimitri P Bertsekas. "Extended monotropic programming and duality". In: Journal of optimization theory and applications 139.2 (2008), pp. 209-225.
[Roc84] RT Rockafellar. Network flows and monotropic programming. Wiley, 1984.
[Ber15] Dimitri Bertsekas. Convex optimization algorithms. Athena Scientific, 2015.
[Ang+22a] Guillermo Angeris, Alex Evans, Tarun Chitra, and Stephen Boyd. "Optimal routing for constant function market makers". In: Proceedings of the 23rd ACM Conference on Economics and Computation. 2022, pp. 115-128.
[Dia+23] Theo Diamandis, Max Resnick, Tarun Chitra, and Guillermo Angeris. "An efficient algorithm for optimal routing through constant function market makers". In: International Conference on Financial Cryptography and Data Security. Springer. 2023, pp. 128-145.
[DAE24a] Theo Diamandis, Guillermo Angeris, and Alan Edelman. "Convex Network Flows". In: arXiv preprint arXiv:2404.00765 (2024).
[DA24] Theo Diamandis and Guillermo Angeris. "Solving the Convex Flow Problem". In: to be presented at the Conference on Decision and Control 2024 (2024). URL: theodiamandis.com/pdfs/papers/routing-algorithm.pdf.
[Ang+23] Guillermo Angeris, Tarun Chitra, Theo Diamandis, Alex Evans, and Kshitij Kulkarni. "The geometry of constant function market makers". In: arXiv preprint arXiv:2308.08066 (2023).
[DAE24b] Theo Diamandis, Guillermo Angeris, and Alan Edelman. "The Geometry of Convex Network Flows". In: in preparation (2024).
[KDC23] Kshitij Kulkarni, Theo Diamandis, and Tarun Chitra. "Routing MEV in Constant Function Market Makers". In: International Conference on Web and Internet Economics. Springer. 2023, pp. 456-473.
[BV04] Stephen Boyd and Lieven Vandenberghe. Convex Optimization. 1st ed. Cambridge, United Kingdom: Cambridge University Press, 2004. 716 pp. ISBN: 978-0-521-83378-3.
[Roc70] R. Tyrrell Rockafellar. Convex Analysis. Vol. 28. Princeton university press, 1970.
[HR55] TE Harris and FS Ross. Fundamentals of a method for evaluating rail net capacities. Tech. rep. Rand Corporation, 1955.
[Sch02] Alexander Schrijver. "On the history of the transportation and maximum flow problems". In: Mathematical programming 91 (2002), pp. 437-445.
[FF56] Lester Randolph Ford and Delbert R Fulkerson. "Maximal flow through a network". In: Canadian journal of Mathematics 8 (1956), pp. 399-404.
[FF57] Lester Randolph Ford and Delbert R Fulkerson. "A simple algorithm for finding maximal network flows and an application to the Hitchcock problem". In: Canadian journal of Mathematics 9 (1957), pp. 210-218.
[WWS13] Allen J Wood, Bruce F Wollenberg, and Gerald B Sheblé. Power generation, operation, and control. John Wiley \& Sons, 2013.
[Stu19] Paul Melvin Stursberg. "On the mathematics of energy system optimization". PhD thesis. Technische Universität München, 2019.
[BG92] Dimitri Bertsekas and Robert Gallager. Data networks. Athena Scientific, 1992.
[Rou07] Tim Roughgarden. "Routing games". In: Algorithmic game theory 18 (2007), pp. 459-484.
[NB22] Anna Nagurney and Deniz Besik. "Spatial price equilibrium networks with flowdependent arc multipliers". In: Optimization Letters 16.8 (2022), pp. 2483-2500.
[XJB04] Lin Xiao, Mikael Johansson, and Stephen P Boyd. "Simultaneous routing and resource allocation via dual decomposition". In: IEEE Transactions on Communications 52.7 (2004), pp. 1136-1144.
[Sha48] Claude Elwood Shannon. "A mathematical theory of communication". In: The Bell system technical journal 27.3 (1948), pp. 379-423.
[Vaz12] Vijay V Vazirani. "The notion of a rational convex program, and an algorithm for the Arrow-Debreu Nash bargaining game". In: Journal of the ACM (JACM) 59.2 (2012), pp. 1-36.
[EG59] Edmund Eisenberg and David Gale. "Consensus of subjective probabilities: The pari-mutuel method". In: The Annals of Mathematical Statistics 30.1 (1959), pp. 165-168.
[Vaz07] Vijay V Vazirani. "Combinatorial algorithms for market equilibria". In: Algorithmic game theory (2007), pp. 103-134.
[Agr+22] Akshay Agrawal, Stephen Boyd, Deepak Narayanan, Fiodar Kazhamiaka, and Matei Zaharia. "Allocation of fungible resources via a fast, scalable price discovery method". In: Mathematical Programming Computation 14.3 (2022), pp. 593622.
[STA09] Peter Schütz, Asgeir Tomasgard, and Shabbir Ahmed. "Supply chain design under uncertainty using sample average approximation and dual decomposition". In: European journal of operational research 199.2 (2009), pp. 409-419.
[Ego19] Michael Egorov. "Stableswap-efficient mechanism for stablecoin liquidity". In: (2019).
[AZR20] Hayden Adams, Noah Zinsmeister, and Dan Robinson. "Uniswap v2 Core". In: URL: https://uniswap.org/whitepaper.pdf (2020).
[MM19a] Fernando Martinelli and Nikolai Mushegian. "Balancer: A Non-Custodial Portfolio Manager, Liquidity Provider, and Price Sensor". In: (2019).
[AC20a] Guillermo Angeris and Tarun Chitra. "Improved Price Oracles: Constant Function Market Makers". In: Proceedings of the 2nd ACM Conference on Advances in Financial Technologies. AFT '20: 2nd ACM Conference on Advances in Financial Technologies. New York NY USA: ACM, Oct. 21, 2020, pp. 80-91. ISBN: 978-1-4503-8139-0. DOI: 10.1145/3419614.3423251. (Visited on 02/17/2021).
[ZCP18] Yi Zhang, Xiaohong Chen, and Daejun Park. "Formal Specification of Constant Product ( $\mathrm{Xy}=\mathrm{k}$ ) Market Maker Model and Implementation". In: (2018).
[Ego] Michael Egorov. "StableSwap - Efficient Mechanism for Stablecoin Liquidity". In: (), p. 6. url: https://www.curve.fi/stableswap-paper.pdf.
[Ada+21a] Hayden Adams, Noah Zinsmeister, Moody Salem, River Keefer, and Dan Robinson. "Uniswap v3 Core". In: (2021). URL: https://uniswap.org/whitepaperv3.pdf.
[Ang+22b] Guillermo Angeris, Akshay Agrawal, Alex Evans, Tarun Chitra, and Stephen Boyd. "Constant Function Market Makers: Multi-asset Trades via Convex Optimization". In: Handbook on Blockchain. Cham: Springer International Publishing, 2022, pp. 415-444. ISBN: 978-3-031-07535-3. DOI: 10.1007/978-3-031-07535-3_13.
[Boy+07] Stephen Boyd, Lin Xiao, Almir Mutapcic, and Jacob Mattingley. "Notes on decomposition methods". In: Notes for EE364B, Stanford University 635 (2007), pp. 1-36.
[DF56] George Bernard Dantzig and Delbert R Fulkerson. "On the Max-Flow Min-Cut Theorem of Networks". In: 12 (1956), pp. 215-222.
[EFS56] Peter Elias, Amiel Feinstein, and Claude Shannon. "A note on the maximum flow through a network". In: IRE Transactions on Information Theory 2.4 (1956), pp. 117-119.
$[\mathrm{ODo}+16]$ Brendan O'Donoghue, Eric Chu, Neal Parikh, and Stephen Boyd. "Conic Optimization via Operator Splitting and Homogeneous Self-Dual Embedding". In: Journal of Optimization Theory and Applications 169.3 (June 2016), pp. 10421068. ISSN: 0022-3239, 1573-2878. DOI: 10.1007/ s10957-016-0892-3. URL: http://link.springer.com/10.1007/s10957-016-0892-3 (visited on 10/30/2020).
[CKV21] Chris Coey, Lea Kapelevich, and Juan Pablo Vielma. Solving natural conic formulations with Hypatia.jl. 2021. arXiv: 2005.01136 [math.OC].
[ApS24a] MOSEK ApS. MOSEK Optimizer API for Julia. 2024. url: https://docs. mosek.com/10.1/juliaapi/index.html.
[Byr+95] Richard H Byrd, Peihuang Lu, Jorge Nocedal, and Ciyou Zhu. "A limited memory algorithm for bound constrained optimization". In: SIAM Journal on scientific computing 16.5 (1995), pp. 1190-1208.
[Zhu+97] Ciyou Zhu, Richard H Byrd, Peihuang Lu, and Jorge Nocedal. "Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization". In: ACM Transactions on mathematical software (TOMS) 23.4 (1997), pp. 550-560.
[MN11] José Luis Morales and Jorge Nocedal. "Remark on "Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound constrained optimization'". In: ACM Transactions on Mathematical Software (TOMS) 38.1 (2011), pp. 1-4.
[DHL17] Iain Dunning, Joey Huchette, and Miles Lubin. "JuMP: A Modeling Language for Mathematical Optimization". In: SIAM Review 59.2 (Jan. 2017), pp. 295320. ISSN: 0036-1445, 1095-7200. DOI: 10.1137 / 15M1020575. URL: https: / / epubs.siam.org/doi/10.1137/15M1020575 (visited on 01/06/2020).
[Leg+21] Benoît Legat, Oscar Dowson, Joaquim Garcia, and Miles Lubin. "MathOptInterface: A Data Structure for Mathematical Optimization Problems". In: INFORMS Journal on Computing (Oct. 22, 2021), ijoc.2021.1067. ISSN: 10919856, 1526-5528. DOI: 10.1287/ijoc.2021.1067. URL: http://pubsonline.informs. org/doi/10.1287/ijoc.2021.1067 (visited on 02/08/2022).
[Lub+23] Miles Lubin, Oscar Dowson, Joaquim Dias Garcia, Joey Huchette, Benoît Legat, and Juan Pablo Vielma. "JuMP 1.0: Recent improvements to a modeling language for mathematical optimization". In: Mathematical Programming Computation (2023). DOI: 10.1007/s12532-023-00239-3.
[Bez+17] Jeff Bezanson, Alan Edelman, Stefan Karpinski, and Viral Shah. "Julia: A Fresh Approach to Numerical Computing". In: SIAM Review 59.1 (Jan. 2017), pp. 6598. ISSN: 0036-1445, 1095-7200. DOI: 10.1137/141000671. URL: https://epubs. siam.org/doi/10.1137/141000671 (visited on 01/06/2020).
[Kra+13] Matt Kraning, Eric Chu, Javad Lavaei, Stephen Boyd, et al. "Dynamic network energy management via proximal message passing". In: Foundations and Trends(R) in Optimization 1.2 (2013), pp. 73-126.
[Ste+20] Bartolomeo Stellato, Goran Banjac, Paul Goulart, Alberto Bemporad, and Stephen Boyd. "OSQP: An Operator Splitting Solver for Quadratic Programs". In: Mathematical Programming Computation (Feb. 20, 2020). ISSN: 1867-2949, 1867-2957. DOI: 10.1007/s12532-020-00179-2. URL: http://link.springer.com/ 10.1007/s12532-020-00179-2 (visited on 05/27/2020).
[Ude+14] Madeleine Udell, Karanveer Mohan, David Zeng, Jenny Hong, Steven Diamond, and Stephen Boyd. "Convex optimization in Julia". In: 2014 First Workshop for High Performance Technical Computing in Dynamic Languages. IEEE. 2014, pp. 18-28.
[RLP16] J. Revels, M. Lubin, and T. Papamarkou. "Forward-Mode Automatic Differentiation in Julia". In: arXiv:1607.07892 [cs.MS] (2016). url: https://arxiv.org/ abs/1607.07892.
[LO13] Adrian S Lewis and Michael L Overton. "Nonsmooth optimization via quasiNewton methods". In: Mathematical Programming 141 (2013), pp. 135-163.
[AO21] Azam Asl and Michael L Overton. "Behavior of limited memory BFGS when applied to nonsmooth functions and their Nesterov smoothings". In: Numerical Analysis and Optimization: NAO-V, Muscat, Oman, January 2020 V. Springer. 2021, pp. 25-55.
[NW06] Jorge Nocedal and Stephen J. Wright. Numerical Optimization. 2nd ed. Springer Series in Operations Research. New York: Springer, 2006. 664 pp. ISBN: 978-0-387-30303-1.
[Kar72] Richard M Karp. Reducibility among combinatorial problems. Springer, 1972.
[ApS24b] Mosek ApS. MOSEK Modeling Cookbook. Version 3.3.0. Jan. 2024.
[Ang+20] Guillermo Angeris, Hsien-Tang Kao, Rei Chiang, Charlie Noyes, and Tarun Chitra. "An Analysis of Uniswap Markets". In: Cryptoeconomic Systems (Nov. 25, 2020). In collab. with Reuben Youngblom. doi: 10.21428/58320208.c9738e64. URL: https: / / cryptoeconomicsystems . pubpub . org / pub / angeris - uniswap analysis (visited on 07/08/2021).
[Wan+22] Ye Wang, Yan Chen, Haotian Wu, Liyi Zhou, Shuiguang Deng, and Roger Wattenhofer. "Cyclic Arbitrage in Decentralized Exchanges". In: Companion Proceedings of the Web Conference 2022. Virtual Event, Lyon France: ACM, Apr. 2022, pp. 12-19. ISBN: 978-1-4503-9130-6. DOI: 10.1145/3487553.3524201.
[DKP21a] Vincent Danos, Hamza El Khalloufi, and Julien Prat. "Global Order Routing on Exchange Networks". In: Financial Cryptography and Data Security. FC 2021 International Workshops. Ed. by Matthew Bernhard, Andrea Bracciali, Lewis Gudgeon, Thomas Haines, Ariah Klages-Mundt, Shin'ichiro Matsuo, Daniel Perez, Massimiliano Sala, and Sam Werner. Vol. 12676. Berlin, Heidelberg: Springer Berlin Heidelberg, 2021, pp. 207-226. ISBN: 978-3-662-63957-3 978-3-662-63958-0. DOI: 10.1007/978-3-662-63958-0_19.
[DW60] George B Dantzig and Philip Wolfe. "Decomposition principle for linear programs". In: Operations research 8.1 (1960), pp. 101-111.
[ACE22a] Guillermo Angeris, Tarun Chitra, and Alex Evans. "When Does The Tail Wag The Dog? Curvature and Market Making". In: Cryptoeconomic Systems 2.1 (June 2022). Ed. by Reuben Youngblom.
[McC56] John McCarthy. "Measures of the value of information". In: Proceedings of the National Academy of Sciences 42.9 (1956), pp. 654-655.
[MM19b] Fernando Martinelli and Nikolai Mushegian. "A non-custodial portfolio manager, liquidity provider, and price sensor". In: URl: https://balancer. finance/whitepaper (2019).
[AC20b] Guillermo Angeris and Tarun Chitra. "Improved price oracles: Constant function market makers". In: Proceedings of the 2nd ACM Conference on Advances in Financial Technologies. 2020, pp. 80-91.
[LP21] Alfred Lehar and Christine A Parlour. "Decentralized exchanges". In: Available at SSRN 3905316 (2021).
[WM22] Mike Wu and Will McTighe. Constant Power Root Market Makers. 2022. arXiv: 2205.07452 [cs.CE].
[MMR23a] Jason Milionis, Ciamac C. Moallemi, and Tim Roughgarden. Complexity-Approximation Trade-offs in Exchange Mechanisms: AMMs vs. LOBs. 2023. arXiv: 2302.11652 [math. FA].
[SKM23] Jan Christoph Schlegel, Mateusz Kwaśnicki, and Akaki Mamageishvili. Axioms for Constant Function Market Makers. 2023. arXiv: 2210.00048 [cs.GT].
[FPW23] Rafael Frongillo, Maneesha Papireddygari, and Bo Waggoner. "An Axiomatic Characterization of CFMMs and Equivalence to Prediction Markets". In: arXiv preprint arXiv:2302.00196 (2023).
[MMR23b] Jason Milionis, Ciamac C. Moallemi, and Tim Roughgarden. A Myersonian Framework for Optimal Liquidity Provision in Automated Market Makers. 2023. arXiv: 2303.00208 [cs.GT].
[Goy+23] Mohak Goyal, Geoffrey Ramseyer, Ashish Goel, and David Mazières. "Finding the Right Curve: Optimal Design of Constant Function Market Makers". In: Proceedings of the 24th ACM Conference on Economics and Computation. 2023, pp. 783-812.
[FKP23] Michele Fabi, Myriam Kassoul, and Julien Prat. "SoK: constant function market makers". In: Working paper (2023).
[FP23] Michele Fabi and Julien Prat. "The economics of constant function market makers". In: Working paper (2023).
[ACE22b] Guillermo Angeris, Tarun Chitra, and Alex Evans. "When Does The Tail Wag The Dog? Curvature and Market Making". In: Cryptoeconomic Systems 2.1 (2022). Ed. by Reuben Youngblom.
[DKP21b] Vincent Danos, Hamza El Khalloufi, and Julien Prat. "Global Order Routing on Exchange Networks". In: Financial Cryptography and Data Security. FC 2021 International Workshops. Ed. by Matthew Bernhard, Andrea Bracciali, Lewis Gudgeon, Thomas Haines, Ariah Klages-Mundt, Shin'ichiro Matsuo, Daniel Perez, Massimiliano Sala, and Sam Werner. Berlin, Heidelberg: Springer Berlin Heidelberg, 2021, pp. 207-226. ISBN: 978-3-662-63958-0.
[FMW23] Masaaki Fukasawa, Basile Maire, and Marcus Wunsch. "Weighted variance swaps hedge against impermanent loss". In: Quantitative Finance 23.6 (2023), pp. 901-911. DOI: 10.1080/14697688.2023.2202708. eprint: https://doi.org/10. 1080/14697688.2023.2202708. URL: https:/ / doi.org/10.1080/14697688.2023. 2202708.
[MD23] Bruno Mazorra and Nicolás Della Penna. "Towards Optimal Prior-Free Permissionless Rebate Mechanisms, with applications to Automated Market Makers \& Combinatorial Orderflow Auctions". In: arXiv preprint arXiv:2306.17024 (2023).
[Ang+22c] Guillermo Angeris, Akshay Agrawal, Alex Evans, Tarun Chitra, and Stephen Boyd. "Constant function market makers: Multi-asset trades via convex optimization". In: Handbook on Blockchain. Springer, 2022, pp. 415-444.
[AEC21] Guillermo Angeris, Alex Evans, and Tarun Chitra. A Note on Privacy in Constant Function Market Makers. 2021. arXiv: 2103.01193 [cs, math].
[AEC23] Guillermo Angeris, Alex Evans, and Tarun Chitra. "Replicating market makers". In: Digital Finance (2023), pp. 1-21.
[Ada+21b] Hayden Adams, Noah Zinsmeister, Moody Salem, River Keefer, and Dan Robinson. "Uniswap v3 core". In: Tech. rep., Uniswap, Tech. Rep. (2021).
[EH21] Daniel Engel and Maurice Herlihy. "Composing networks of automated market makers". In: Proceedings of the 3rd ACM Conference on Advances in Financial Technologies. ACM, 2021. DOI: 10.1145/3479722.3480987. uRL: https:// doi. org/10.1145\%2F3479722.3480987.
[CAE21] Tarun Chitra, Guillermo Angeris, and Alex Evans. "How Liveness Separates CFMMs and Order Books". In: (2021).
[Art+99] Philippe Artzner, Freddy Delbaen, Jean-Marc Eber, and David Heath. "Coherent measures of risk". In: Mathematical finance 9.3 (1999), pp. 203-228.
[AFK15] Jacob Abernethy, Rafael Frongillo, and Sindhu Kutty. "On risk measures, market making, and exponential families". In: ACM SIGecom Exchanges 13.2 (2015), pp. 21-25.
[Wil14] Robert C Williamson. "The geometry of losses". In: Conference on Learning Theory. PMLR. 2014, pp. 1078-1108.
[WC23] Robert C Williamson and Zac Cranko. "The geometry and calculus of losses". In: Journal of Machine Learning Research 24.342 (2023), pp. 1-72.
[Ber09] Dimitri Bertsekas. Convex optimization theory. Vol. 1. Athena Scientific, 2009.