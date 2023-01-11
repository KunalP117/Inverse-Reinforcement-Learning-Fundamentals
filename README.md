# IRL-Fundamentals
Implementation of Andrew Ng's seminal IRL paper's feasibility based results (https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
for recovering an MDP's rewards given its optimal policy \pi*. 

Notation: Number of states - X, Number of actions - A

Showed how regularization can bring down the size of the feasible set by 30% (and increase precision)

- Main file: main_irl.m
- Used a full backup based policy iteration to compute the unique deterministic optimal MDP policy (policy_iteration.m)
- IRL analysis:
  (a) Vanilla implementation : Eq. 4 in Theorem 3 (Figure 1)
  
  (b) Intuitive discriminatory testing : Ensuring V^{\pi*} >= V^{\pi} for all deterministic policies \pi (Figure 2). In words, finding all reward functions
  whose value function with policy \pi* is atleast as good as any other deterministic policy \pi. 
  - Same computational and memory complexity as (a)
  
  (c) One-state deviation testing: Eq. 4 in Theorem 3 in the paper, but restricted only to those policies that deviate from the optimal policy in only
  ONE state (Figure 3). Can be showed that (c) is equivalent to (a).
  - Massive drop in computational and memory complexity: Resulting constraint matrix has dimensions [X*(A-1) x X] compared to [ (A^X - 1) x X] that grows
  exponentially with number of states X. 
  - Performance EQUIVALENT to (a)
  
  (d) Implementation of (c) with regularization: 3 vertically concatenated variants of Constraint matrix in (c). First copy is the same as (c), 
  Second copy: Constraint matrix of (c) + L*I (regularization term), Third copy: Constraint matrix of (c) - L*I (regularization term).
  - Intuition: Due to smoothness, continuity of optimal policy with reward function, one can expect that the optimal policy stays optimal
  even if the reward is tweaked slightly. (d) tests for: (i) rewards R, R + L, R-L for which the optimal policy is \pi*.
  - 3x memory and computational complexity compared to (c), but yields a feasible set upto 30% smaller (Increased Precision).
  - Careful with L. Too large L will cause the true reward to fail the feasibility test.


Notes: I was initially expecting (b) to outperform (a), but simulations showed otherwise. Due to the beautiful infinite horizon structure,
  the "Best" one can do is ensure one step sub-optimal policy choice followed by optimal policy choice is worse off, for all sub-optimal
  policies. Implementation (a) tests this condition. I realized (a) is a subset of (b) since one-step sub-optimal choice being worse off than 
  optimal policy choice from step 1 "implies" infinitely many steps of sub-optimal choice is worse off compared to the optimal choice.
