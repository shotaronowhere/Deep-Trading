pub mod diagnostics;
mod cfmmrouter_bridge;
mod global_solver;
mod global_solver_dual;
mod global_solver_dual_router;
mod merge;
mod planning;
mod rebalancer;
mod sim;
mod solver;
mod trading;
mod types;
mod waterfall;

pub use global_solver::{
    GlobalCandidateInvalidReason, GlobalOptimizer, GlobalSolveConfig, GlobalSolveResult,
};
pub use rebalancer::{
    RebalanceConfig, RebalanceDecisionDiagnostics, RebalanceEngine, RebalanceMode, rebalance,
    rebalance_with_config, rebalance_with_config_and_diagnostics, rebalance_with_mode,
};
pub use types::Action;

#[cfg(test)]
#[path = "../tests.rs"]
mod tests;
