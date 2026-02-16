use std::{collections::HashMap, fmt, io::IsTerminal};

use crate::execution::GroupKind;
use crate::execution::grouping::{
    ActionGroup, ProfitabilityStepKind, group_actions_by_profitability_step,
};
use crate::markets::MarketData;
use crate::pools::{Slot0Result, prediction_to_sqrt_price_x96};

use super::Action;
use super::sim::build_sims;

#[derive(Debug, Clone, Copy)]
pub struct TraceConfig {
    pub ansi_enabled: bool,
    pub full_market_deltas: bool,
    pub full_portfolio: bool,
}

impl TraceConfig {
    pub fn from_env() -> Self {
        let ansi_enabled = ansi_enabled();
        Self {
            ansi_enabled,
            full_market_deltas: env_flag_enabled("REBALANCE_FULL_MARKET_DELTAS"),
            full_portfolio: env_flag_enabled("REBALANCE_FULL_PORTFOLIO"),
        }
    }
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

pub fn replay_actions_to_portfolio_state(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
    initial_balances: &HashMap<&str, f64>,
    initial_susd: f64,
) -> (HashMap<&'static str, f64>, f64) {
    let mut holdings: HashMap<&'static str, f64> = HashMap::new();
    for (_, market) in slot0_results {
        holdings.insert(
            market.name,
            initial_balances
                .get(market.name)
                .copied()
                .unwrap_or(0.0)
                .max(0.0),
        );
    }

    let mut cash = initial_susd;
    for action in actions {
        match action {
            Action::Buy {
                market_name,
                amount,
                cost,
            } => {
                *holdings.entry(*market_name).or_insert(0.0) += *amount;
                cash -= *cost;
            }
            Action::Sell {
                market_name,
                amount,
                proceeds,
            } => {
                *holdings.entry(*market_name).or_insert(0.0) -= *amount;
                cash += *proceeds;
            }
            Action::Mint { amount, .. } => {
                for (_, market) in slot0_results {
                    *holdings.entry(market.name).or_insert(0.0) += *amount;
                }
            }
            Action::Merge { amount, .. } => {
                for (_, market) in slot0_results {
                    *holdings.entry(market.name).or_insert(0.0) -= *amount;
                }
                cash += *amount;
            }
            Action::FlashLoan { amount } => cash += *amount,
            Action::RepayFlashLoan { amount } => cash -= *amount,
        }
    }

    (holdings, cash)
}

fn try_replay_actions_to_market_state(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> Option<Vec<(Slot0Result, &'static MarketData)>> {
    let preds = crate::pools::prediction_map();
    let mut sims = build_sims(slot0_results, &preds).ok()?;
    let mut idx_by_market: HashMap<&str, usize> = HashMap::new();
    for (i, sim) in sims.iter().enumerate() {
        idx_by_market.insert(sim.market_name, i);
    }

    for action in actions {
        match action {
            Action::Buy {
                market_name,
                amount,
                ..
            } => {
                if let Some(&idx) = idx_by_market.get(market_name) {
                    if let Some((bought, _, new_price)) = sims[idx].buy_exact(*amount) {
                        if bought > 0.0 {
                            sims[idx].set_price(new_price);
                        }
                    }
                }
            }
            Action::Sell {
                market_name,
                amount,
                ..
            } => {
                if let Some(&idx) = idx_by_market.get(market_name) {
                    if let Some((sold, _, new_price)) = sims[idx].sell_exact(*amount) {
                        if sold > 0.0 {
                            sims[idx].set_price(new_price);
                        }
                    }
                }
            }
            Action::Mint { .. }
            | Action::Merge { .. }
            | Action::FlashLoan { .. }
            | Action::RepayFlashLoan { .. } => {}
        }
    }

    Some(
        slot0_results
            .iter()
            .map(|(slot0, market)| {
                let mut next = slot0.clone();
                if let Some(&idx) = idx_by_market.get(market.name) {
                    if let Some(pool) = market.pool.as_ref() {
                        let is_token1_outcome =
                            pool.token1.to_lowercase() == market.outcome_token.to_lowercase();
                        let p = sims[idx].price().max(1e-12);
                        next.sqrt_price_x96 = prediction_to_sqrt_price_x96(p, is_token1_outcome)
                            .unwrap_or(slot0.sqrt_price_x96);
                    }
                }
                (next, *market)
            })
            .collect(),
    )
}

#[cfg(test)]
pub(crate) fn replay_actions_to_market_state(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> Vec<(Slot0Result, &'static MarketData)> {
    try_replay_actions_to_market_state(actions, slot0_results)
        .expect("test fixtures should have prediction coverage")
}

fn sorted_market_prices(
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> Option<Vec<(&'static str, f64)>> {
    let preds = crate::pools::prediction_map();
    let sims = build_sims(slot0_results, &preds).ok()?;
    let mut prices: Vec<(&'static str, f64)> = sims
        .iter()
        .map(|sim| (sim.market_name, sim.price()))
        .collect();
    prices.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    Some(prices)
}

fn replay_trade_boundary_hits(
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
) -> Option<HashMap<usize, bool>> {
    let preds = crate::pools::prediction_map();
    let mut sims = build_sims(slot0_results, &preds).ok()?;
    let mut idx_by_market: HashMap<&str, usize> = HashMap::new();
    for (i, sim) in sims.iter().enumerate() {
        idx_by_market.insert(sim.market_name, i);
    }

    let mut hits_by_action: HashMap<usize, bool> = HashMap::new();
    for (action_index, action) in actions.iter().enumerate() {
        match action {
            Action::Buy {
                market_name,
                amount,
                ..
            } => {
                let Some(&idx) = idx_by_market.get(market_name) else {
                    continue;
                };
                let Some((bought, _, new_price)) = sims[idx].buy_exact(*amount) else {
                    continue;
                };
                if bought <= 0.0 {
                    continue;
                }
                let boundary = sims[idx].buy_limit_price;
                let tol = 1e-9 * (1.0 + boundary.abs().max(new_price.abs()));
                let hit_boundary = (new_price - boundary).abs() <= tol;
                hits_by_action.insert(action_index, hit_boundary);
                sims[idx].set_price(new_price);
            }
            Action::Sell {
                market_name,
                amount,
                ..
            } => {
                let Some(&idx) = idx_by_market.get(market_name) else {
                    continue;
                };
                let Some((sold, _, new_price)) = sims[idx].sell_exact(*amount) else {
                    continue;
                };
                if sold <= 0.0 {
                    continue;
                }
                let boundary = sims[idx].sell_limit_price;
                let tol = 1e-9 * (1.0 + boundary.abs().max(new_price.abs()));
                let hit_boundary = (new_price - boundary).abs() <= tol;
                hits_by_action.insert(action_index, hit_boundary);
                sims[idx].set_price(new_price);
            }
            Action::Mint { .. }
            | Action::Merge { .. }
            | Action::FlashLoan { .. }
            | Action::RepayFlashLoan { .. } => {}
        }
    }

    Some(hits_by_action)
}

fn print_market_price_changes(
    label: &str,
    prices_before: &[(&'static str, f64)],
    prices_after: &[(&'static str, f64)],
    config: TraceConfig,
) {
    #[derive(Clone)]
    struct DeltaRow {
        market: String,
        before: f64,
        after: f64,
        pct_change: f64,
        direction: &'static str,
        color: &'static str,
    }

    let after_by_market: HashMap<&'static str, f64> = prices_after.iter().copied().collect();
    let ansi = config.ansi_enabled;
    let reset = if ansi { ANSI_RESET } else { "" };
    let full_output = config.full_market_deltas;
    let mut rows: Vec<DeltaRow> = Vec::new();
    let mut missing: Vec<(String, f64)> = Vec::new();
    let mut up_count = 0usize;
    let mut down_count = 0usize;
    let mut flat_count = 0usize;

    for &(market_name, before) in prices_before {
        let Some(after) = after_by_market.get(market_name).copied() else {
            missing.push((display_market_name(market_name), before));
            continue;
        };
        let delta = after - before;
        let pct_change = if before.abs() <= 1e-12 {
            0.0
        } else {
            100.0 * delta / before
        };
        let (color, direction) = if delta > 1e-12 {
            (ANSI_GREEN, "up")
        } else if delta < -1e-12 {
            (ANSI_RED, "down")
        } else {
            (ANSI_GRAY, "flat")
        };
        match direction {
            "up" => up_count += 1,
            "down" => down_count += 1,
            _ => flat_count += 1,
        }
        rows.push(DeltaRow {
            market: display_market_name(market_name),
            before,
            after,
            pct_change,
            direction,
            color,
        });
    }

    let total_rows = rows.len() + missing.len();
    if total_rows == 0 {
        println!("[rebalance][{}] market price changes: unavailable", label);
        return;
    }

    if full_output || rows.len() <= MARKET_DELTA_PREVIEW_ROWS {
        println!("[rebalance][{}] market price changes:", label);
        for row in &rows {
            println!(
                "  {}{:<36}: {:.6} -> {:.6} ({:+.3}%, {}){}",
                if ansi { row.color } else { "" },
                row.market,
                row.before,
                row.after,
                row.pct_change,
                row.direction,
                reset
            );
        }
    } else {
        let mut up_rows: Vec<&DeltaRow> = rows.iter().filter(|row| row.direction == "up").collect();
        up_rows.sort_by(|lhs, rhs| {
            rhs.pct_change
                .partial_cmp(&lhs.pct_change)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| lhs.market.cmp(&rhs.market))
        });
        let mut down_rows: Vec<&DeltaRow> =
            rows.iter().filter(|row| row.direction == "down").collect();
        down_rows.sort_by(|lhs, rhs| {
            lhs.pct_change
                .partial_cmp(&rhs.pct_change)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| lhs.market.cmp(&rhs.market))
        });
        let mut flat_rows: Vec<&DeltaRow> =
            rows.iter().filter(|row| row.direction == "flat").collect();
        flat_rows.sort_by(|lhs, rhs| lhs.market.cmp(&rhs.market));

        let per_side = MARKET_DELTA_PREVIEW_ROWS / 2;
        let mut up_take = up_rows.len().min(per_side);
        let mut down_take = down_rows.len().min(per_side);
        let mut remaining = MARKET_DELTA_PREVIEW_ROWS.saturating_sub(up_take + down_take);
        while remaining > 0 {
            let up_remaining = up_rows.len().saturating_sub(up_take);
            let down_remaining = down_rows.len().saturating_sub(down_take);
            if up_remaining == 0 && down_remaining == 0 {
                break;
            }
            if up_remaining >= down_remaining && up_remaining > 0 {
                up_take += 1;
                remaining -= 1;
                continue;
            }
            if down_remaining > 0 {
                down_take += 1;
                remaining -= 1;
                continue;
            }
            if up_remaining > 0 {
                up_take += 1;
                remaining -= 1;
                continue;
            }
        }
        let flat_take = flat_rows.len().min(remaining);
        let shown = up_take + down_take + flat_take;

        println!(
            "[rebalance][{}] market price changes: top movers (up={}, down={}, flat={}) of {} (set REBALANCE_FULL_MARKET_DELTAS=1 for full list)",
            label,
            up_take,
            down_take,
            flat_take,
            rows.len()
        );

        let print_row = |row: &DeltaRow| {
            println!(
                "  {}{:<36}: {:.6} -> {:.6} ({:+.3}%, {}){}",
                if ansi { row.color } else { "" },
                row.market,
                row.before,
                row.after,
                row.pct_change,
                row.direction,
                reset
            );
        };

        if up_take > 0 {
            println!("  top up movers:");
            for row in up_rows.iter().take(up_take) {
                print_row(row);
            }
        } else {
            println!("  top up movers: (none)");
        }

        if down_take > 0 {
            println!("  top down movers:");
            for row in down_rows.iter().take(down_take) {
                print_row(row);
            }
        } else {
            println!("  top down movers: (none)");
        }

        if flat_take > 0 {
            println!("  top flat movers:");
            for row in flat_rows.iter().take(flat_take) {
                print_row(row);
            }
        }

        let omitted = rows.len().saturating_sub(shown);
        println!(
            "  ... omitted {} rows (up={}, down={}, flat={})",
            omitted, up_count, down_count, flat_count
        );
    }

    if !missing.is_empty() {
        let preview: Vec<&str> = missing
            .iter()
            .take(3)
            .map(|(name, _)| name.as_str())
            .collect();
        println!(
            "  {}missing after-price for {} markets (e.g. {}){}",
            if ansi { ANSI_GRAY } else { "" },
            missing.len(),
            preview.join(", "),
            reset
        );
    }
}

fn split_actions_by_complete_set_arb_phase(actions: &[Action]) -> (&[Action], &[Action]) {
    const COMPLETE_SET_ARB: &str = "complete_set_arb";
    let marker_idx = actions.iter().position(|action| match action {
        Action::Merge { source_market, .. } => *source_market == COMPLETE_SET_ARB,
        Action::Mint { target_market, .. } => *target_market == COMPLETE_SET_ARB,
        _ => false,
    });

    let Some(marker_idx) = marker_idx else {
        return (&actions[0..0], actions);
    };

    let mut arb_end = marker_idx;
    if matches!(actions.first(), Some(Action::FlashLoan { .. }))
        && let Some(repay_offset) = actions[marker_idx + 1..]
            .iter()
            .position(|action| matches!(action, Action::RepayFlashLoan { .. }))
    {
        arb_end = marker_idx + 1 + repay_offset;
    }

    (&actions[..=arb_end], &actions[arb_end + 1..])
}

const LARGE_ACTION_PREVIEW_MIN_ACTIONS: usize = 20;
const ACTION_PREVIEW_HEAD_GROUPS: usize = 2;
const ACTION_PREVIEW_TAIL_GROUPS: usize = 2;
const FLASH_SUBGROUP_PREVIEW_HEAD: usize = 2;
const FLASH_SUBGROUP_PREVIEW_TAIL: usize = 1;
const MARKET_DELTA_PREVIEW_ROWS: usize = 16;
const PORTFOLIO_PREVIEW_ROWS: usize = 20;
const GROUP_KIND_COL_WIDTH: usize = 28;
const GROUP_INDICES_COL_WIDTH: usize = 15;
const DISPLAY_MARKET_NAME_MAX_CHARS: usize = 36;
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_RED: &str = "\x1b[31m";
const ANSI_GREEN: &str = "\x1b[32m";
const ANSI_CYAN: &str = "\x1b[36m";
const ANSI_YELLOW: &str = "\x1b[33m";
const ANSI_GRAY: &str = "\x1b[90m";

#[derive(Debug, Clone, Copy, Default)]
struct BoundaryStats {
    trades: usize,
    hits: usize,
    unknown: usize,
}

impl BoundaryStats {
    fn record(&mut self, hit: Option<bool>) {
        self.trades += 1;
        match hit {
            Some(true) => self.hits += 1,
            Some(false) => {}
            None => self.unknown += 1,
        }
    }

    fn merge(&mut self, other: BoundaryStats) {
        self.trades += other.trades;
        self.hits += other.hits;
        self.unknown += other.unknown;
    }
}

fn ansi_enabled() -> bool {
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if std::env::var("TERM").ok().as_deref() == Some("dumb") {
        return false;
    }
    std::io::stdout().is_terminal()
}

fn colorize(ansi: bool, color: &str, text: &str) -> String {
    if ansi {
        format!("{color}{text}{ANSI_RESET}")
    } else {
        text.to_string()
    }
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn sanitize_label(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_space = false;
    for ch in raw.chars() {
        let normalized = if ch.is_control() || ch.is_whitespace() {
            ' '
        } else {
            ch
        };
        if normalized == ' ' {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(normalized);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

fn truncate_middle_ascii(raw: &str, max_chars: usize) -> String {
    let char_count = raw.chars().count();
    if char_count <= max_chars {
        return raw.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let front = (max_chars - 3) / 2;
    let back = max_chars - 3 - front;
    let prefix: String = raw.chars().take(front).collect();
    let suffix: String = raw.chars().skip(char_count - back).collect();
    format!("{prefix}...{suffix}")
}

fn display_market_name(raw: &str) -> String {
    truncate_middle_ascii(&sanitize_label(raw), DISPLAY_MARKET_NAME_MAX_CHARS)
}

fn trade_boundary_status(
    action: &Action,
    global_idx: usize,
    tick_boundary_hits: Option<&HashMap<usize, bool>>,
) -> Option<Option<bool>> {
    match action {
        Action::Buy { amount, .. } | Action::Sell { amount, .. } if amount.abs() > 1e-12 => Some(
            tick_boundary_hits
                .and_then(|hits| hits.get(&global_idx))
                .copied(),
        ),
        _ => None,
    }
}

fn collect_boundary_stats(
    actions: &[Action],
    indices: &[usize],
    global_offset: usize,
    tick_boundary_hits: Option<&HashMap<usize, bool>>,
) -> BoundaryStats {
    let mut stats = BoundaryStats::default();
    for &idx in indices {
        if let Some(hit) =
            trade_boundary_status(&actions[idx], global_offset + idx, tick_boundary_hits)
        {
            stats.record(hit);
        }
    }
    stats
}

fn format_boundary_badge(stats: BoundaryStats, ansi: bool) -> Option<String> {
    if stats.trades == 0 {
        return None;
    }
    let text = if stats.unknown == 0 {
        format!("[BOUNDARY {}/{}]", stats.hits, stats.trades)
    } else {
        let known = stats.trades - stats.unknown;
        format!(
            "[BOUNDARY {}/{} known, {} unknown]",
            stats.hits, known, stats.unknown
        )
    };
    let color = if stats.hits > 0 || stats.unknown > 0 {
        ANSI_YELLOW
    } else {
        ANSI_GRAY
    };
    Some(colorize(ansi, color, &text))
}

fn group_kind_label(kind: GroupKind) -> &'static str {
    match kind {
        GroupKind::DirectBuy => "DirectBuy",
        GroupKind::DirectSell => "DirectSell",
        GroupKind::DirectMerge => "DirectMerge",
        GroupKind::MintSell => "Arb:MintSell",
        GroupKind::BuyMerge => "Arb:BuyMerge",
    }
}

fn format_group_badges(
    actions: &[Action],
    group: &ActionGroup,
    global_offset: usize,
    tick_boundary_hits: Option<&HashMap<usize, bool>>,
    ansi: bool,
) -> String {
    let mut badges: Vec<String> = Vec::new();
    if matches!(group.kind, GroupKind::MintSell | GroupKind::BuyMerge) {
        badges.push(colorize(ansi, ANSI_CYAN, "[FLASH]"));
    }
    let boundary_stats = collect_boundary_stats(
        actions,
        &group.action_indices,
        global_offset,
        tick_boundary_hits,
    );
    if let Some(boundary_badge) = format_boundary_badge(boundary_stats, ansi) {
        badges.push(boundary_badge);
    }
    badges.join(" ")
}

fn print_group_preview_header(ansi: bool) {
    let header = format!(
        "{:<8} {:<kind_w$} {:>7} {:<idx_w$} {}",
        "group",
        "kind",
        "count",
        "indices",
        "badges",
        kind_w = GROUP_KIND_COL_WIDTH,
        idx_w = GROUP_INDICES_COL_WIDTH
    );
    println!("    {}", colorize(ansi, ANSI_CYAN, &header));
}

fn is_mixed_step(group: &ActionGroup) -> bool {
    matches!(
        group.step_kind,
        ProfitabilityStepKind::MixedDirectBuyMintSell
            | ProfitabilityStepKind::MixedDirectSellBuyMerge
    )
}

fn group_display_label(group: &ActionGroup) -> &'static str {
    match group.step_kind {
        ProfitabilityStepKind::MixedDirectBuyMintSell => "Mixed:Buy+MintSell",
        ProfitabilityStepKind::MixedDirectSellBuyMerge => "Mixed:Sell+BuyMerge",
        ProfitabilityStepKind::ArbMintSell
        | ProfitabilityStepKind::ArbBuyMerge
        | ProfitabilityStepKind::PureDirectBuy
        | ProfitabilityStepKind::PureDirectSell
        | ProfitabilityStepKind::PureDirectMerge => group_kind_label(group.kind),
    }
}

fn action_outline_detail(action: &Action) -> String {
    match action {
        Action::Buy {
            market_name,
            amount,
            cost,
        } => format!(
            "{} units={:.6} quote={:.6} px={:.6}",
            display_market_name(market_name),
            amount,
            cost,
            cost / amount.max(1e-18)
        ),
        Action::Sell {
            market_name,
            amount,
            proceeds,
        } => format!(
            "{} units={:.6} quote={:.6} px={:.6}",
            display_market_name(market_name),
            amount,
            proceeds,
            proceeds / amount.max(1e-18)
        ),
        Action::FlashLoan { amount } => format!("amount={:.6}", amount),
        Action::RepayFlashLoan { amount } => format!("amount={:.6}", amount),
        Action::Mint {
            amount,
            target_market,
            ..
        } => format!(
            "target={} amount={:.6}",
            display_market_name(target_market),
            amount
        ),
        Action::Merge {
            amount,
            source_market,
            ..
        } => format!(
            "source={} amount={:.6}",
            display_market_name(source_market),
            amount
        ),
    }
}

fn action_kind_label(action: &Action) -> &'static str {
    match action {
        Action::Buy { .. } => "Buy",
        Action::Sell { .. } => "Sell",
        Action::FlashLoan { .. } => "FlashLoan",
        Action::RepayFlashLoan { .. } => "Repay",
        Action::Mint { .. } => "Mint",
        Action::Merge { .. } => "Merge",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionSetKind {
    Buy,
    Sell,
    FlashLoan,
    Mint,
    Merge,
    RepayFlashLoan,
}

impl ActionSetKind {
    fn matches(self, action: &Action) -> bool {
        match self {
            Self::Buy => matches!(action, Action::Buy { .. }),
            Self::Sell => matches!(action, Action::Sell { .. }),
            Self::FlashLoan => matches!(action, Action::FlashLoan { .. }),
            Self::Mint => matches!(action, Action::Mint { .. }),
            Self::Merge => matches!(action, Action::Merge { .. }),
            Self::RepayFlashLoan => matches!(action, Action::RepayFlashLoan { .. }),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ActionSetStyle {
    StageSnapshot {
        label: &'static str,
    },
    Homogeneous {
        single_label: &'static str,
        first_label: &'static str,
        last_label: &'static str,
    },
}

#[derive(Debug, Clone, Copy)]
struct ActionSetSpec {
    kind: ActionSetKind,
    style: ActionSetStyle,
}

impl ActionSetSpec {
    const fn stage(kind: ActionSetKind, label: &'static str) -> Self {
        Self {
            kind,
            style: ActionSetStyle::StageSnapshot { label },
        }
    }

    const fn homogeneous(
        kind: ActionSetKind,
        single_label: &'static str,
        first_label: &'static str,
        last_label: &'static str,
    ) -> Self {
        Self {
            kind,
            style: ActionSetStyle::Homogeneous {
                single_label,
                first_label,
                last_label,
            },
        }
    }
}

const MIXED_BUY_MINT_SELL_SPECS: [ActionSetSpec; 6] = [
    ActionSetSpec::stage(ActionSetKind::Buy, "Buy"),
    ActionSetSpec::stage(ActionSetKind::FlashLoan, "Flash Loan"),
    ActionSetSpec::stage(ActionSetKind::Mint, "Mint"),
    ActionSetSpec::homogeneous(ActionSetKind::Sell, "Sell", "First Sell", "Last Sell"),
    ActionSetSpec::stage(ActionSetKind::Merge, "Merge"),
    ActionSetSpec::stage(ActionSetKind::RepayFlashLoan, "Repay"),
];

const MIXED_SELL_BUY_MERGE_SPECS: [ActionSetSpec; 6] = [
    ActionSetSpec::stage(ActionSetKind::Sell, "Sell"),
    ActionSetSpec::stage(ActionSetKind::FlashLoan, "Flash Loan"),
    ActionSetSpec::homogeneous(ActionSetKind::Buy, "Buy", "First Buy", "Last Buy"),
    ActionSetSpec::stage(ActionSetKind::Mint, "Mint"),
    ActionSetSpec::stage(ActionSetKind::Merge, "Merge"),
    ActionSetSpec::stage(ActionSetKind::RepayFlashLoan, "Repay"),
];

fn collect_action_set_indices(
    actions: &[Action],
    indices: &[usize],
    kind: ActionSetKind,
) -> Vec<usize> {
    indices
        .iter()
        .copied()
        .filter(|&idx| kind.matches(&actions[idx]))
        .collect()
}

struct ActionSetView<'a> {
    actions: &'a [Action],
    global_offset: usize,
    indices: &'a [usize],
    style: ActionSetStyle,
}

impl fmt::Display for ActionSetView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.style {
            ActionSetStyle::StageSnapshot { label } => match self.indices.len() {
                0 => Ok(()),
                1 => {
                    let idx = self.indices[0];
                    write!(
                        f,
                        "      -> {}[{}] {}",
                        label,
                        self.global_offset + idx,
                        action_outline_detail(&self.actions[idx])
                    )
                }
                _ => {
                    let first = self.indices[0];
                    let last = self.indices[self.indices.len() - 1];
                    write!(
                        f,
                        "      -> {}[{}..={}] count={}\n",
                        label,
                        self.global_offset + first,
                        self.global_offset + last,
                        self.indices.len(),
                    )?;
                    write!(
                        f,
                        "         first[{}] {}\n",
                        self.global_offset + first,
                        action_outline_detail(&self.actions[first]),
                    )?;
                    write!(
                        f,
                        "         last [{}] {}",
                        self.global_offset + last,
                        action_outline_detail(&self.actions[last])
                    )
                }
            },
            ActionSetStyle::Homogeneous {
                single_label,
                first_label,
                last_label,
            } => match self.indices.len() {
                0 => Ok(()),
                1 => {
                    let idx = self.indices[0];
                    write!(
                        f,
                        "      -> {}[{}] {}",
                        single_label,
                        self.global_offset + idx,
                        action_outline_detail(&self.actions[idx])
                    )
                }
                2 => {
                    let first = self.indices[0];
                    let second = self.indices[1];
                    write!(
                        f,
                        "      -> {}[{}] {}\n",
                        first_label,
                        self.global_offset + first,
                        action_outline_detail(&self.actions[first])
                    )?;
                    write!(
                        f,
                        "      -> {}[{}] {}",
                        last_label,
                        self.global_offset + second,
                        action_outline_detail(&self.actions[second])
                    )
                }
                _ => {
                    let first = self.indices[0];
                    let last = self.indices[self.indices.len() - 1];
                    write!(
                        f,
                        "      -> {}[{}] {}\n",
                        first_label,
                        self.global_offset + first,
                        action_outline_detail(&self.actions[first])
                    )?;
                    // Only used for homogeneous sequences, so both sides share the same action type.
                    write!(f, "      -> ...\n")?;
                    write!(
                        f,
                        "      -> {}[{}] {}",
                        last_label,
                        self.global_offset + last,
                        action_outline_detail(&self.actions[last])
                    )
                }
            },
        }
    }
}

fn print_mixed_group_outline(actions: &[Action], group: &ActionGroup, global_offset: usize) {
    let specs: &[ActionSetSpec] = match group.step_kind {
        ProfitabilityStepKind::MixedDirectBuyMintSell => &MIXED_BUY_MINT_SELL_SPECS,
        ProfitabilityStepKind::MixedDirectSellBuyMerge => &MIXED_SELL_BUY_MERGE_SPECS,
        ProfitabilityStepKind::ArbMintSell
        | ProfitabilityStepKind::ArbBuyMerge
        | ProfitabilityStepKind::PureDirectBuy
        | ProfitabilityStepKind::PureDirectSell
        | ProfitabilityStepKind::PureDirectMerge => return,
    };

    for spec in specs {
        let indices = collect_action_set_indices(actions, &group.action_indices, spec.kind);
        if indices.is_empty() {
            continue;
        }
        println!(
            "{}",
            ActionSetView {
                actions,
                global_offset,
                indices: &indices,
                style: spec.style,
            }
        );
    }
}

fn trade_preview_values(action: &Action) -> Option<(&'static str, &'static str, f64, f64, f64)> {
    match action {
        Action::Buy {
            market_name,
            amount,
            cost,
        } => {
            if amount.abs() <= 1e-12 {
                return None;
            }
            Some(("buy", *market_name, *amount, *cost, *cost / *amount))
        }
        Action::Sell {
            market_name,
            amount,
            proceeds,
        } => {
            if amount.abs() <= 1e-12 {
                return None;
            }
            Some((
                "sell",
                *market_name,
                *amount,
                *proceeds,
                *proceeds / *amount,
            ))
        }
        _ => None,
    }
}

fn print_flash_trade_preview(
    actions: &[Action],
    group: &ActionGroup,
    global_offset: usize,
    tick_boundary_hits: Option<&HashMap<usize, bool>>,
    ansi: bool,
) {
    struct SubgroupSummary {
        start_idx: usize,
        end_idx: usize,
        trade_indices: Vec<usize>,
        sum_p: Option<f64>,
        execution_price: Option<f64>,
        label: &'static str,
    }

    fn summarize_trades(
        actions: &[Action],
        trade_indices: &[usize],
    ) -> (Option<usize>, Option<usize>, usize, f64, f64, f64, f64) {
        let mut first_trade: Option<usize> = None;
        let mut last_trade: Option<usize> = None;
        let mut trade_count = 0usize;
        let mut total_units = 0.0f64;
        let mut total_quote = 0.0f64;
        let mut min_price = f64::INFINITY;
        let mut max_price = f64::NEG_INFINITY;

        for &action_idx in trade_indices {
            let Some((_, _, units, quote, unit_price)) = trade_preview_values(&actions[action_idx])
            else {
                continue;
            };
            if first_trade.is_none() {
                first_trade = Some(action_idx);
            }
            last_trade = Some(action_idx);
            trade_count += 1;
            total_units += units;
            total_quote += quote;
            min_price = min_price.min(unit_price);
            max_price = max_price.max(unit_price);
        }

        (
            first_trade,
            last_trade,
            trade_count,
            total_units,
            total_quote,
            min_price,
            max_price,
        )
    }

    let mut subgroups: Vec<SubgroupSummary> = Vec::new();
    let mut i = 0usize;
    while i < group.action_indices.len() {
        let idx = group.action_indices[i];
        let Action::FlashLoan { .. } = actions[idx] else {
            i += 1;
            continue;
        };

        let mut j = i + 1;
        let mut saw_mint = false;
        let mut saw_merge = false;
        let mut buys_in_bracket: Vec<usize> = Vec::new();
        let mut sells_in_bracket: Vec<usize> = Vec::new();

        while j < group.action_indices.len() {
            let inner_idx = group.action_indices[j];
            match actions[inner_idx] {
                Action::Mint { .. } => saw_mint = true,
                Action::Merge { .. } => saw_merge = true,
                Action::Buy { .. } => buys_in_bracket.push(inner_idx),
                Action::Sell { .. } => sells_in_bracket.push(inner_idx),
                Action::RepayFlashLoan { .. } => break,
                Action::FlashLoan { .. } => break,
            }
            j += 1;
        }

        if j >= group.action_indices.len()
            || !matches!(
                actions[group.action_indices[j]],
                Action::RepayFlashLoan { .. }
            )
        {
            i += 1;
            continue;
        }

        if saw_mint {
            let sum_p = if sells_in_bracket.is_empty() {
                None
            } else {
                Some(
                    sells_in_bracket
                        .iter()
                        .filter_map(|&k| match actions[k] {
                            Action::Sell {
                                amount, proceeds, ..
                            } if amount > 0.0 => Some(proceeds / amount),
                            _ => None,
                        })
                        .sum(),
                )
            };
            subgroups.push(SubgroupSummary {
                start_idx: idx,
                end_idx: group.action_indices[j],
                trade_indices: sells_in_bracket,
                sum_p,
                execution_price: sum_p.map(|v| 1.0 - v),
                label: "mint+sell trades",
            });
        } else if saw_merge {
            let sum_p = if buys_in_bracket.is_empty() {
                None
            } else {
                Some(
                    buys_in_bracket
                        .iter()
                        .filter_map(|&k| match actions[k] {
                            Action::Buy { amount, cost, .. } if amount > 0.0 => Some(cost / amount),
                            _ => None,
                        })
                        .sum(),
                )
            };
            subgroups.push(SubgroupSummary {
                start_idx: idx,
                end_idx: group.action_indices[j],
                trade_indices: buys_in_bracket,
                sum_p,
                execution_price: sum_p.map(|v| 1.0 - v),
                label: "buy+merge trades",
            });
        }

        i = j + 1;
    }

    if subgroups.is_empty() {
        println!("      flash-trade subgroup: unavailable (no flash bracket found)");
        return;
    }

    let print_subgroup = |sub_idx: usize, subgroup: &SubgroupSummary| {
        println!(
            "      subgroup[{sub_idx}] {} indices=[{}..={}] trades={}",
            subgroup.label,
            global_offset + subgroup.start_idx,
            global_offset + subgroup.end_idx,
            subgroup.trade_indices.len()
        );
        if let (Some(sum_p), Some(execution_price)) = (subgroup.sum_p, subgroup.execution_price) {
            println!(
                "        exec_px(net)={:>10.6} sum_p={:>10.6}",
                execution_price, sum_p
            );
        }

        let (first_trade, last_trade, trade_count, total_units, total_quote, min_price, max_price) =
            summarize_trades(actions, &subgroup.trade_indices);

        if let Some(action_idx) = first_trade
            && let Some((side, market_name, units, quote, unit_price)) =
                trade_preview_values(&actions[action_idx])
        {
            println!(
                "        first trade[{}]: {} {} units={:.6} quote={:.6} px={:.6}",
                global_offset + action_idx,
                side,
                display_market_name(market_name),
                units,
                quote,
                unit_price
            );
        }

        if let Some(action_idx) = last_trade
            && let Some((side, market_name, units, quote, unit_price)) =
                trade_preview_values(&actions[action_idx])
        {
            println!(
                "        last trade[{}]:  {} {} units={:.6} quote={:.6} px={:.6}",
                global_offset + action_idx,
                side,
                display_market_name(market_name),
                units,
                quote,
                unit_price
            );
        }

        if trade_count == 0 {
            println!("        price summary: unavailable (no buy/sell trades in subgroup)");
            return;
        }

        let avg_price = total_quote / total_units;
        let boundary_stats = collect_boundary_stats(
            actions,
            &subgroup.trade_indices,
            global_offset,
            tick_boundary_hits,
        );
        let boundary_badge = format_boundary_badge(boundary_stats, ansi)
            .unwrap_or_else(|| "[BOUNDARY n/a]".to_string());
        println!(
            "        price summary: trades={:>3} avg_px={:>10.6} min_px={:>10.6} max_px={:>10.6} total_units={:>12.6} total_quote={:>12.6} {}",
            trade_count, avg_price, min_price, max_price, total_units, total_quote, boundary_badge
        );
    };

    if subgroups.len() <= FLASH_SUBGROUP_PREVIEW_HEAD + FLASH_SUBGROUP_PREVIEW_TAIL {
        for (sub_idx, subgroup) in subgroups.iter().enumerate() {
            print_subgroup(sub_idx, subgroup);
        }
        return;
    }

    for (sub_idx, subgroup) in subgroups
        .iter()
        .take(FLASH_SUBGROUP_PREVIEW_HEAD)
        .enumerate()
    {
        print_subgroup(sub_idx, subgroup);
    }

    let tail_start = subgroups.len() - FLASH_SUBGROUP_PREVIEW_TAIL;
    let skipped = &subgroups[FLASH_SUBGROUP_PREVIEW_HEAD..tail_start];
    let skipped_trade_count: usize = skipped
        .iter()
        .map(|subgroup| subgroup.trade_indices.len())
        .sum();
    println!(
        "      ... skipped {} flash subgroups ({} trades)",
        skipped.len(),
        skipped_trade_count
    );

    for (sub_idx, subgroup) in subgroups.iter().enumerate().skip(tail_start) {
        print_subgroup(sub_idx, subgroup);
    }
}

fn print_direct_group_execution_price(
    actions: &[Action],
    group: &ActionGroup,
    global_offset: usize,
    tick_boundary_hits: Option<&HashMap<usize, bool>>,
    ansi: bool,
) {
    let mut total_units = 0.0_f64;
    let mut total_quote = 0.0_f64;
    let mut boundary_stats = BoundaryStats::default();
    for &idx in &group.action_indices {
        match (&group.kind, &actions[idx]) {
            (GroupKind::DirectBuy, Action::Buy { amount, cost, .. }) if *amount > 0.0 => {
                total_units += *amount;
                total_quote += *cost;
                if let Some(hit) =
                    trade_boundary_status(&actions[idx], global_offset + idx, tick_boundary_hits)
                {
                    boundary_stats.record(hit);
                }
            }
            (
                GroupKind::DirectSell,
                Action::Sell {
                    amount, proceeds, ..
                },
            ) if *amount > 0.0 => {
                total_units += *amount;
                total_quote += *proceeds;
                if let Some(hit) =
                    trade_boundary_status(&actions[idx], global_offset + idx, tick_boundary_hits)
                {
                    boundary_stats.record(hit);
                }
            }
            _ => {}
        }
    }
    if total_units > 0.0 {
        let execution_price = total_quote / total_units;
        let boundary_badge = format_boundary_badge(boundary_stats, ansi)
            .unwrap_or_else(|| "[BOUNDARY n/a]".to_string());
        println!(
            "      execution price={:>10.6} quote={:>12.6} units={:>12.6} {}",
            execution_price, total_quote, total_units, boundary_badge
        );
    }
}

struct GroupPreviewRow<'a> {
    group_index: usize,
    group: &'a ActionGroup,
    global_start: usize,
    global_end: usize,
    badges: &'a str,
}

impl fmt::Display for GroupPreviewRow<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let indices = format!("[{}..={}]", self.global_start, self.global_end);
        if self.badges.is_empty() {
            write!(
                f,
                "    #{:<6} {:<kind_w$} {:>7} {:<idx_w$}",
                self.group_index,
                group_display_label(self.group),
                self.group.action_indices.len(),
                indices,
                kind_w = GROUP_KIND_COL_WIDTH,
                idx_w = GROUP_INDICES_COL_WIDTH
            )
        } else {
            write!(
                f,
                "    #{:<6} {:<kind_w$} {:>7} {:<idx_w$} {}",
                self.group_index,
                group_display_label(self.group),
                self.group.action_indices.len(),
                indices,
                self.badges,
                kind_w = GROUP_KIND_COL_WIDTH,
                idx_w = GROUP_INDICES_COL_WIDTH
            )
        }
    }
}

fn print_action_group_preview(
    actions: &[Action],
    group_index: usize,
    group: &ActionGroup,
    global_offset: usize,
    tick_boundary_hits: Option<&HashMap<usize, bool>>,
    ansi: bool,
) {
    let Some(&local_start) = group.action_indices.first() else {
        return;
    };
    let Some(&local_end) = group.action_indices.last() else {
        return;
    };
    let global_start = global_offset + local_start;
    let global_end = global_offset + local_end;
    let badges = format_group_badges(actions, group, global_offset, tick_boundary_hits, ansi);
    println!(
        "{}",
        GroupPreviewRow {
            group_index,
            group,
            global_start,
            global_end,
            badges: &badges,
        }
    );
    if is_mixed_step(group) {
        print_mixed_group_outline(actions, group, global_offset);
    } else {
        println!(
            "      first [{}] {:<9} {}",
            global_start,
            action_kind_label(&actions[local_start]),
            action_outline_detail(&actions[local_start])
        );
        if local_end != local_start {
            println!(
                "      last  [{}] {:<9} {}",
                global_end,
                action_kind_label(&actions[local_end]),
                action_outline_detail(&actions[local_end])
            );
        }
    }
    if matches!(group.kind, GroupKind::DirectBuy | GroupKind::DirectSell) {
        print_direct_group_execution_price(actions, group, global_offset, tick_boundary_hits, ansi);
    } else if matches!(group.kind, GroupKind::MintSell | GroupKind::BuyMerge) {
        print_flash_trade_preview(actions, group, global_offset, tick_boundary_hits, ansi);
    }
}

fn print_skipped_group_rollup(
    actions: &[Action],
    groups: &[ActionGroup],
    start: usize,
    end: usize,
    global_offset: usize,
    tick_boundary_hits: Option<&HashMap<usize, bool>>,
    ansi: bool,
) {
    if start >= end || end > groups.len() {
        return;
    }
    let skipped = &groups[start..end];
    let skipped_group_count = skipped.len();
    let skipped_actions: usize = skipped.iter().map(|group| group.action_indices.len()).sum();
    let skipped_flash = skipped
        .iter()
        .filter(|group| matches!(group.kind, GroupKind::MintSell | GroupKind::BuyMerge))
        .count();
    let skipped_mixed = skipped.iter().filter(|group| is_mixed_step(group)).count();
    let mut boundary_stats = BoundaryStats::default();
    for group in skipped {
        boundary_stats.merge(collect_boundary_stats(
            actions,
            &group.action_indices,
            global_offset,
            tick_boundary_hits,
        ));
    }
    if let Some(boundary_badge) = format_boundary_badge(boundary_stats, ansi) {
        println!(
            "    ... skipped {} groups ({} actions, flash={}, mixed={}) {}",
            skipped_group_count, skipped_actions, skipped_flash, skipped_mixed, boundary_badge
        );
    } else {
        println!(
            "    ... skipped {} groups ({} actions, flash={}, mixed={})",
            skipped_group_count, skipped_actions, skipped_flash, skipped_mixed
        );
    }
}

fn print_compact_action_groups(
    label: &str,
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
    config: TraceConfig,
) {
    if actions.len() < LARGE_ACTION_PREVIEW_MIN_ACTIONS {
        return;
    }
    let ansi = config.ansi_enabled;

    println!(
        "[rebalance][{}] compact action groups ({} actions):",
        label,
        actions.len()
    );
    let tick_boundary_hits = replay_trade_boundary_hits(actions, slot0_results);
    let (arb_actions, post_arb_actions) = split_actions_by_complete_set_arb_phase(actions);
    let mut global_offset = 0usize;

    if !arb_actions.is_empty() {
        println!("  arb phase:");
        match group_actions_by_profitability_step(arb_actions) {
            Ok(groups) => {
                print_group_preview_header(ansi);
                for (idx, group) in groups.iter().enumerate() {
                    print_action_group_preview(
                        arb_actions,
                        idx,
                        group,
                        global_offset,
                        tick_boundary_hits.as_ref(),
                        ansi,
                    );
                }
            }
            Err(err) => {
                let end = arb_actions.len() - 1;
                println!(
                    "    unable to group arb phase ({err}); indices=[0..={}] actions={}",
                    end,
                    arb_actions.len()
                );
                println!(
                    "      first [{}] {:<9} {}",
                    0,
                    action_kind_label(&arb_actions[0]),
                    action_outline_detail(&arb_actions[0])
                );
                if end > 0 {
                    println!(
                        "      last  [{}] {:<9} {}",
                        end,
                        action_kind_label(&arb_actions[end]),
                        action_outline_detail(&arb_actions[end])
                    );
                }
            }
        }
        global_offset = arb_actions.len();
    }

    if post_arb_actions.is_empty() {
        return;
    }

    let groups = match group_actions_by_profitability_step(post_arb_actions) {
        Ok(groups) => groups,
        Err(err) => {
            let end = actions.len() - 1;
            println!(
                "  post-arb groups unavailable ({err}); indices=[{}..={}] actions={}",
                global_offset,
                end,
                post_arb_actions.len()
            );
            println!(
                "      first [{}] {:<9} {}",
                global_offset,
                action_kind_label(&post_arb_actions[0]),
                action_outline_detail(&post_arb_actions[0])
            );
            if post_arb_actions.len() > 1 {
                let last_global = global_offset + post_arb_actions.len() - 1;
                println!(
                    "      last  [{}] {:<9} {}",
                    last_global,
                    action_kind_label(&post_arb_actions[post_arb_actions.len() - 1]),
                    action_outline_detail(&post_arb_actions[post_arb_actions.len() - 1])
                );
            }
            return;
        }
    };

    println!("  post-arb phase groups: {}", groups.len());
    print_group_preview_header(ansi);
    if groups.len() <= ACTION_PREVIEW_HEAD_GROUPS + ACTION_PREVIEW_TAIL_GROUPS {
        for (idx, group) in groups.iter().enumerate() {
            print_action_group_preview(
                post_arb_actions,
                idx,
                group,
                global_offset,
                tick_boundary_hits.as_ref(),
                ansi,
            );
        }
        return;
    }

    for (idx, group) in groups.iter().take(ACTION_PREVIEW_HEAD_GROUPS).enumerate() {
        print_action_group_preview(
            post_arb_actions,
            idx,
            group,
            global_offset,
            tick_boundary_hits.as_ref(),
            ansi,
        );
    }
    let tail_start = groups.len() - ACTION_PREVIEW_TAIL_GROUPS;
    print_skipped_group_rollup(
        post_arb_actions,
        &groups,
        ACTION_PREVIEW_HEAD_GROUPS,
        tail_start,
        global_offset,
        tick_boundary_hits.as_ref(),
        ansi,
    );
    for (idx, group) in groups.iter().enumerate().skip(tail_start) {
        print_action_group_preview(
            post_arb_actions,
            idx,
            group,
            global_offset,
            tick_boundary_hits.as_ref(),
            ansi,
        );
    }
}

pub fn print_rebalance_execution_summary(
    label: &str,
    actions: &[Action],
    slot0_results: &[(Slot0Result, &'static MarketData)],
    config: TraceConfig,
) {
    print_compact_action_groups(label, actions, slot0_results, config);
    let Some(prices_before) = sorted_market_prices(slot0_results) else {
        print_trade_summary(label, actions);
        println!(
            "[rebalance][{}] market price deltas unavailable (snapshot could not be mapped to sims)",
            label
        );
        return;
    };
    let price_sum_before: f64 = prices_before.iter().map(|(_, p)| *p).sum();
    let (arb_actions, non_arb_actions) = split_actions_by_complete_set_arb_phase(actions);

    if !arb_actions.is_empty() {
        let arb_label = format!("{}_arb_phase", label);
        print_trade_summary(&arb_label, arb_actions);
        if let Some(slot0_after_arb) =
            try_replay_actions_to_market_state(arb_actions, slot0_results)
            && let Some(prices_after_arb) = sorted_market_prices(&slot0_after_arb)
        {
            let price_sum_after_arb: f64 = prices_after_arb.iter().map(|(_, p)| *p).sum();
            print_market_price_changes(&arb_label, &prices_before, &prices_after_arb, config);
            println!(
                "[rebalance][{}] market price sum before={:.9}",
                arb_label, price_sum_before
            );
            println!(
                "[rebalance][{}] market price sum after={:.9}",
                arb_label, price_sum_after_arb
            );

            let non_arb_label = format!("{}_non_arb_phase", label);
            print_trade_summary(&non_arb_label, non_arb_actions);
            if let Some(slot0_after_non_arb) =
                try_replay_actions_to_market_state(non_arb_actions, &slot0_after_arb)
                && let Some(prices_after_non_arb) = sorted_market_prices(&slot0_after_non_arb)
            {
                let price_sum_after_non_arb: f64 =
                    prices_after_non_arb.iter().map(|(_, p)| *p).sum();
                print_market_price_changes(
                    &non_arb_label,
                    &prices_after_arb,
                    &prices_after_non_arb,
                    config,
                );
                println!(
                    "[rebalance][{}] market price sum before={:.9}",
                    non_arb_label, price_sum_after_arb
                );
                println!(
                    "[rebalance][{}] market price sum after={:.9}",
                    non_arb_label, price_sum_after_non_arb
                );

                let total_label = format!("{}_total", label);
                print_trade_summary(&total_label, actions);
                println!(
                    "[rebalance][{}] market price sum before={:.9}",
                    total_label, price_sum_before
                );
                println!(
                    "[rebalance][{}] market price sum after={:.9}",
                    total_label, price_sum_after_non_arb
                );
                return;
            }
        }
    }

    print_trade_summary(label, actions);
    let Some(slot0_after) = try_replay_actions_to_market_state(actions, slot0_results) else {
        println!(
            "[rebalance][{}] market price deltas unavailable after replay",
            label
        );
        return;
    };
    if let Some(prices_after) = sorted_market_prices(&slot0_after) {
        let price_sum_after: f64 = prices_after.iter().map(|(_, p)| *p).sum();
        print_market_price_changes(label, &prices_before, &prices_after, config);
        println!(
            "[rebalance][{}] market price sum before={:.9}",
            label, price_sum_before
        );
        println!(
            "[rebalance][{}] market price sum after={:.9}",
            label, price_sum_after
        );
    } else {
        println!(
            "[rebalance][{}] market price deltas unavailable after replay",
            label
        );
    }
}

pub fn print_portfolio_snapshot(
    label: &str,
    stage: &str,
    holdings: &HashMap<&'static str, f64>,
    cash: f64,
    config: TraceConfig,
) {
    let non_zero_positions: Vec<(&'static str, f64)> = holdings
        .iter()
        .map(|(name, units)| (*name, *units))
        .filter(|(_, units)| units.abs() > 1e-12)
        .collect();

    println!(
        "[rebalance][{}] {} portfolio: cash={:.9}, non_zero_positions={}/{}",
        label,
        stage,
        cash,
        non_zero_positions.len(),
        holdings.len()
    );
    if non_zero_positions.is_empty() {
        println!("  (no non-zero holdings)");
        return;
    }

    let full_output = config.full_portfolio;
    if full_output || non_zero_positions.len() <= PORTFOLIO_PREVIEW_ROWS {
        let mut sorted = non_zero_positions;
        sorted.sort_by(|(lhs_name, _), (rhs_name, _)| lhs_name.cmp(rhs_name));
        for (name, units) in sorted {
            println!("  {}: {:.9}", name, units);
        }
        return;
    }

    let mut ranked = non_zero_positions;
    ranked.sort_by(|(lhs_name, lhs_units), (rhs_name, rhs_units)| {
        rhs_units
            .abs()
            .partial_cmp(&lhs_units.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs_name.cmp(rhs_name))
    });

    println!(
        "  showing top {} positions by |units| (set REBALANCE_FULL_PORTFOLIO=1 for full list):",
        PORTFOLIO_PREVIEW_ROWS
    );
    for (name, units) in ranked.iter().take(PORTFOLIO_PREVIEW_ROWS) {
        println!("  {}: {:.9}", name, units);
    }
    let omitted = ranked.len().saturating_sub(PORTFOLIO_PREVIEW_ROWS);
    println!("  ... omitted {} positions", omitted);
}

pub fn print_trade_summary(label: &str, actions: &[Action]) {
    let mut buy_count = 0usize;
    let mut buy_units = 0.0_f64;
    let mut buy_cost = 0.0_f64;
    let mut sell_count = 0usize;
    let mut sell_units = 0.0_f64;
    let mut sell_proceeds = 0.0_f64;
    let mut mint_count = 0usize;
    let mut mint_amount = 0.0_f64;
    let mut merge_count = 0usize;
    let mut merge_amount = 0.0_f64;
    let mut flash_count = 0usize;
    let mut flash_amount = 0.0_f64;
    let mut repay_count = 0usize;
    let mut repay_amount = 0.0_f64;

    for action in actions {
        match action {
            Action::Buy { amount, cost, .. } => {
                buy_count += 1;
                buy_units += *amount;
                buy_cost += *cost;
            }
            Action::Sell {
                amount, proceeds, ..
            } => {
                sell_count += 1;
                sell_units += *amount;
                sell_proceeds += *proceeds;
            }
            Action::Mint { amount, .. } => {
                mint_count += 1;
                mint_amount += *amount;
            }
            Action::Merge { amount, .. } => {
                merge_count += 1;
                merge_amount += *amount;
            }
            Action::FlashLoan { amount } => {
                flash_count += 1;
                flash_amount += *amount;
            }
            Action::RepayFlashLoan { amount } => {
                repay_count += 1;
                repay_amount += *amount;
            }
        }
    }

    println!("[rebalance][{}] trade summary:", label);
    println!("  actions: {}", actions.len());
    println!(
        "  buy: count={}, units={:.9}, cost={:.9}",
        buy_count, buy_units, buy_cost
    );
    println!(
        "  sell: count={}, units={:.9}, proceeds={:.9}",
        sell_count, sell_units, sell_proceeds
    );
    println!("  mint: count={}, amount={:.9}", mint_count, mint_amount);
    println!("  merge: count={}, amount={:.9}", merge_count, merge_amount);
    println!(
        "  flash_loan: count={}, amount={:.9}",
        flash_count, flash_amount
    );
    println!(
        "  repay_flash_loan: count={}, amount={:.9}",
        repay_count, repay_amount
    );
}
