use std::fmt;

use crate::portfolio::Action;

use super::GroupKind;

#[derive(Debug, Clone)]
pub struct ActionGroup {
    pub kind: GroupKind,
    pub action_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupingError {
    UnexpectedActionOutsideGroup {
        index: usize,
    },
    NestedFlashLoan {
        index: usize,
    },
    MissingRepayFlashLoan {
        flash_loan_index: usize,
    },
    InvalidFlashLoanBracket {
        start_index: usize,
        end_index: usize,
    },
    UnsupportedFlashLoanFee {
        start_index: usize,
        end_index: usize,
    },
}

impl fmt::Display for GroupingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedActionOutsideGroup { index } => {
                write!(
                    f,
                    "unsupported action shape outside execution groups at index {index}"
                )
            }
            Self::NestedFlashLoan { index } => {
                write!(f, "nested flash loan bracket at index {index}")
            }
            Self::MissingRepayFlashLoan { flash_loan_index } => {
                write!(
                    f,
                    "missing repay flash loan for bracket starting at index {flash_loan_index}"
                )
            }
            Self::InvalidFlashLoanBracket {
                start_index,
                end_index,
            } => write!(
                f,
                "unsupported flash-loan bracket shape between indices {start_index} and {end_index}"
            ),
            Self::UnsupportedFlashLoanFee {
                start_index,
                end_index,
            } => write!(
                f,
                "flash-loan fee is not supported between indices {start_index} and {end_index}"
            ),
        }
    }
}

impl std::error::Error for GroupingError {}

/// Groups optimizer actions into execution units supported by the strict executor.
///
/// Supported shapes:
/// - `DirectBuy`, `DirectSell`, `DirectMerge`
/// - `MintSell` flash bracket: `FlashLoan -> Mint -> Sell+ -> RepayFlashLoan`
/// - `BuyMerge` flash bracket: `FlashLoan -> Buy+ -> Merge -> RepayFlashLoan`
///
/// Any unsupported shape returns an error (fail closed) so callers can skip submission.
pub fn group_actions(actions: &[Action]) -> Result<Vec<ActionGroup>, GroupingError> {
    let mut groups = Vec::new();
    let mut i = 0usize;

    while i < actions.len() {
        match &actions[i] {
            Action::Buy { .. } => {
                groups.push(ActionGroup {
                    kind: GroupKind::DirectBuy,
                    action_indices: vec![i],
                });
                i += 1;
            }
            Action::Sell { .. } => {
                groups.push(ActionGroup {
                    kind: GroupKind::DirectSell,
                    action_indices: vec![i],
                });
                i += 1;
            }
            Action::Merge { .. } => {
                groups.push(ActionGroup {
                    kind: GroupKind::DirectMerge,
                    action_indices: vec![i],
                });
                i += 1;
            }
            Action::FlashLoan { .. } => {
                let start = i;
                i += 1;
                let mut end: Option<usize> = None;
                while i < actions.len() {
                    match &actions[i] {
                        Action::FlashLoan { .. } => {
                            return Err(GroupingError::NestedFlashLoan { index: i });
                        }
                        Action::RepayFlashLoan { .. } => {
                            end = Some(i);
                            break;
                        }
                        _ => {
                            i += 1;
                        }
                    }
                }

                let Some(end_index) = end else {
                    return Err(GroupingError::MissingRepayFlashLoan {
                        flash_loan_index: start,
                    });
                };

                let kind = classify_flash_loan_bracket(actions, start, end_index)?;
                let action_indices = (start..=end_index).collect();
                groups.push(ActionGroup {
                    kind,
                    action_indices,
                });
                i = end_index + 1;
            }
            Action::Mint { .. } | Action::RepayFlashLoan { .. } => {
                return Err(GroupingError::UnexpectedActionOutsideGroup { index: i });
            }
        }
    }

    Ok(groups)
}

fn classify_flash_loan_bracket(
    actions: &[Action],
    start_index: usize,
    end_index: usize,
) -> Result<GroupKind, GroupingError> {
    const ZERO_FEE_REL_TOL: f64 = 1e-8;

    let borrowed = match actions.get(start_index) {
        Some(Action::FlashLoan { amount }) => *amount,
        _ => {
            return Err(GroupingError::InvalidFlashLoanBracket {
                start_index,
                end_index,
            });
        }
    };
    let repaid = match actions.get(end_index) {
        Some(Action::RepayFlashLoan { amount }) => *amount,
        _ => {
            return Err(GroupingError::InvalidFlashLoanBracket {
                start_index,
                end_index,
            });
        }
    };
    let tol = ZERO_FEE_REL_TOL * (1.0 + borrowed.abs() + repaid.abs());
    if !borrowed.is_finite()
        || !repaid.is_finite()
        || borrowed < 0.0
        || repaid < 0.0
        || (borrowed - repaid).abs() > tol
    {
        return Err(GroupingError::UnsupportedFlashLoanFee {
            start_index,
            end_index,
        });
    }

    let inner = &actions[(start_index + 1)..end_index];

    if let Some((first, rest)) = inner.split_first() {
        if matches!(first, Action::Mint { .. })
            && !rest.is_empty()
            && rest.iter().all(|a| matches!(a, Action::Sell { .. }))
        {
            return Ok(GroupKind::MintSell);
        }
    }

    if let Some((last, rest)) = inner.split_last() {
        if matches!(last, Action::Merge { .. })
            && !rest.is_empty()
            && rest.iter().all(|a| matches!(a, Action::Buy { .. }))
        {
            return Ok(GroupKind::BuyMerge);
        }
    }

    Err(GroupingError::InvalidFlashLoanBracket {
        start_index,
        end_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buy(name: &'static str, amount: f64, cost: f64) -> Action {
        Action::Buy {
            market_name: name,
            amount,
            cost,
        }
    }

    fn sell(name: &'static str, amount: f64, proceeds: f64) -> Action {
        Action::Sell {
            market_name: name,
            amount,
            proceeds,
        }
    }

    #[test]
    fn groups_direct_and_bracket_actions() {
        let actions = vec![
            buy("a", 1.0, 2.0),
            Action::FlashLoan { amount: 10.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "t",
            },
            sell("b", 10.0, 4.0),
            Action::RepayFlashLoan { amount: 10.0 },
            Action::FlashLoan { amount: 5.0 },
            buy("x", 5.0, 1.5),
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 5.0,
                source_market: "x",
            },
            Action::RepayFlashLoan { amount: 5.0 },
            sell("z", 1.0, 1.0),
        ];

        let groups = group_actions(&actions).expect("grouping should succeed");
        assert_eq!(groups.len(), 4);
        assert_eq!(groups[0].kind, GroupKind::DirectBuy);
        assert_eq!(groups[1].kind, GroupKind::MintSell);
        assert_eq!(groups[2].kind, GroupKind::BuyMerge);
        assert_eq!(groups[3].kind, GroupKind::DirectSell);
    }

    #[test]
    fn errors_when_flash_loan_bracket_is_unclosed() {
        let actions = vec![Action::FlashLoan { amount: 1.0 }, buy("a", 1.0, 1.0)];
        let err = group_actions(&actions).expect_err("missing repay must error");
        assert_eq!(
            err,
            GroupingError::MissingRepayFlashLoan {
                flash_loan_index: 0
            }
        );
    }

    #[test]
    fn errors_when_mint_appears_outside_bracket() {
        let actions = vec![Action::Mint {
            contract_1: "c1",
            contract_2: "c2",
            amount: 1.0,
            target_market: "m",
        }];
        let err = group_actions(&actions).expect_err("standalone mint should error");
        assert_eq!(
            err,
            GroupingError::UnexpectedActionOutsideGroup { index: 0 }
        );
    }

    #[test]
    fn groups_standalone_merge_as_direct_merge() {
        let actions = vec![Action::Merge {
            contract_1: "c1",
            contract_2: "c2",
            amount: 1.0,
            source_market: "s",
        }];
        let groups = group_actions(&actions).expect("standalone merge should group");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].kind, GroupKind::DirectMerge);
        assert_eq!(groups[0].action_indices, vec![0]);
    }

    #[test]
    fn errors_on_flash_arb_pattern_without_mint_or_merge() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            buy("a", 1.0, 0.8),
            sell("b", 1.0, 0.9),
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let err = group_actions(&actions).expect_err("unsupported flash arb must error");
        assert_eq!(
            err,
            GroupingError::InvalidFlashLoanBracket {
                start_index: 0,
                end_index: 3
            }
        );
    }

    #[test]
    fn errors_when_mint_bracket_contains_buy_leg() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "t",
            },
            buy("a", 1.0, 0.8),
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let err = group_actions(&actions).expect_err("mint bracket with buy leg must error");
        assert_eq!(
            err,
            GroupingError::InvalidFlashLoanBracket {
                start_index: 0,
                end_index: 3
            }
        );
    }

    #[test]
    fn errors_when_buy_merge_bracket_contains_sell_leg() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            buy("a", 1.0, 0.8),
            sell("b", 1.0, 0.9),
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 1.0,
                source_market: "s",
            },
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let err = group_actions(&actions).expect_err("buy-merge bracket with sell leg must error");
        assert_eq!(
            err,
            GroupingError::InvalidFlashLoanBracket {
                start_index: 0,
                end_index: 4
            }
        );
    }

    #[test]
    fn errors_when_mint_sell_bracket_is_out_of_order() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            sell("a", 1.0, 0.8),
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "t",
            },
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let err = group_actions(&actions).expect_err("out-of-order mint-sell bracket must error");
        assert_eq!(
            err,
            GroupingError::InvalidFlashLoanBracket {
                start_index: 0,
                end_index: 3
            }
        );
    }

    #[test]
    fn errors_when_buy_merge_bracket_is_out_of_order() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            Action::Merge {
                contract_1: "c1",
                contract_2: "c2",
                amount: 1.0,
                source_market: "s",
            },
            buy("a", 1.0, 0.8),
            Action::RepayFlashLoan { amount: 10.0 },
        ];
        let err = group_actions(&actions).expect_err("out-of-order buy-merge bracket must error");
        assert_eq!(
            err,
            GroupingError::InvalidFlashLoanBracket {
                start_index: 0,
                end_index: 3
            }
        );
    }

    #[test]
    fn errors_when_flash_loan_fee_is_non_zero() {
        let actions = vec![
            Action::FlashLoan { amount: 10.0 },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: 10.0,
                target_market: "t",
            },
            sell("a", 10.0, 7.0),
            Action::RepayFlashLoan { amount: 10.001 },
        ];
        let err = group_actions(&actions).expect_err("non-zero flash-loan fee must error");
        assert_eq!(
            err,
            GroupingError::UnsupportedFlashLoanFee {
                start_index: 0,
                end_index: 3
            }
        );
    }

    #[test]
    fn accepts_flash_loan_fee_at_tolerance_boundary() {
        const ZERO_FEE_REL_TOL: f64 = 1e-8;
        let borrowed = 10.0;
        // Solve d = tol * (1 + |borrowed| + |borrowed + d|) exactly for boundary d.
        let boundary_diff = ZERO_FEE_REL_TOL * (1.0 + 2.0 * borrowed) / (1.0 - ZERO_FEE_REL_TOL);
        let actions = vec![
            Action::FlashLoan { amount: borrowed },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: borrowed,
                target_market: "t",
            },
            sell("a", borrowed, 7.0),
            Action::RepayFlashLoan {
                amount: borrowed + boundary_diff,
            },
        ];
        let groups = group_actions(&actions).expect("boundary fee tolerance should be accepted");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].kind, GroupKind::MintSell);
    }

    #[test]
    fn rejects_flash_loan_fee_just_above_tolerance_boundary() {
        const ZERO_FEE_REL_TOL: f64 = 1e-8;
        let borrowed = 10.0;
        let boundary_diff = ZERO_FEE_REL_TOL * (1.0 + 2.0 * borrowed) / (1.0 - ZERO_FEE_REL_TOL);
        let actions = vec![
            Action::FlashLoan { amount: borrowed },
            Action::Mint {
                contract_1: "c1",
                contract_2: "c2",
                amount: borrowed,
                target_market: "t",
            },
            sell("a", borrowed, 7.0),
            Action::RepayFlashLoan {
                amount: borrowed + boundary_diff * (1.0 + 1e-6),
            },
        ];
        let err = group_actions(&actions).expect_err("fee just above tolerance should be rejected");
        assert_eq!(
            err,
            GroupingError::UnsupportedFlashLoanFee {
                start_index: 0,
                end_index: 3
            }
        );
    }
}
