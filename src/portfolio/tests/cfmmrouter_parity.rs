use super::super::cfmmrouter_bridge::{CfmmrouterFixture, run_cfmmrouter_cli};
use super::build_three_sims_with_preds;

#[test]
#[ignore = "optional parity harness against Julia CFMMRouter.jl"]
fn cfmmrouter_parity() {
    if std::env::var("RUN_CFMMROUTER_PARITY")
        .map(|value| value != "1")
        .unwrap_or(true)
    {
        println!(
            "[cfmmrouter-parity] skipping (set RUN_CFMMROUTER_PARITY=1 to enable parity harness)"
        );
        return;
    }

    let sims = build_three_sims_with_preds([0.13, 0.09, 0.27], [0.16, 0.07, 0.22]);
    let fixture = CfmmrouterFixture {
        prices: sims.iter().map(|sim| sim.price()).collect(),
        predictions: sims.iter().map(|sim| sim.prediction).collect(),
        holdings: vec![0.35, 0.0, 0.9],
        cash: 28.0,
        allow_complete_set: false,
    };

    let result = run_cfmmrouter_cli(&fixture)
        .expect("CFMMRouter CLI bridge should return JSON result when parity is enabled");

    assert_eq!(result.buys.len(), fixture.prices.len());
    assert_eq!(result.sells.len(), fixture.prices.len());
    assert!(result.theta.is_finite());
    assert!(result
        .buys
        .iter()
        .chain(result.sells.iter())
        .all(|value| value.is_finite() && *value >= 0.0));

    println!(
        "[cfmmrouter-parity] status={} objective={:?} theta={:.12}",
        result.status, result.objective, result.theta
    );
}
