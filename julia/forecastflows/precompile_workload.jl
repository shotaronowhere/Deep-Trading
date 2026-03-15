using ForecastFlows
using JSON3

const REPORT_PATH = normpath(joinpath(@__DIR__, "..", "..", "test", "fixtures", "rebalancer_ab_live_l1_snapshot_report.json"))

function wad_to_float(x)
    return Float64(x) / 1e18
end

function tiny_problem()
    outcomes = [
        OutcomeSpec("YES", 0.55, 0.0),
        OutcomeSpec("NO", 0.45, 0.0),
    ]
    markets = [
        UniV3MarketSpec(
            "warm-u1",
            "YES",
            0.5,
            [UniV3LiquidityBand(1.0, 10.0), UniV3LiquidityBand(0.25, 0.0)],
            0.9999,
        ),
        UniV3MarketSpec(
            "warm-u2",
            "NO",
            0.5,
            [UniV3LiquidityBand(1.0, 9.0), UniV3LiquidityBand(0.25, 0.0)],
            0.9999,
        ),
    ]
    return PredictionMarketProblem(outcomes, 1.0, markets; split_bound=nothing)
end

function representative_problem()
    report = JSON3.read(read(REPORT_PATH, String))
    count = length(report.predictions_wad)
    outcomes = Vector{OutcomeSpec}(undef, count)
    markets = Vector{UniV3MarketSpec}(undef, count)
    for index in 1:count
        outcome_id = "REP-$(lpad(string(index - 1), 3, '0'))"
        market_id = outcome_id
        fair_value = wad_to_float(report.predictions_wad[index])
        initial_holding = wad_to_float(report.initial_holdings_wad[index])
        current_price = wad_to_float(report.starting_prices_wad[index])
        liquidity_l = Float64(report.liquidity[index]) / 1e18
        buy_limit = clamp(max(current_price * 1.5, current_price + 1e-6), current_price + 1e-6, 1.0)
        sell_limit = max(current_price * 0.25, 1e-6)
        outcomes[index] = OutcomeSpec(outcome_id, fair_value, initial_holding)
        markets[index] = UniV3MarketSpec(
            market_id,
            outcome_id,
            current_price,
            [UniV3LiquidityBand(buy_limit, liquidity_l), UniV3LiquidityBand(sell_limit, 0.0)],
            0.9999,
        )
    end
    return PredictionMarketProblem(
        outcomes,
        wad_to_float(report.initial_cash_budget_wad),
        markets;
        split_bound=nothing,
    )
end

function live_solver_options()
    return (pgtol=1e-6, max_iter=2_500, max_fun=5_000)
end

function compare_request_json(problem, request_id)
    return JSON3.write(
        ForecastFlows.CompareRequest(
            problem;
            request_id=request_id,
            protocol_version=ForecastFlows.PREDICTION_MARKET_PROTOCOL_VERSION,
            certify=true,
            throw_on_fail=false,
            max_doublings=0,
            solver_options=live_solver_options(),
        ),
    )
end

function run_precompile_workload()
    tiny = tiny_problem()
    representative = representative_problem()
    solver_options = live_solver_options()

    ForecastFlows.handle_protocol_json(JSON3.write(ForecastFlows.HealthRequest(request_id="precompile-health")))
    compare_prediction_market_families(
        tiny;
        certify=true,
        throw_on_fail=false,
        max_doublings=0,
        solver_options=solver_options,
    )
    ForecastFlows.handle_protocol_json(compare_request_json(tiny, "precompile-tiny"))
    compare_prediction_market_families(
        representative;
        certify=true,
        throw_on_fail=false,
        max_doublings=0,
        solver_options=solver_options,
    )
    ForecastFlows.handle_protocol_json(compare_request_json(representative, "precompile-representative"))
    return nothing
end

run_precompile_workload()
