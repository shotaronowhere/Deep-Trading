#!/usr/bin/env julia

function parse_args(args)
    fixture_path = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--fixture" && i < length(args)
            fixture_path = args[i + 1]
            i += 2
            continue
        end
        i += 1
    end
    return fixture_path
end

function parse_number_list(raw)
    body = strip(raw)
    isempty(body) && return Float64[]
    values = Float64[]
    for token in split(body, ",")
        s = strip(token)
        isempty(s) && continue
        push!(values, parse(Float64, s))
    end
    return values
end

function extract_array(text, key)
    pat = Regex("\\\"$(key)\\\"\\s*:\\s*\\[(.*?)\\]", "s")
    m = match(pat, text)
    m === nothing && error("missing array key: $(key)")
    return parse_number_list(m.captures[1])
end

function extract_number(text, key)
    pat = Regex(
        "\\\"$(key)\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
    )
    m = match(pat, text)
    m === nothing && error("missing numeric key: $(key)")
    return parse(Float64, m.captures[1])
end

function extract_bool(text, key)
    pat = Regex("\\\"$(key)\\\"\\s*:\\s*(true|false)")
    m = match(pat, text)
    m === nothing && error("missing bool key: $(key)")
    return m.captures[1] == "true"
end

function heuristic_route(prices, predictions, holdings, cash, allow_complete_set)
    n = length(prices)
    buys = zeros(Float64, n)
    sells = zeros(Float64, n)
    objective = 0.0

    cash_remaining = max(cash, 0.0)
    for i in eachindex(prices)
        px = prices[i]
        pred = predictions[i]
        if !(isfinite(px) && isfinite(pred)) || px <= 0.0
            continue
        end

        edge = pred - px
        if edge > 1e-9
            budget_cap = cash_remaining / max(px, 1e-9)
            amount = min(max(edge * 8.0, 0.0), max(budget_cap, 0.0))
            buys[i] = max(amount, 0.0)
            objective += edge * buys[i]
            cash_remaining -= buys[i] * px
        elseif edge < -1e-9
            max_sell = i <= length(holdings) ? max(holdings[i], 0.0) : 0.0
            amount = min(max_sell, max(-edge * 8.0, 0.0))
            sells[i] = max(amount, 0.0)
            objective += (-edge) * sells[i]
            cash_remaining += sells[i] * px
        end
    end

    theta = 0.0
    if allow_complete_set
        price_sum = sum(prices)
        if price_sum > 1.0 + 1e-6
            theta = min(0.1 * max(cash_remaining, 0.0), price_sum - 1.0)
        elseif price_sum < 1.0 - 1e-6 && !isempty(holdings)
            theta = -min(minimum(max.(holdings, 0.0)), 1.0 - price_sum)
        end
    end

    return buys, sells, theta, objective
end

function json_escape(s)
    replace(replace(String(s), "\\" => "\\\\"), "\"" => "\\\"")
end

function emit_json(status, message, objective, buys, sells, theta)
    buys_text = "[" * join(string.(buys), ",") * "]"
    sells_text = "[" * join(string.(sells), ",") * "]"
    message_text = message === nothing ? "null" : "\"" * json_escape(message) * "\""
    objective_text = objective === nothing ? "null" : string(objective)
    println(
        "{" *
        "\"status\":\"$(json_escape(status))\"," *
        "\"message\":$(message_text)," *
        "\"objective\":$(objective_text)," *
        "\"buys\":$(buys_text)," *
        "\"sells\":$(sells_text)," *
        "\"theta\":$(theta)" *
        "}",
    )
end

function main()
    fixture_path = parse_args(ARGS)
    fixture_path === nothing && error("usage: route_fixture.jl --fixture <path>")

    raw = read(fixture_path, String)
    prices = extract_array(raw, "prices")
    predictions = extract_array(raw, "predictions")
    holdings = extract_array(raw, "holdings")
    cash = extract_number(raw, "cash")
    allow_complete_set = extract_bool(raw, "allow_complete_set")

    n = length(prices)
    if n != length(predictions)
        error("fixture dimension mismatch: prices=$(n), predictions=$(length(predictions))")
    end
    if length(holdings) != n
        holdings = vcat(holdings, zeros(Float64, max(n - length(holdings), 0)))
        holdings = holdings[1:n]
    end

    status = "heuristic_fallback"
    message = nothing

    cfmm_loaded = false
    cfmm_message = nothing
    try
        @eval import CFMMRouter
        cfmm_loaded = true
    catch err
        cfmm_message = sprint(showerror, err)
    end

    # Placeholder mapping: route_fixture currently emits deterministic heuristic flows.
    # When parity backend is promoted, this branch should map fixture data to the
    # CFMMRouter.jl Router/route!/find_arb! APIs and return routed flows.
    buys, sells, theta, objective =
        heuristic_route(prices, predictions, holdings, cash, allow_complete_set)

    if cfmm_loaded
        status = "cfmmrouter_loaded_heuristic"
        message = "CFMMRouter detected; using temporary heuristic fixture mapping"
    else
        status = "cfmmrouter_unavailable"
        message = cfmm_message
    end

    emit_json(status, message, objective, buys, sells, theta)
end

try
    main()
catch err
    emit_json("error", sprint(showerror, err), nothing, Float64[], Float64[], 0.0)
    exit(1)
end
