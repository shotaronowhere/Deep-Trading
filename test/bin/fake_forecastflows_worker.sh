#!/bin/sh

scenario="$1"

extract_field() {
    printf '%s\n' "$1" | sed -n "s/.*\"$2\":\"\\([^\"]*\\)\".*/\\1/p"
}

health_response() {
    request_id="$1"
    cat <<EOF
{"protocol_version":2,"request_id":"$request_id","ok":true,"command":"health","result":{"status":"ok","supported_commands":["health","solve_prediction_market","compare_prediction_market_families"],"supported_modes":["direct_only","mixed_enabled"],"execution_model":"stateless NDJSON; one request at a time per worker process"}}
EOF
}

compare_response() {
    request_id="$1"
    case "$scenario" in
        healthy_direct)
            cat <<EOF
{"protocol_version":2,"request_id":"$request_id","ok":true,"command":"compare_prediction_market_families","result":{"direct_only":{"status":"certified","mode":"direct_only","certificate":{"passed":true},"trades":[{"market_id":"M1","outcome_id":"0x1111111111111111111111111111111111111111","collateral_delta":-0.01,"outcome_delta":0.05}],"split_merge":{"mint":0.0,"merge":0.0},"final_ev":999999.0},"mixed_enabled":{"status":"uncertified","mode":"mixed_enabled","certificate":{"passed":false},"trades":[],"split_merge":{"mint":0.0,"merge":0.0},"final_ev":0.0}}}
EOF
            ;;
        healthy_mixed)
            cat <<EOF
{"protocol_version":2,"request_id":"$request_id","ok":true,"command":"compare_prediction_market_families","result":{"direct_only":{"status":"uncertified","mode":"direct_only","certificate":{"passed":false},"trades":[],"split_merge":{"mint":0.0,"merge":0.0}},"mixed_enabled":{"status":"certified","mode":"mixed_enabled","certificate":{"passed":true},"trades":[{"market_id":"M2","outcome_id":"0x2222222222222222222222222222222222222222","collateral_delta":0.12,"outcome_delta":-0.2},{"market_id":"M1","outcome_id":"0x1111111111111111111111111111111111111111","collateral_delta":-0.03,"outcome_delta":0.04}],"split_merge":{"mint":0.2,"merge":0.0},"final_ev":1.0}}}
EOF
            ;;
        no_op_huge_ev)
            cat <<EOF
{"protocol_version":2,"request_id":"$request_id","ok":true,"command":"compare_prediction_market_families","result":{"direct_only":{"status":"certified","mode":"direct_only","certificate":{"passed":true},"trades":[],"split_merge":{"mint":0.0,"merge":0.0},"initial_ev":0.0,"final_ev":1000000.0},"mixed_enabled":{"status":"uncertified","mode":"mixed_enabled","certificate":{"passed":false},"trades":[],"split_merge":{"mint":0.0,"merge":0.0}}}}
EOF
            ;;
        uncertified_only)
            cat <<EOF
{"protocol_version":2,"request_id":"$request_id","ok":true,"command":"compare_prediction_market_families","result":{"direct_only":{"status":"uncertified","mode":"direct_only","certificate":{"passed":false},"trades":[],"split_merge":{"mint":0.0,"merge":0.0}},"mixed_enabled":{"status":"uncertified","mode":"mixed_enabled","certificate":{"passed":false},"trades":[],"split_merge":{"mint":0.0,"merge":0.0}}}}
EOF
            ;;
        invalid_mint_merge)
            cat <<EOF
{"protocol_version":2,"request_id":"$request_id","ok":true,"command":"compare_prediction_market_families","result":{"direct_only":{"status":"uncertified","mode":"direct_only","certificate":{"passed":false},"trades":[],"split_merge":{"mint":0.0,"merge":0.0}},"mixed_enabled":{"status":"certified","mode":"mixed_enabled","certificate":{"passed":true},"trades":[],"split_merge":{"mint":0.1,"merge":0.1}}}}
EOF
            ;;
        worker_error)
            cat <<EOF
{"protocol_version":2,"request_id":"$request_id","ok":false,"error":{"code":"invalid_request","message":"synthetic worker error"}}
EOF
            ;;
        malformed)
            printf '{not-json\n'
            ;;
        timeout)
            sleep 1
            ;;
        stderr_closed)
            printf 'stderr line one\n' >&2
            printf 'stderr line two\n' >&2
            exit 0
            ;;
        closed)
            exit 0
            ;;
        *)
            cat <<EOF
{"protocol_version":2,"request_id":"$request_id","ok":false,"error":{"code":"unknown_scenario","message":"unknown fake worker scenario $scenario"}}
EOF
            ;;
    esac
}

while IFS= read -r line; do
    command=$(extract_field "$line" command)
    request_id=$(extract_field "$line" request_id)

    if [ "$command" = "health" ]; then
        health_response "$request_id"
        continue
    fi

    if [ "$command" = "compare_prediction_market_families" ]; then
        compare_response "$request_id"
        continue
    fi

    cat <<EOF
{"protocol_version":2,"request_id":"$request_id","ok":false,"error":{"code":"unsupported","message":"unsupported command $command"}}
EOF
done
