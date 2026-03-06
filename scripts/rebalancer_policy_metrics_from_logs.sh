#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/rebalancer_policy_metrics_from_logs.sh \
    --policy-a-log /path/to/constant_vs_exact.log \
    --policy-b-log /path/to/mixed_vs_direct.log \
    [--fallback-log /path/to/fallback_trace.log] \
    [--policy-a-case CASE_NAME] \
    [--include-direct-only-policy-b-cases] \
    [--date YYYY-MM-DD] \
    [--block BLOCK_REF] \
    [--gas-price-gwei FLOAT] \
    [--eth-usd FLOAT] \
    [--out-json /path/to/out.json] \
    [--append-csv /path/to/dashboard.csv]
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "file not found: $path"
}

csv_escape() {
  local s="$1"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

to_susd_from_wei() {
  local wei="$1"
  local v
  v="$(echo "scale=18; $wei / 1000000000000000000" | bc -l)"
  if [[ "$v" == .* ]]; then
    echo "0$v"
  elif [[ "$v" == -.* ]]; then
    echo "-0${v#-}"
  else
    echo "$v"
  fi
}

to_susd_from_gas() {
  local gas_units="$1"
  local gas_price_gwei="$2"
  local eth_usd="$3"
  local v
  v="$(echo "scale=18; ($gas_units * $gas_price_gwei / 1000000000) * $eth_usd" | bc -l)"
  if [[ "$v" == .* ]]; then
    echo "0$v"
  elif [[ "$v" == -.* ]]; then
    echo "-0${v#-}"
  else
    echo "$v"
  fi
}

max_float() {
  local a="$1"
  local b="$2"
  if [[ "$(echo "$a >= $b" | bc -l)" -eq 1 ]]; then
    echo "$a"
  else
    echo "$b"
  fi
}

bool_from_bc() {
  local expr="$1"
  if [[ "$(echo "$expr" | bc -l)" -eq 1 ]]; then
    echo "true"
  else
    echo "false"
  fi
}

median_from_values() {
  if [[ "$#" -eq 0 ]]; then
    echo ""
    return 0
  fi
  printf '%s\n' "$@" | sort -n | awk '
    { a[NR] = $1 }
    END {
      n = NR
      if (n % 2 == 1) {
        printf "%.18f\n", a[(n + 1) / 2]
      } else {
        printf "%.18f\n", (a[n / 2] + a[n / 2 + 1]) / 2.0
      }
    }
  '
}

p10_from_values() {
  if [[ "$#" -eq 0 ]]; then
    echo ""
    return 0
  fi
  printf '%s\n' "$@" | sort -n | awk '
    { a[NR] = $1 }
    END {
      n = NR
      idx = int((0.10 * n) + 0.999999)
      if (idx < 1) idx = 1
      if (idx > n) idx = n
      printf "%.18f\n", a[idx]
    }
  '
}

parse_forge_logs_to_tsv() {
  local in_file="$1"
  local out_file="$2"
  awk '
    BEGIN { active_test = ""; active_case = "" }

    /^\[PASS\] / {
      test_line = $0
      sub(/^\[PASS\] /, "", test_line)
      sub(/\(\).*/, "", test_line)
      active_test = test_line
      next
    }

    /^  [^:]+$/ {
      candidate = substr($0, 3)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", candidate)
      if (candidate == "" || candidate == "Logs") next
      active_case = candidate
      next
    }

    /^  [A-Za-z0-9_]+: / {
      line = substr($0, 3)
      split(line, parts, ": ")
      key = parts[1]
      value = parts[2]
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      case_name = active_case
      if (case_name == "") case_name = active_test
      if (case_name == "") case_name = "default_case"
      print case_name "\t" key "\t" value
      next
    }
  ' "$in_file" > "$out_file"
}

parse_mixed_fallbacks_to_tsv() {
  local in_file="$1"
  local cases_file="$2"
  local out_file="$3"
  awk '
    BEGIN {
      while ((getline line < ARGV[2]) > 0) {
        included[line] = 1
      }
      close(ARGV[2])
      ARGV[2] = ""
      active_case = ""
    }

    /^\[PASS\] / {
      active_case = ""
      next
    }

    /MixedSolveFallback\([0-9]+\)/ {
      if (active_case == "") next
      reason = $0
      sub(/^.*MixedSolveFallback\(/, "", reason)
      sub(/\).*$/, "", reason)
      print active_case "\t" reason
      next
    }

    /^  [^:]+$/ {
      candidate = substr($0, 3)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", candidate)
      if (candidate == "" || candidate == "Logs") next
      if (candidate in included) {
        active_case = candidate
      } else {
        active_case = ""
      }
      next
    }
  ' "$in_file" "$cases_file" > "$out_file"
}

list_cases() {
  local tsv="$1"
  awk -F '\t' '{print $1}' "$tsv" | sort -u
}

metric_for() {
  local tsv="$1"
  local case_name="$2"
  local key="$3"
  awk -F '\t' -v c="$case_name" -v k="$key" '$1==c && $2==k {print $3; exit}' "$tsv"
}

has_metric() {
  local tsv="$1"
  local case_name="$2"
  local key="$3"
  [[ -n "$(metric_for "$tsv" "$case_name" "$key")" ]]
}

POLICY_A_LOG=""
POLICY_B_LOG=""
FALLBACK_LOG=""
POLICY_A_CASE=""
INCLUDE_DIRECT_ONLY_POLICY_B=0
DATE_UTC="$(date -u +%F)"
BLOCK_REF=""
GAS_PRICE_GWEI="1.0"
ETH_USD="3000.0"
OUT_JSON=""
APPEND_CSV=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --policy-a-log)
      POLICY_A_LOG="$2"
      shift 2
      ;;
    --policy-b-log)
      POLICY_B_LOG="$2"
      shift 2
      ;;
    --fallback-log)
      FALLBACK_LOG="$2"
      shift 2
      ;;
    --policy-a-case)
      POLICY_A_CASE="$2"
      shift 2
      ;;
    --include-direct-only-policy-b-cases)
      INCLUDE_DIRECT_ONLY_POLICY_B=1
      shift
      ;;
    --date)
      DATE_UTC="$2"
      shift 2
      ;;
    --block)
      BLOCK_REF="$2"
      shift 2
      ;;
    --gas-price-gwei)
      GAS_PRICE_GWEI="$2"
      shift 2
      ;;
    --eth-usd)
      ETH_USD="$2"
      shift 2
      ;;
    --out-json)
      OUT_JSON="$2"
      shift 2
      ;;
    --append-csv)
      APPEND_CSV="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$POLICY_A_LOG" || -n "$POLICY_B_LOG" ]] || die "at least one of --policy-a-log or --policy-b-log is required"
[[ -n "$POLICY_A_LOG" ]] && require_file "$POLICY_A_LOG"
[[ -n "$POLICY_B_LOG" ]] && require_file "$POLICY_B_LOG"
[[ -n "$FALLBACK_LOG" ]] && require_file "$FALLBACK_LOG"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

policy_a_tsv=""
policy_b_tsv=""

if [[ -n "$POLICY_A_LOG" ]]; then
  policy_a_tsv="$tmp_dir/policy_a.tsv"
  parse_forge_logs_to_tsv "$POLICY_A_LOG" "$policy_a_tsv"
fi

if [[ -n "$POLICY_B_LOG" ]]; then
  policy_b_tsv="$tmp_dir/policy_b.tsv"
  parse_forge_logs_to_tsv "$POLICY_B_LOG" "$policy_b_tsv"
fi

pick_policy_a_case() {
  local selected="$POLICY_A_CASE"
  if [[ -n "$selected" ]]; then
    echo "$selected"
    return 0
  fi

  local c
  for c in $(list_cases "$policy_a_tsv"); do
    if has_metric "$policy_a_tsv" "$c" "constant_ev" && has_metric "$policy_a_tsv" "$c" "exact_ev"; then
      echo "$c"
      return 0
    fi
  done
  for c in $(list_cases "$policy_a_tsv"); do
    if has_metric "$policy_a_tsv" "$c" "onchain_constant_ev" && has_metric "$policy_a_tsv" "$c" "onchain_exact_ev"; then
      echo "$c"
      return 0
    fi
  done
  echo ""
}

POLICY_A_SELECTED_CASE=""
if [[ -n "$policy_a_tsv" ]]; then
  POLICY_A_SELECTED_CASE="$(pick_policy_a_case)"
fi

policy_a_constant_ev_wei=""
policy_a_exact_ev_wei=""
policy_a_constant_gas=""
policy_a_exact_gas=""
policy_a_ev_gain_wei=""
policy_a_ev_gain_susd=""
policy_a_extra_gas_susd=""
policy_a_hurdle_susd=""
policy_a_hurdle_met=""
policy_a_trigger_now=""

if [[ -n "$POLICY_A_SELECTED_CASE" ]]; then
  if has_metric "$policy_a_tsv" "$POLICY_A_SELECTED_CASE" "constant_ev"; then
    policy_a_constant_ev_wei="$(metric_for "$policy_a_tsv" "$POLICY_A_SELECTED_CASE" "constant_ev")"
    policy_a_exact_ev_wei="$(metric_for "$policy_a_tsv" "$POLICY_A_SELECTED_CASE" "exact_ev")"
    policy_a_constant_gas="$(metric_for "$policy_a_tsv" "$POLICY_A_SELECTED_CASE" "constant_gas")"
    policy_a_exact_gas="$(metric_for "$policy_a_tsv" "$POLICY_A_SELECTED_CASE" "exact_gas")"
  else
    policy_a_constant_ev_wei="$(metric_for "$policy_a_tsv" "$POLICY_A_SELECTED_CASE" "onchain_constant_ev")"
    policy_a_exact_ev_wei="$(metric_for "$policy_a_tsv" "$POLICY_A_SELECTED_CASE" "onchain_exact_ev")"
  fi

  if [[ -n "$policy_a_constant_ev_wei" && -n "$policy_a_exact_ev_wei" ]]; then
    policy_a_ev_gain_wei="$(echo "$policy_a_exact_ev_wei - $policy_a_constant_ev_wei" | bc)"
    policy_a_ev_gain_susd="$(to_susd_from_wei "$policy_a_ev_gain_wei")"
  fi

  if [[ -n "$policy_a_constant_gas" && -n "$policy_a_exact_gas" ]]; then
    extra_gas_units="$(echo "$policy_a_exact_gas - $policy_a_constant_gas" | bc)"
    policy_a_extra_gas_susd="$(to_susd_from_gas "$extra_gas_units" "$GAS_PRICE_GWEI" "$ETH_USD")"
    policy_a_hurdle_susd="$(max_float "0.10" "$(echo "scale=18; 3 * $policy_a_extra_gas_susd" | bc -l)")"
    policy_a_hurdle_met="$(bool_from_bc "$policy_a_ev_gain_susd >= $policy_a_hurdle_susd")"
    policy_a_trigger_now="$policy_a_hurdle_met"
  fi
fi

policy_b_cases_file="$tmp_dir/policy_b_cases.txt"
policy_b_ev_file="$tmp_dir/policy_b_ev_gains.txt"
policy_b_gas_file="$tmp_dir/policy_b_extra_gas.txt"
policy_b_rows_file="$tmp_dir/policy_b_rows.tsv"
> "$policy_b_cases_file"
> "$policy_b_ev_file"
> "$policy_b_gas_file"
> "$policy_b_rows_file"

if [[ -n "$policy_b_tsv" ]]; then
  for c in $(list_cases "$policy_b_tsv"); do
    rebalancer_ev="$(metric_for "$policy_b_tsv" "$c" "rebalancer_ev_wei")"
    mixed_ev="$(metric_for "$policy_b_tsv" "$c" "rebalancer_mixed_ev_wei")"
    rebalancer_gas="$(metric_for "$policy_b_tsv" "$c" "rebalancer_gas")"
    mixed_gas="$(metric_for "$policy_b_tsv" "$c" "rebalancer_mixed_gas")"
    if [[ -z "$rebalancer_ev" || -z "$mixed_ev" || -z "$rebalancer_gas" || -z "$mixed_gas" ]]; then
      continue
    fi
    if [[ "$INCLUDE_DIRECT_ONLY_POLICY_B" -ne 1 && "$c" == *"direct_only"* ]]; then
      continue
    fi

    ev_gain_wei="$(echo "$mixed_ev - $rebalancer_ev" | bc)"
    ev_gain_susd="$(to_susd_from_wei "$ev_gain_wei")"
    extra_gas_units="$(echo "$mixed_gas - $rebalancer_gas" | bc)"
    extra_gas_susd="$(to_susd_from_gas "$extra_gas_units" "$GAS_PRICE_GWEI" "$ETH_USD")"

    echo "$c" >> "$policy_b_cases_file"
    echo "$ev_gain_susd" >> "$policy_b_ev_file"
    echo "$extra_gas_susd" >> "$policy_b_gas_file"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$rebalancer_ev" "$mixed_ev" "$ev_gain_susd" "$rebalancer_gas" "$mixed_gas" >> "$policy_b_rows_file"
  done
fi

policy_b_case_count="$(wc -l < "$policy_b_cases_file" | tr -d ' ')"
policy_b_median_ev_gain_susd=""
policy_b_p10_ev_gain_susd=""
policy_b_median_extra_gas_susd=""
policy_b_hurdle_susd=""
policy_b_hurdle_met=""
policy_b_negative_tail_breach=""
policy_b_enable_now=""

if [[ "$policy_b_case_count" -gt 0 ]]; then
  # shellcheck disable=SC2207
  ev_values=( $(cat "$policy_b_ev_file") )
  # shellcheck disable=SC2207
  gas_values=( $(cat "$policy_b_gas_file") )
  policy_b_median_ev_gain_susd="$(median_from_values "${ev_values[@]}")"
  policy_b_p10_ev_gain_susd="$(p10_from_values "${ev_values[@]}")"
  policy_b_median_extra_gas_susd="$(median_from_values "${gas_values[@]}")"
  policy_b_hurdle_susd="$(max_float "0.20" "$(echo "scale=18; 5 * $policy_b_median_extra_gas_susd" | bc -l)")"
  policy_b_hurdle_met="$(bool_from_bc "$policy_b_median_ev_gain_susd >= $policy_b_hurdle_susd")"
  policy_b_negative_tail_breach="$(bool_from_bc "$policy_b_p10_ev_gain_susd < -0.05")"
fi

fallback_total=0
fallback_dominant_reason=""
fallback_dominant_count=0
fallback_rate_pct=""
fallback_dominant_share_pct=""
fallback_reliability_met="false"
fallback_dominance_met="false"

if [[ -n "$FALLBACK_LOG" && "$policy_b_case_count" -gt 0 ]]; then
  fallback_events_tsv="$tmp_dir/fallback_events.tsv"
  fallback_counts_tsv="$tmp_dir/fallback_counts.tsv"
  parse_mixed_fallbacks_to_tsv "$FALLBACK_LOG" "$policy_b_cases_file" "$fallback_events_tsv"
  awk -F '\t' '
    { c[$2]++ }
    END {
      for (k in c) print k "\t" c[k]
    }
  ' "$fallback_events_tsv" > "$fallback_counts_tsv"

  if [[ -s "$fallback_counts_tsv" ]]; then
    fallback_total="$(awk '{s+=$2} END {print s+0}' "$fallback_counts_tsv")"
    fallback_dominant_reason="$(sort -k2,2nr "$fallback_counts_tsv" | awk 'NR==1 {print $1}')"
    fallback_dominant_count="$(sort -k2,2nr "$fallback_counts_tsv" | awk 'NR==1 {print $2}')"
  fi
  fallback_rate_pct="$(echo "scale=6; ($fallback_total * 100.0) / $policy_b_case_count" | bc -l)"
  if [[ "$fallback_total" -gt 0 ]]; then
    fallback_dominant_share_pct="$(echo "scale=6; ($fallback_dominant_count * 100.0) / $fallback_total" | bc -l)"
  else
    fallback_dominant_share_pct="0"
  fi
  fallback_reliability_met="$(bool_from_bc "$fallback_rate_pct <= 20")"
  fallback_dominance_met="$(bool_from_bc "$fallback_dominant_share_pct <= 60")"
fi

if [[ -n "$policy_b_hurdle_met" && -n "$policy_b_negative_tail_breach" && -n "$fallback_rate_pct" && -n "$fallback_dominant_share_pct" ]]; then
  if [[ "$policy_b_hurdle_met" == "true" && "$policy_b_negative_tail_breach" == "false" && "$fallback_reliability_met" == "true" && "$fallback_dominance_met" == "true" ]]; then
    policy_b_enable_now="true"
  else
    policy_b_enable_now="false"
  fi
elif [[ -n "$policy_b_hurdle_met" && -n "$policy_b_negative_tail_breach" ]]; then
  policy_b_enable_now="false"
fi

echo "Rebalancer policy metrics summary"
echo "  date_utc: $DATE_UTC"
echo "  block_ref: ${BLOCK_REF:-n/a}"
echo "  gas_price_gwei: $GAS_PRICE_GWEI"
echo "  eth_usd: $ETH_USD"
echo "  policy_a_case: ${POLICY_A_SELECTED_CASE:-n/a}"
echo "  policy_a_ev_gain_susd: ${policy_a_ev_gain_susd:-n/a}"
echo "  policy_a_extra_gas_susd: ${policy_a_extra_gas_susd:-n/a}"
echo "  policy_a_hurdle_susd: ${policy_a_hurdle_susd:-n/a}"
echo "  policy_a_hurdle_met: ${policy_a_hurdle_met:-n/a}"
echo "  policy_a_trigger_now: ${policy_a_trigger_now:-n/a}"
echo "  policy_b_case_count: $policy_b_case_count"
echo "  policy_b_median_ev_gain_susd: ${policy_b_median_ev_gain_susd:-n/a}"
echo "  policy_b_p10_ev_gain_susd: ${policy_b_p10_ev_gain_susd:-n/a}"
echo "  policy_b_median_extra_gas_susd: ${policy_b_median_extra_gas_susd:-n/a}"
echo "  policy_b_hurdle_susd: ${policy_b_hurdle_susd:-n/a}"
echo "  policy_b_hurdle_met: ${policy_b_hurdle_met:-n/a}"
echo "  policy_b_negative_tail_breach: ${policy_b_negative_tail_breach:-n/a}"
echo "  fallback_total: $fallback_total"
echo "  fallback_rate_pct: ${fallback_rate_pct:-n/a}"
echo "  fallback_dominant_reason: ${fallback_dominant_reason:-n/a}"
echo "  fallback_dominant_share_pct: ${fallback_dominant_share_pct:-n/a}"
echo "  policy_b_enable_now: ${policy_b_enable_now:-n/a}"

if [[ -n "$OUT_JSON" ]]; then
  mkdir -p "$(dirname "$OUT_JSON")"

  cases_json_file="$tmp_dir/policy_b_cases.json.rows"
  > "$cases_json_file"
  if [[ -s "$policy_b_rows_file" ]]; then
    first=1
    while IFS=$'\t' read -r c rebalancer_ev mixed_ev ev_gain_susd rebalancer_gas mixed_gas; do
      extra_gas_units="$(echo "$mixed_gas - $rebalancer_gas" | bc)"
      extra_gas_susd="$(to_susd_from_gas "$extra_gas_units" "$GAS_PRICE_GWEI" "$ETH_USD")"
      if [[ "$first" -eq 0 ]]; then
        echo "," >> "$cases_json_file"
      fi
      first=0
      cat >> "$cases_json_file" <<EOF
    {
      "case": "${c}",
      "rebalancer_ev_wei": "${rebalancer_ev}",
      "mixed_ev_wei": "${mixed_ev}",
      "ev_gain_susd": "${ev_gain_susd}",
      "rebalancer_gas": ${rebalancer_gas},
      "mixed_gas": ${mixed_gas},
      "extra_gas_susd": "${extra_gas_susd}"
    }
EOF
    done < "$policy_b_rows_file"
  fi

  cat > "$OUT_JSON" <<EOF
{
  "date_utc": "${DATE_UTC}",
  "block_ref": "${BLOCK_REF}",
  "gas_price_gwei": "${GAS_PRICE_GWEI}",
  "eth_usd": "${ETH_USD}",
  "policy_a": {
    "case": "${POLICY_A_SELECTED_CASE}",
    "constant_ev_wei": "${policy_a_constant_ev_wei}",
    "exact_ev_wei": "${policy_a_exact_ev_wei}",
    "ev_gain_wei": "${policy_a_ev_gain_wei}",
    "ev_gain_susd": "${policy_a_ev_gain_susd}",
    "constant_gas": "${policy_a_constant_gas}",
    "exact_gas": "${policy_a_exact_gas}",
    "extra_gas_susd": "${policy_a_extra_gas_susd}",
    "hurdle_susd": "${policy_a_hurdle_susd}",
    "hurdle_met": "${policy_a_hurdle_met}",
    "trigger_now": "${policy_a_trigger_now}"
  },
  "policy_b": {
    "case_count": ${policy_b_case_count},
    "median_ev_gain_susd": "${policy_b_median_ev_gain_susd}",
    "p10_ev_gain_susd": "${policy_b_p10_ev_gain_susd}",
    "median_extra_gas_susd": "${policy_b_median_extra_gas_susd}",
    "hurdle_susd": "${policy_b_hurdle_susd}",
    "hurdle_met": "${policy_b_hurdle_met}",
    "negative_tail_breach": "${policy_b_negative_tail_breach}",
    "fallback_total": ${fallback_total},
    "fallback_rate_pct": "${fallback_rate_pct}",
    "fallback_dominant_reason": "${fallback_dominant_reason}",
    "fallback_dominant_share_pct": "${fallback_dominant_share_pct}",
    "fallback_reliability_met": "${fallback_reliability_met}",
    "fallback_dominance_met": "${fallback_dominance_met}",
    "enable_now": "${policy_b_enable_now}",
    "cases": [
$(cat "$cases_json_file")
    ]
  }
}
EOF
fi

if [[ -n "$APPEND_CSV" ]]; then
  mkdir -p "$(dirname "$APPEND_CSV")"
  if [[ ! -f "$APPEND_CSV" ]]; then
    cat > "$APPEND_CSV" <<'EOF'
"date_utc","block_ref","policy_a_case","policy_a_constant_ev_wei","policy_a_exact_ev_wei","policy_a_ev_gain_susd","policy_a_constant_gas","policy_a_exact_gas","policy_a_extra_gas_susd","policy_a_hurdle_susd","policy_a_hurdle_met","policy_a_trigger_now","policy_b_case_count","policy_b_median_ev_gain_susd","policy_b_p10_ev_gain_susd","policy_b_median_extra_gas_susd","policy_b_hurdle_susd","policy_b_hurdle_met","policy_b_negative_tail_breach","fallback_total","fallback_rate_pct","fallback_dominant_reason","fallback_dominant_share_pct","fallback_reliability_met","fallback_dominance_met","policy_b_enable_now","gas_price_gwei","eth_usd"
EOF
  fi

  {
    csv_escape "$DATE_UTC"; printf ","
    csv_escape "$BLOCK_REF"; printf ","
    csv_escape "$POLICY_A_SELECTED_CASE"; printf ","
    csv_escape "$policy_a_constant_ev_wei"; printf ","
    csv_escape "$policy_a_exact_ev_wei"; printf ","
    csv_escape "$policy_a_ev_gain_susd"; printf ","
    csv_escape "$policy_a_constant_gas"; printf ","
    csv_escape "$policy_a_exact_gas"; printf ","
    csv_escape "$policy_a_extra_gas_susd"; printf ","
    csv_escape "$policy_a_hurdle_susd"; printf ","
    csv_escape "$policy_a_hurdle_met"; printf ","
    csv_escape "$policy_a_trigger_now"; printf ","
    csv_escape "$policy_b_case_count"; printf ","
    csv_escape "$policy_b_median_ev_gain_susd"; printf ","
    csv_escape "$policy_b_p10_ev_gain_susd"; printf ","
    csv_escape "$policy_b_median_extra_gas_susd"; printf ","
    csv_escape "$policy_b_hurdle_susd"; printf ","
    csv_escape "$policy_b_hurdle_met"; printf ","
    csv_escape "$policy_b_negative_tail_breach"; printf ","
    csv_escape "$fallback_total"; printf ","
    csv_escape "$fallback_rate_pct"; printf ","
    csv_escape "$fallback_dominant_reason"; printf ","
    csv_escape "$fallback_dominant_share_pct"; printf ","
    csv_escape "$fallback_reliability_met"; printf ","
    csv_escape "$fallback_dominance_met"; printf ","
    csv_escape "$policy_b_enable_now"; printf ","
    csv_escape "$GAS_PRICE_GWEI"; printf ","
    csv_escape "$ETH_USD"; printf "\n"
  } >> "$APPEND_CSV"
fi
