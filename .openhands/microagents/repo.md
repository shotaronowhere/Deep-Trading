# Deep-Trading Repository

This is a Deep Trading Bot project written in Rust. It rebalances portfolio according to predictions across a set of markets, specifically targeting deep funding markets.

## Project Structure

- **src/**: Main Rust source code
  - `main.rs`: Entry point
  - `lib.rs`: Core library functionality
  - `markets.rs`: Market data (large file with market information)
  - `predictions.rs`: Trading predictions
  - `pools.rs`: Pool management
  - `execution/`: Execution logic
  - `pools/`: Pool implementations
  - `portfolio/`: Portfolio management
- **contracts/**: Solidity smart contracts (BatchRouter.sol)
- **test/**: Test files (Solidity tests with Foundry)
- **docs/**: Documentation files including architecture, portfolio, slippage, and model details

## Running the Project

- **Build**: `cargo build`
- **Run**: `cargo run`
- **Tests (Rust)**: `cargo test`
- **Tests (Solidity/Foundry)**: Tests are in `test/` directory using Foundry framework

## Key Dependencies

The project uses Alloy (Ethereum library), Tokio (async runtime), reqwest (HTTP client), and uniswap_v3_math for DEX calculations.

## Development Notes

- Configuration is managed via `.env.dev` file
- The project uses CSV files for predictions data (l1-predictions.csv, l2-predictions.csv, originality-predictions.csv)
- JSON market data files are also included (markets_data_l1.json, markets_data_l2.json, markets_data_originality.json)
