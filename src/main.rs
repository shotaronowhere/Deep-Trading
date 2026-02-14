use alloy::{
    primitives::{U256, address},
    providers::{ProviderBuilder, WsConnect},
    sol,
};

sol! {
    #[sol(rpc)]
    contract Market {
        function wrappedOutcome(uint256 index) external view returns (address wrapped1155, bytes memory data);
        function outcomes(uint256) external view returns (string memory);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to an Ethereum node via WebSocket
    let ws = WsConnect::new("wss://optimism.drpc.org");
    let provider = ProviderBuilder::new().connect_ws(ws).await?;
    // Setup the Market contract instance
    let market_add = address!("0xf93c838e4b5dc163320ca1bd2d23a4f59ad6e57c");
    let market = Market::new(market_add, &provider);

    // Example: Fetch wrapped outcome for index 0
    let outcome = market.outcomes(U256::from(0)).call().await?;
    println!("Outcome: {:?}", outcome);

    Ok(())
}
