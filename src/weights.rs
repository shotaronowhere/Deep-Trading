// Auto-generated from weights.csv - do not edit manually

#[derive(Debug, Clone, Copy)]
pub struct Weight {
    pub market: &'static str,
    pub prediction: f64,
}

pub static WEIGHTS: [Weight; 2] = [
    Weight { market: "a16z/helios", prediction: 0.01363775945 },
    Weight { market: "ethereum/go-ethereum", prediction: 0.02 },
];
