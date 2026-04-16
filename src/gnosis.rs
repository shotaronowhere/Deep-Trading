//! Gnosis chain (Swapr / AlgebraV1.9) movie futarchy market data.
//!
//! 15 scalar markets from the "Session 1 - Movies Experiment".
//! Each movie has an upToken and downToken with pools against its underlyingToken.
//! Data sourced from futarchy-ui/src/consts/markets.ts and data/movie_predictions.csv.

use alloy::primitives::{Address, address, keccak256};

// ── Gnosis chain constants ──

/// sDAI — collateral for all movie markets.
pub const SDAI: Address = address!("af204776c7245bF4147c2612BF6e5972Ee483701");

/// GnosisRouter (CTF) — splitPosition / mergePositions.
pub const GNOSIS_ROUTER: Address = address!("eC9048b59b3467415b1a38F63416407eA0c70fB8");

/// Swapr AlgebraV1.9 Router.
pub const SWAPR_ROUTER: Address = address!("fFB643E73f280B97809A8b41f7232AB401a04EE1");

/// Algebra Pool Deployer (CREATE2 factory).
pub const POOL_DEPLOYER: Address = address!("C1b576AC6Ec749d5Ace1787bF9Ec6340908ddB47");

/// Pool init code hash for CREATE2 derivation.
const POOL_INIT_CODE_HASH: [u8; 32] = [
    0xbc, 0xe3, 0x7a, 0x54, 0xea, 0xb2, 0xfc, 0xd7, 0x19, 0x13, 0xa0, 0xd4, 0x07, 0x23, 0xe0, 0x42,
    0x38, 0x97, 0x0e, 0x7f, 0xc1, 0x15, 0x9b, 0xfd, 0x58, 0xad, 0x5b, 0x79, 0x53, 0x16, 0x97, 0xe7,
];

/// Parent market — all movie outcomes are conditional on this.
pub const PARENT_MARKET: Address = address!("6f7ae2815e7e13c14a6560f4b382ae78e7b1493e");

// ── CREATE2 pool address derivation ──

/// Compute Swapr pool address for a token pair via CREATE2.
pub fn compute_pool_address(token_a: Address, token_b: Address) -> Address {
    // Sort tokens
    let (token0, token1) = if token_a < token_b {
        (token_a, token_b)
    } else {
        (token_b, token_a)
    };

    // salt = keccak256(abi.encode(token0, token1))
    let mut encoded = [0u8; 64];
    encoded[12..32].copy_from_slice(token0.as_slice());
    encoded[44..64].copy_from_slice(token1.as_slice());
    let salt = keccak256(&encoded);

    // CREATE2: keccak256(0xff ++ deployer ++ salt ++ initCodeHash)[12..]
    let mut create2_input = [0u8; 85];
    create2_input[0] = 0xff;
    create2_input[1..21].copy_from_slice(POOL_DEPLOYER.as_slice());
    create2_input[21..53].copy_from_slice(salt.as_slice());
    create2_input[53..85].copy_from_slice(&POOL_INIT_CODE_HASH);
    let hash = keccak256(&create2_input);
    Address::from_slice(&hash[12..])
}

// ── Movie market data ──

pub struct MovieMarket {
    pub name: &'static str,
    pub up_token: Address,
    pub down_token: Address,
    pub underlying_token: Address,
    pub market_id: Address,
    pub score: f64, // 0-100 percentile from CSV
}

impl MovieMarket {
    pub fn up_prediction(&self) -> f64 {
        self.score / 100.0
    }
    pub fn down_prediction(&self) -> f64 {
        1.0 - self.score / 100.0
    }
    pub fn up_pool(&self) -> Address {
        compute_pool_address(self.up_token, self.underlying_token)
    }
    pub fn down_pool(&self) -> Address {
        compute_pool_address(self.down_token, self.underlying_token)
    }
    pub fn up_is_token1(&self) -> bool {
        self.up_token > self.underlying_token
    }
    pub fn down_is_token1(&self) -> bool {
        self.down_token > self.underlying_token
    }
}

/// All 15 movie markets with predictions.
/// Source: futarchy-ui/src/consts/markets.ts + data/movie_predictions.csv
pub static MOVIES: [MovieMarket; 15] = [
    MovieMarket {
        name: "Judge Dredd (1995)",
        up_token: address!("0ee25eb2e22c01fa832dd5fea5637fba4cd5e870"),
        down_token: address!("4abea4bf9e35f4e957695374c388cee9f83ca1d0"),
        underlying_token: address!("b72a1271caa3d84d3fbbbcbb0f63ee358b94f96a"),
        score: 72.51,
        market_id: address!("105d957043ee12f7705efa072af11e718f8c5b83"),
    },
    MovieMarket {
        name: "Bacurau (2019)",
        up_token: address!("028ec9938471bbad5167c2e5281144a94d1acbe9"),
        down_token: address!("53f82c3f6836dcba9d35450d906286a6ea089a26"),
        underlying_token: address!("cb1f243baaf93199742e09dc98b16fc8b714b67c"),
        score: 86.08,
        market_id: address!("68af0afe82dda5c9c26e6a458a143caad35708d6"),
    },
    MovieMarket {
        name: "The Hitchhiker's Guide to the Galaxy (2005)",
        up_token: address!("ad2248b8eaa3e3a405c1ba79dd436947f8b427df"),
        down_token: address!("dd510abc6a848662371c3455717949035cc24019"),
        underlying_token: address!("fb06c25e59302d8a0318d6df41a2f29deeea1c8a"),
        score: 80.93,
        market_id: address!("fdd8af90af2722d5fe39adf1002fbd069b8a76c0"),
    },
    MovieMarket {
        name: "Everything, Everywhere, All At Once (2022)",
        up_token: address!("fa020fcd05e0b91dae83a2a08c5b5533edf8c851"),
        down_token: address!("372d0798ffe8c3aa982a15258c0fea22c6a768df"),
        underlying_token: address!("e85d556d1aaae2f6027336e468e9c981251a4bef"),
        score: 87.00,
        market_id: address!("1f2e76d66047e7f8e0deea373a0c04ffecab31df"),
    },
    MovieMarket {
        name: "12 angry men (1957)",
        up_token: address!("7ee3806d16dc6a76bef2b11880b70cc70f74fa1a"),
        down_token: address!("34f8572eab463606a014c37ff68b78ac9361cacc"),
        underlying_token: address!("b3933fd994af5db7ae985a0d62ed2dda918a839b"),
        score: 92.04,
        market_id: address!("2338ca7d59b7e15bd03dd81cf5f5bb59b6c6c6d4"),
    },
    MovieMarket {
        name: "Alien (1979)",
        up_token: address!("37e70bae5e87327feece73a7c227446571f92137"),
        down_token: address!("31e3d82a613e5aeea7c3a65c3d657cacaaaf2674"),
        underlying_token: address!("6d0407b5ae419fdd92ffdc64abf04c5f28950e02"),
        score: 90.00,
        market_id: address!("9a274ea86665d872fc58c8f26fd97a18b844c6ac"),
    },
    MovieMarket {
        name: "Demolition Man (1993)",
        up_token: address!("53a9011c5570bfb8148954c4f49a6625dc44077b"),
        down_token: address!("64974d3bf944fafec6fa19a900f3679a716b3a86"),
        underlying_token: address!("20025021e440edd39d486f3c6a1d7adb9c269faf"),
        score: 79.07,
        market_id: address!("c25af7d4a5cb36bb3ce9faf652a5f7f989a1d57a"),
    },
    MovieMarket {
        name: "Barbie (2023)",
        up_token: address!("aed0fad91e7149ec84bb4d0a2a77be819169275f"),
        down_token: address!("044e1b6d8aacbda5699423578bd200484f7473c3"),
        underlying_token: address!("67d0f938ea12e7e30b8ccc24dd031d656cc3927d"),
        score: 82.06,
        market_id: address!("d31d05158722f64b6a49e25bccc47d3203eecbe9"),
    },
    MovieMarket {
        name: "Eduardo e M\u{00f4}nica (2020)",
        up_token: address!("9d64a3e7e55880f3c8f9c584ed32397bb6f0b9f6"),
        down_token: address!("e9d025d3cbd783d6a92626b650a32f7cbaca0e7d"),
        underlying_token: address!("58ce7a53abeca1db90cec0e6b7dcbe3a36d986c4"),
        score: 70.00,
        market_id: address!("13d48a73811c01f574e1bfa4c58b7d95d2f590e4"),
    },
    MovieMarket {
        name: "Thor: The Dark World (2013)",
        up_token: address!("0c569fbc021119b778ea160efd718a5d592ef46c"),
        down_token: address!("d8d2dfe1912239451b5a4a0462006e95393f2151"),
        underlying_token: address!("72ec9aade867b5b41705c6a83f66bc56485669b5"),
        score: 72.02,
        market_id: address!("878a332b5efc0a4bf983036beece050352baa73d"),
    },
    MovieMarket {
        name: "Talk to me (2022)",
        up_token: address!("f3c17e909bd1f9367ecdc786d137465d7ee96b6a"),
        down_token: address!("f99be182b6b0e6d994509ecdced281b94100435f"),
        underlying_token: address!("2b3a8ac53ba42da13f542a867d2859642fb1db44"),
        score: 81.04,
        market_id: address!("ee4a77447069f32f555f3d75aaba18a4acb54ac4"),
    },
    MovieMarket {
        name: "Fast & Furious 6 (2013)",
        up_token: address!("850d2ffa4475296cfbbd76247894a773e3b1be6c"),
        down_token: address!("b28c716f63b0dd272f62e25765a914baeebab8c2"),
        underlying_token: address!("71c3df5edcab48cfb6a1a99255eff063f33b6265"),
        score: 81.02,
        market_id: address!("38a2923cc391b9cd926e5a2d07462dc7d189c407"),
    },
    MovieMarket {
        name: "Elysium (2013)",
        up_token: address!("e9427a7a32daad2d29db2aad809b2a44060d8fc8"),
        down_token: address!("75b5cd86828f7c9009e30619a83b1b2da67f1342"),
        underlying_token: address!("f52e0e144b73a0d5748bc53667efe3ba62fe5695"),
        score: 78.12,
        market_id: address!("c0dab34c6c2008391bdc742cec0bd0afb60d4d59"),
    },
    MovieMarket {
        name: "Session 9 (2001)",
        up_token: address!("e080c03ad6bc9f8fd5b45b5d3bf14ebcfa1ec0b5"),
        down_token: address!("76cce8491785789c2c5542f043ec6c35b12cd909"),
        underlying_token: address!("1086a95c224dd586809a7f4d875b4f09d2ac9290"),
        score: 81.10,
        market_id: address!("a7cf69c4c93d2f6811a394e92320979c3cf86b37"),
    },
    MovieMarket {
        name: "Mamma Mia! (2008)",
        up_token: address!("fa82984fc8ddeb71fdb2e6e471f30995178ad5f0"),
        down_token: address!("5d528dbec7e37927d8af41bfb1b54e7641dd3ccb"),
        underlying_token: address!("11ed86c399f455819f495cda1256e9b52afd0971"),
        score: 72.04,
        market_id: address!("96638d67ac5bc5f8223f9e2d60e92f4d8dcf3147"),
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::providers::Provider;

    #[test]
    fn test_movie_count() {
        assert_eq!(MOVIES.len(), 15);
    }

    #[test]
    fn test_predictions_match_movies() {
        for movie in MOVIES.iter() {
            let up = movie.up_prediction();
            let down = movie.down_prediction();
            assert!(
                up > 0.0 && up < 1.0,
                "{}: up_prediction {} out of range",
                movie.name,
                up
            );
            let sum = up + down;
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "{}: predictions don't sum to 1.0 (got {})",
                movie.name,
                sum
            );
        }
    }

    #[test]
    fn test_pool_addresses_are_distinct() {
        let mut seen = std::collections::HashSet::new();
        for movie in MOVIES.iter() {
            let up_pool = movie.up_pool();
            assert!(seen.insert(up_pool), "duplicate up pool {up_pool}");
            let down_pool = movie.down_pool();
            assert!(seen.insert(down_pool), "duplicate down pool {down_pool}");
        }
    }

    #[test]
    fn test_is_token1_consistency() {
        for movie in MOVIES.iter() {
            assert_eq!(
                movie.up_is_token1(),
                movie.up_token > movie.underlying_token,
                "{}: up_is_token1 mismatch",
                movie.name
            );
            assert_eq!(
                movie.down_is_token1(),
                movie.down_token > movie.underlying_token,
                "{}: down_is_token1 mismatch",
                movie.name
            );
        }
    }

    #[tokio::test]
    async fn test_pool_existence_on_gnosis() {
        dotenvy::dotenv().ok();
        let rpc = match std::env::var("RPC_GNOSIS") {
            Ok(url) => url,
            Err(_) => {
                eprintln!("RPC_GNOSIS not set — skipping");
                return;
            }
        };
        let provider = alloy::providers::ProviderBuilder::new()
            .with_reqwest(rpc.parse().unwrap(), |b| b.no_proxy().build().unwrap());

        let mut up_active = 0;
        let mut down_active = 0;

        for movie in MOVIES.iter() {
            let up_pool = movie.up_pool();
            let code = provider.get_code_at(up_pool).await.unwrap();
            if !code.is_empty() {
                up_active += 1;
                println!("  UP  pool OK: {} -> {}", movie.name, up_pool);
            } else {
                println!("  UP  pool MISSING: {} -> {}", movie.name, up_pool);
            }

            let down_pool = movie.down_pool();
            let code = provider.get_code_at(down_pool).await.unwrap();
            if !code.is_empty() {
                down_active += 1;
                println!("  DOWN pool OK: {} -> {}", movie.name, down_pool);
            } else {
                println!("  DOWN pool MISSING: {} -> {}", movie.name, down_pool);
            }
        }

        println!("\nActive: {up_active} up pools, {down_active} down pools out of 15 each");
        assert_eq!(up_active, 15, "expected all 15 up pools to exist");
        assert_eq!(down_active, 15, "expected all 15 down pools to exist");
    }
}
