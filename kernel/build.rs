//! Generates the model config consts from ../models.toml (PGT_MODEL selects
//! the entry), so adding a same-architecture model never touches Rust code.
use std::{env, fs, path::Path};

fn main() {
    println!("cargo:rerun-if-env-changed=PGT_MODEL");
    println!("cargo:rerun-if-changed=../models.toml");

    let key = env::var("PGT_MODEL")
        .expect("set PGT_MODEL to a key from models.toml (e.g. PGT_MODEL=all-minilm)");
    let raw = fs::read_to_string("../models.toml").expect("../models.toml not found");
    let reg: toml::Table = raw.parse().expect("invalid models.toml");
    let mut m = reg
        .get(&key)
        .unwrap_or_else(|| panic!("model '{key}' not in models.toml"))
        .as_table()
        .unwrap()
        .clone();
    // resolve `base = "<key>"` inheritance (local fields override)
    if let Some(base) = m.get("base").and_then(|v| v.as_str()).map(String::from) {
        let b = reg
            .get(&base)
            .unwrap_or_else(|| panic!("{key}.base = '{base}' not in models.toml"))
            .as_table()
            .unwrap();
        for (bk, bv) in b {
            m.entry(bk.clone()).or_insert_with(|| bv.clone());
        }
    }

    let int = |k: &str| -> i64 {
        m.get(k)
            .unwrap_or_else(|| panic!("{key}.{k} missing in models.toml"))
            .as_integer()
            .unwrap_or_else(|| panic!("{key}.{k} must be an integer"))
    };
    let flt = |k: &str| -> f64 {
        match m.get(k).unwrap_or_else(|| panic!("{key}.{k} missing")) {
            toml::Value::Float(f) => *f,
            toml::Value::Integer(i) => *i as f64,
            _ => panic!("{key}.{k} must be a number"),
        }
    };
    let (h, i_, heads) = (int("hidden"), int("intermediate"), int("heads"));
    assert!(h % 16 == 0 && i_ % 16 == 0 && (h / heads) % 16 == 0,
        "hidden, intermediate and hidden/heads must be multiples of 16 (SIMD dot)");
    let pooling = m.get("pooling").and_then(|v| v.as_str()).expect("pooling missing");
    assert!(pooling == "cls" || pooling == "mean", "pooling must be 'cls' or 'mean'");
    let quant = m.get("quant").and_then(|v| v.as_str()).unwrap_or("");
    assert!(quant.is_empty() || quant == "int8", "quant must be 'int8' if set");

    let cfg = format!(
        "pub const V: usize = {};\n\
         pub const H: usize = {};\n\
         pub const L: usize = {};\n\
         pub const I: usize = {};\n\
         pub const HEADS: usize = {};\n\
         pub const MAXN: usize = {};\n\
         pub const POS_OFFSET: usize = {};\n\
         pub const LN_EPS: f32 = {:e};\n\
         pub const POOL_CLS: bool = {};\n\
         pub const QUANT_INT8: bool = {};\n",
        int("vocab_size"), h, int("layers"), i_, heads,
        int("max_tokens"), int("pos_offset"), flt("ln_eps"), pooling == "cls",
        quant == "int8",
    );
    fs::write(Path::new(&env::var("OUT_DIR").unwrap()).join("cfg.rs"), cfg).unwrap();
}
