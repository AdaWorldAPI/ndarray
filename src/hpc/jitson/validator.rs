//! Schema validation for JITSON templates.
//!
//! Validates parsed JSON against the JITSON template schema, checking:
//! - Required fields (version, kernel, scan)
//! - Known kernels, features, instructions, presets, backends
//! - Pipeline stage structure and backend references
//! - Type constraints (integers, booleans, strings)

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::parser::JsonValue;

/// Validation error with a JSON-pointer path.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl core::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "JITSON schema error at {}: {}", self.path, self.message)
    }
}

/// All AVX-512 feature flags supported by the patched Cranelift backend.
pub const KNOWN_FEATURES: &[&str] = &[
    "avx512f",
    "avx512vl",
    "avx512bw",
    "avx512dq",
    "avx512bitalg",
    "avx512vbmi",
    "avx512vpopcntdq",
    "avx512vnni",
    "avx512ifma",
];

/// All AVX-512 instruction mnemonics from the patched Cranelift.
pub const KNOWN_INSTRUCTIONS: &[&str] = &[
    // abs
    "vpabsb", "vpabsw", "vpabsd", "vpabsq",
    // and / ternlog
    "vpandd", "vpandq", "vpandnd", "vpandnq", "vpternlogd", "vpternlogq",
    // bitmanip
    "vpopcntb", "vpopcntw", "vpopcntd", "vpopcntq",
    // fma (132/213/231 x ps/pd x add/sub/nmadd)
    "vfmadd132ps", "vfmadd213ps", "vfmadd231ps",
    "vfmadd132pd", "vfmadd213pd", "vfmadd231pd",
    "vfmsub132ps", "vfmsub213ps", "vfmsub231ps",
    "vfmsub132pd", "vfmsub213pd", "vfmsub231pd",
    "vfnmadd132ps", "vfnmadd213ps", "vfnmadd231ps",
    "vfnmadd132pd", "vfnmadd213pd", "vfnmadd231pd",
    // mul / vnni
    "vpmulld", "vpmullq",
    "vpdpbusd", "vpdpbusds", "vpdpwssd", "vpdpwssds",
    // or
    "vpord", "vporq",
    // shift
    "vpsllw", "vpslld", "vpsllq",
    "vpsraw", "vpsrad", "vpsraq",
    "vpsrlw", "vpsrld", "vpsrlq",
    // xor
    "vpxord", "vpxorq",
    // add
    "vaddpd",
    // cvt
    "vcvtudq2ps",
    // lanes
    "vpermi2b",
];

/// Known kernel names.
const KNOWN_KERNELS: &[&str] = &["hamming_distance", "cosine_i8", "dot_f32"];

/// Known backend names for CPU-lane data sources/sinks.
pub const KNOWN_BACKENDS: &[&str] = &["lancedb", "dragonfly"];

/// Known Cranelift presets.
const KNOWN_PRESETS: &[&str] = &[
    "baseline", "nehalem", "haswell", "broadwell", "skylake",
    "knl", "knm", "skylake_avx512", "cascade_lake", "cooper_lake",
    "cannon_lake", "ice_lake_client", "ice_lake_server", "tiger_lake",
    "sapphire_rapids", "x86_64_v2", "x86_64_v3", "x86_64_v4",
];

/// Known opt levels.
const KNOWN_OPT_LEVELS: &[&str] = &["none", "speed", "speed_and_size"];

/// Validate a parsed JITSON template against the schema.
///
/// Returns a list of all validation errors found (empty = valid).
pub fn validate(root: &JsonValue) -> Vec<ValidationError> {
    let mut errs = Vec::new();

    let obj = match root.as_object() {
        Some(o) => o,
        None => {
            errs.push(ValidationError {
                path: String::from("/"),
                message: String::from("root must be a JSON object"),
            });
            return errs;
        }
    };

    // version (required, must be 1)
    match root.get("version") {
        Some(v) => match v.as_u64() {
            Some(1) => {}
            Some(n) => errs.push(ValidationError {
                path: String::from("/version"),
                message: alloc::format!("unsupported version {}, expected 1", n),
            }),
            None => errs.push(ValidationError {
                path: String::from("/version"),
                message: String::from("must be an integer"),
            }),
        },
        None => errs.push(ValidationError {
            path: String::from("/version"),
            message: String::from("required field missing"),
        }),
    }

    // kernel (required, one of known kernels)
    match root.get("kernel") {
        Some(v) => match v.as_str() {
            Some(s) if KNOWN_KERNELS.contains(&s) => {}
            Some(s) => errs.push(ValidationError {
                path: String::from("/kernel"),
                message: alloc::format!(
                    "unknown kernel \"{}\", expected one of: {}",
                    s,
                    KNOWN_KERNELS.join(", ")
                ),
            }),
            None => errs.push(ValidationError {
                path: String::from("/kernel"),
                message: String::from("must be a string"),
            }),
        },
        None => errs.push(ValidationError {
            path: String::from("/kernel"),
            message: String::from("required field missing"),
        }),
    }

    // scan (required object with threshold, record_size, top_k)
    match root.get("scan") {
        Some(scan) => {
            if scan.as_object().is_none() {
                errs.push(ValidationError {
                    path: String::from("/scan"),
                    message: String::from("must be an object"),
                });
            } else {
                validate_uint_field(scan, "threshold", "/scan/threshold", &mut errs);
                validate_uint_field(scan, "record_size", "/scan/record_size", &mut errs);
                validate_uint_field(scan, "top_k", "/scan/top_k", &mut errs);
            }
        }
        None => errs.push(ValidationError {
            path: String::from("/scan"),
            message: String::from("required field missing"),
        }),
    }

    // pipeline (optional array of stage objects)
    if let Some(pipeline) = root.get("pipeline") {
        match pipeline.as_array() {
            Some(stages) => {
                for (i, stage) in stages.iter().enumerate() {
                    let prefix = alloc::format!("/pipeline/{}", i);
                    if stage.as_object().is_none() {
                        errs.push(ValidationError {
                            path: prefix.clone(),
                            message: String::from("each pipeline stage must be an object"),
                        });
                        continue;
                    }
                    if stage.get("stage").and_then(|v| v.as_str()).is_none() {
                        errs.push(ValidationError {
                            path: alloc::format!("{}/stage", prefix),
                            message: String::from("required string field"),
                        });
                    }
                    if let Some(instr) = stage.get("avx512").and_then(|v| v.as_str()) {
                        if !KNOWN_INSTRUCTIONS.contains(&instr) {
                            errs.push(ValidationError {
                                path: alloc::format!("{}/avx512", prefix),
                                message: alloc::format!(
                                    "unknown instruction \"{}\"; not in patched Cranelift",
                                    instr
                                ),
                            });
                        }
                    }
                }
            }
            None => errs.push(ValidationError {
                path: String::from("/pipeline"),
                message: String::from("must be an array"),
            }),
        }
    }

    // features (optional object of bool flags)
    if let Some(features) = root.get("features") {
        match features.as_object() {
            Some(pairs) => {
                for (key, val) in pairs {
                    if !KNOWN_FEATURES.contains(&key.as_str()) {
                        errs.push(ValidationError {
                            path: alloc::format!("/features/{}", key),
                            message: alloc::format!(
                                "unknown feature \"{}\"; known: {}",
                                key,
                                KNOWN_FEATURES.join(", ")
                            ),
                        });
                    }
                    if val.as_bool().is_none() {
                        errs.push(ValidationError {
                            path: alloc::format!("/features/{}", key),
                            message: String::from("must be a boolean"),
                        });
                    }
                }
            }
            None => errs.push(ValidationError {
                path: String::from("/features"),
                message: String::from("must be an object"),
            }),
        }
    }

    // cranelift (optional)
    if let Some(cl) = root.get("cranelift") {
        if cl.as_object().is_none() {
            errs.push(ValidationError {
                path: String::from("/cranelift"),
                message: String::from("must be an object"),
            });
        } else {
            if let Some(preset) = cl.get("preset").and_then(|v| v.as_str()) {
                if !KNOWN_PRESETS.contains(&preset) {
                    errs.push(ValidationError {
                        path: String::from("/cranelift/preset"),
                        message: alloc::format!("unknown preset \"{}\"", preset),
                    });
                }
            }
            if let Some(opt) = cl.get("opt_level").and_then(|v| v.as_str()) {
                if !KNOWN_OPT_LEVELS.contains(&opt) {
                    errs.push(ValidationError {
                        path: String::from("/cranelift/opt_level"),
                        message: alloc::format!("unknown opt_level \"{}\"", opt),
                    });
                }
            }
        }
    }

    // backends (optional object of named backend configs)
    if let Some(backends) = root.get("backends") {
        match backends.as_object() {
            Some(pairs) => {
                for (key, val) in pairs {
                    if !KNOWN_BACKENDS.contains(&key.as_str()) {
                        errs.push(ValidationError {
                            path: alloc::format!("/backends/{}", key),
                            message: alloc::format!(
                                "unknown backend \"{}\"; known: {}",
                                key,
                                KNOWN_BACKENDS.join(", ")
                            ),
                        });
                    }
                    if val.as_object().is_none() {
                        errs.push(ValidationError {
                            path: alloc::format!("/backends/{}", key),
                            message: String::from("must be an object"),
                        });
                    } else if val.get("uri").and_then(|v| v.as_str()).is_none() {
                        errs.push(ValidationError {
                            path: alloc::format!("/backends/{}/uri", key),
                            message: String::from("required string field"),
                        });
                    }
                }
            }
            None => errs.push(ValidationError {
                path: String::from("/backends"),
                message: String::from("must be an object"),
            }),
        }
    }

    // Validate pipeline stage backend references
    if let Some(pipeline) = root.get("pipeline").and_then(|v| v.as_array()) {
        let declared_backends: Vec<&str> = root
            .get("backends")
            .and_then(|v| v.as_object())
            .map(|pairs| pairs.iter().map(|(k, _)| k.as_str()).collect())
            .unwrap_or_default();
        for (i, stage) in pipeline.iter().enumerate() {
            if let Some(backend) = stage.get("backend").and_then(|v| v.as_str()) {
                if !declared_backends.contains(&backend) {
                    errs.push(ValidationError {
                        path: alloc::format!("/pipeline/{}/backend", i),
                        message: alloc::format!(
                            "backend \"{}\" referenced but not declared in /backends",
                            backend
                        ),
                    });
                }
            }
        }
    }

    // Warn on unknown top-level keys
    let known_top: &[&str] = &[
        "version", "kernel", "scan", "pipeline", "features", "cranelift", "backends",
    ];
    for (key, _) in obj {
        if !known_top.contains(&key.as_str()) {
            errs.push(ValidationError {
                path: alloc::format!("/{}", key),
                message: alloc::format!("unknown field \"{}\"", key),
            });
        }
    }

    errs
}

fn validate_uint_field(parent: &JsonValue, key: &str, path: &str, errs: &mut Vec<ValidationError>) {
    match parent.get(key) {
        Some(v) => {
            if v.as_u64().is_none() {
                errs.push(ValidationError {
                    path: String::from(path),
                    message: String::from("must be a non-negative integer"),
                });
            }
        }
        None => errs.push(ValidationError {
            path: String::from(path),
            message: String::from("required field missing"),
        }),
    }
}

/// Return the required AVX-512 feature flags for a given instruction mnemonic.
pub fn required_features(instruction: &str) -> &'static [&'static str] {
    match instruction {
        "vpabsb" | "vpabsw" => &["avx512vl", "avx512bw"],
        "vpabsd" | "vpabsq" => &["avx512vl", "avx512f"],
        "vpandd" | "vpandq" | "vpandnd" | "vpandnq" | "vpternlogd" | "vpternlogq" => {
            &["avx512vl", "avx512f"]
        }
        "vpopcntb" | "vpopcntw" => &["avx512vl", "avx512bitalg"],
        "vpopcntd" | "vpopcntq" => &["avx512vl", "avx512vpopcntdq"],
        "vfmadd132ps" | "vfmadd213ps" | "vfmadd231ps" | "vfmadd132pd" | "vfmadd213pd"
        | "vfmadd231pd" | "vfmsub132ps" | "vfmsub213ps" | "vfmsub231ps" | "vfmsub132pd"
        | "vfmsub213pd" | "vfmsub231pd" | "vfnmadd132ps" | "vfnmadd213ps" | "vfnmadd231ps"
        | "vfnmadd132pd" | "vfnmadd213pd" | "vfnmadd231pd" => &["avx512vl", "avx512f"],
        "vpmulld" => &["avx512vl", "avx512f"],
        "vpmullq" => &["avx512vl", "avx512dq"],
        "vpdpbusd" | "vpdpbusds" | "vpdpwssd" | "vpdpwssds" => &["avx512vl", "avx512vnni"],
        "vpord" | "vporq" => &["avx512vl", "avx512f"],
        "vpsllw" | "vpsraw" | "vpsrlw" => &["avx512vl", "avx512bw"],
        "vpslld" | "vpsllq" | "vpsrad" | "vpsraq" | "vpsrld" | "vpsrlq" => {
            &["avx512vl", "avx512f"]
        }
        "vpxord" | "vpxorq" => &["avx512vl", "avx512f"],
        "vaddpd" => &["avx512vl"],
        "vcvtudq2ps" => &["avx512vl", "avx512f"],
        "vpermi2b" => &["avx512vl", "avx512vbmi"],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::jitson::parser::parse_json;

    const VALID_TEMPLATE: &str = r#"{
        "version": 1,
        "kernel": "hamming_distance",
        "scan": {
            "threshold": 2048,
            "record_size": 256,
            "top_k": 10
        },
        "pipeline": [
            { "stage": "xor",    "avx512": "vpxord" },
            { "stage": "popcnt", "avx512": "vpopcntd", "fallback": "avx2_lookup" },
            { "stage": "reduce", "avx512": "vpord" }
        ],
        "features": {
            "avx512f": true,
            "avx512vl": true,
            "avx512vpopcntdq": true,
            "avx512bw": false
        },
        "cranelift": {
            "preset": "sapphire_rapids",
            "opt_level": "speed"
        }
    }"#;

    #[test]
    fn test_validate_valid() {
        let root = parse_json(VALID_TEMPLATE).unwrap();
        let errs = validate(&root);
        assert!(errs.is_empty(), "unexpected errors: {:?}", errs);
    }

    #[test]
    fn test_validate_missing_fields() {
        let input = r#"{"version": 1}"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.path == "/kernel"));
        assert!(errs.iter().any(|e| e.path == "/scan"));
    }

    #[test]
    fn test_validate_bad_version() {
        let input = r#"{"version": 99, "kernel": "dot_f32", "scan": {"threshold": 1, "record_size": 64, "top_k": 5}}"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.path == "/version"));
    }

    #[test]
    fn test_validate_unknown_feature() {
        let input = r#"{"version": 1, "kernel": "dot_f32", "scan": {"threshold": 1, "record_size": 64, "top_k": 5}, "features": {"avx512_bogus": true}}"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.path.contains("avx512_bogus")));
    }

    #[test]
    fn test_validate_unknown_instruction() {
        let input = r#"{"version": 1, "kernel": "dot_f32", "scan": {"threshold": 1, "record_size": 64, "top_k": 5}, "pipeline": [{"stage": "nope", "avx512": "vfakeop"}]}"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.message.contains("vfakeop")));
    }

    #[test]
    fn test_required_features_mapping() {
        assert_eq!(required_features("vpxord"), &["avx512vl", "avx512f"]);
        assert_eq!(
            required_features("vpopcntd"),
            &["avx512vl", "avx512vpopcntdq"]
        );
        assert_eq!(required_features("vpdpbusd"), &["avx512vl", "avx512vnni"]);
        assert_eq!(required_features("vpermi2b"), &["avx512vl", "avx512vbmi"]);
        assert_eq!(required_features("not_real"), &[] as &[&str]);
    }

    #[test]
    fn test_validate_undeclared_backend() {
        let input = r#"{
            "version": 1,
            "kernel": "dot_f32",
            "scan": { "threshold": 1, "record_size": 64, "top_k": 5 },
            "pipeline": [{ "stage": "fetch", "backend": "postgres" }]
        }"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.message.contains("postgres")));
    }

    #[test]
    fn test_validate_unknown_backend_type() {
        let input = r#"{
            "version": 1,
            "kernel": "dot_f32",
            "scan": { "threshold": 1, "record_size": 64, "top_k": 5 },
            "backends": { "mongodb": { "uri": "mongodb://localhost" } }
        }"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.message.contains("mongodb")));
    }
}
