//! Cognitive style primitives — 34 tactics as `fn` on `Base17 + NarsTruth`.
//!
//! Each tactic is a submodule implementing one cognitive primitive.
//! No LLM prompting — pure substrate operations.
//!
//! ```text
//! crate::hpc::styles::rte::expand()     #1  Recursive Thought Expansion
//! crate::hpc::styles::htd::decompose()  #2  Hierarchical Thought Decomposition
//! crate::hpc::styles::smad::debate()    #3  Structured Multi-Agent Debate
//! crate::hpc::styles::rcr::reverse()    #4  Reverse Causality Reasoning
//! crate::hpc::styles::tcp::prune()      #5  Thought Chain Pruning
//! crate::hpc::styles::tr::randomize()   #6  Thought Randomization
//! crate::hpc::styles::asc::critique()   #7  Adversarial Self-Critique
//! crate::hpc::styles::cas::scale()      #8  Conditional Abstraction Scaling
//! crate::hpc::styles::irs::sweep()      #9  Iterative Roleplay Synthesis
//! crate::hpc::styles::mcp::assess()     #10 Meta-Cognition Prompting
//! crate::hpc::styles::cr::detect()      #11 Contradiction Resolution
//! crate::hpc::styles::tca::augment()    #12 Temporal Context Augmentation
//! // #13-#34: pending
//! ```

pub mod rte;
pub mod htd;
pub mod smad;
pub mod tcp;
pub mod irs;
pub mod mcp;
pub mod tca;
pub mod cdt;
pub mod mct;
pub mod lsi;
pub mod pso;
pub mod cdi;
pub mod cws;
pub mod are;
pub mod tcf;
pub mod ssr;
pub mod etd;
pub mod amp;
pub mod zcf;
pub mod hpm;
pub mod cur;
pub mod mpc;
pub mod ssam;
pub mod idr;
pub mod spp;
pub mod icr;
pub mod sdd;
pub mod dtmf;
pub mod hkf;
