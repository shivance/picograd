pub mod engine;
pub use crate::engine::Value;

pub mod nn;
pub use crate::nn::{MLP, Neuron, Layer};