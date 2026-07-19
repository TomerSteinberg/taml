//! # taml(l) - Tomer's Amazing Machine Learning Library
//!
//! A lightweight, n-dimensional automatic differentiation library for learning and experimentation.
//! `taml` provides a dynamic computation graph, backpropagation, and basic optimizers (like SGD)
//! to build and train simple machine learning models from scratch.
//!
//! ## Quick Start
//!
//! ```rust
//! use taml::graph::Graph;
//! use taml::model::Model;
//! use taml::optimizer::SGD;
//! use ndarray::{array, ArrayD, IxDyn};
//!
//! let mut g = Graph::new();
//! let x = g.input();
//! let w = g.variable(&[1, 1]); // Weight
//! let b = g.variable(&[1]);    // Bias
//!
//! // Build the graph: y = x * w + b
//! let y = g.chain(x).matmul(w).add(b).end();
//!
//! // Compile the model
//! let mut model = Model::compile(g, SGD::new(0.01));
//! ```

#![warn(missing_docs)]

/// The chain builder for fluent graph construction.
pub mod chain;
/// Execution context for storing intermediate values and gradients.
pub mod context;
/// The computation graph and node definitions.
pub mod graph;
/// Initializers for variables and constants.
pub mod initializer;
/// Shorthands for quick graph definition.
pub mod layer;
/// The high-level model API.
pub mod model;
/// Mathematical operations.
pub mod ops;
/// Optimizers for updating variables.
pub mod optimizer;
