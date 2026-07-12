# TAML(L) - Tomer's Amazing Machine Learning Library

[![Build Status](https://github.com/TomerSteinberg/taml/actions/workflows/taml.yml/badge.svg)](https://github.com/TomerSteinberg/taml/actions/workflows/taml.yml)
[![Rust 2024](https://img.shields.io/badge/Rust-2024-blue.svg)](https://blog.rust-lang.org/)

**TAML** is a lightweight, graph-based Automatic Differentiation and Machine Learning library written in Rust. Developed primarily as a learning project to deepen the understanding of Automatic Differentiation, backward propagation, and computational graphs, TAML is designed to be intuitive, clean, and extensible.

It uses `ndarray` as its sole dependency for N-dimensional tensor operations.

## Features

- **Define-then-Run Architecture**: Construct a computational graph (`Graph`), then compile it into a `Model` for execution.
- **Automatic Differentiation**: Full support for reverse-mode automatic differentiation. Gradients are computed recursively by traversing the topological order of the graph.
- **Fluent Graph API**: Easily chain mathematical operations together (`.chain(x).matmul(w).add(b).relu().end()`).
- **Extensive Operators**: Support for `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Exp`, `ReLU`, `Pow`, `Sum`, and `Mean`.
- **Modular Components**: Clean separation between graph definition (`graph.rs`), runtime context (`context.rs`), mathematical operations (`ops.rs`), and models (`model.rs`).
- **Optimizers**: Integrated optimizers for updating trainable variables.

## Installation

To use TAML in your Rust project, add it to your `Cargo.toml`:

```toml
[dependencies]
taml = "0.1.0"
```

Or just run:
```bash
cargo add taml
```

> **Note**: TAML requires the Rust 2024 edition.

## Quick Start

Here is a simple example of how to construct a graph, compile a model, and run a forward and backward pass.

```rust
use ndarray::{ArrayD, IxDyn};
use taml::graph::Graph;
use taml::model::Model;
// (Assuming SGD is implemented in taml::optimizer)
// use taml::optimizer::SGD; 

fn main() {
    // 1. Initialize the Graph
    let mut g = Graph::new();
    
    // 2. Define inputs and variables
    let x = g.input(); // Placeholder for input data
    let w = g.variable(&[2, 3]); // Trainable weights
    let b = g.variable(&[3]); // Trainable bias
    
    // 3. Define the computation (Forward Pass)
    let y = g.chain(x)
        .matmul(w)
        .add(b)
        .relu()
        .end();

    // 4. Compile the Model with an Optimizer
    // let mut model = Model::compile(g, SGD::new(0.01));
    
    // ... Provide data and execute ...
    // let input_data = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0);
    // model.set_input(x, input_data);
    
    // Forward pass to compute 'y'
    // model.forward(y);
    // println!("Output: {:?}", model.value(y));
    
    // Backward pass to compute gradients w.r.t 'y'
    // model.backward(y);
    
    // Update weights
    // model.optimizer_step();
}
```

## Architecture

TAML consists of several decoupled core modules:
- **`Graph`** (`src/graph.rs`): A blueprint of nodes (`Input`, `Const`, `Var`, `Op`). Doesn't hold data.
- **`ExecutionContext`** (`src/context.rs`): Holds the runtime values and gradients. Handles topological sorting, the `forward` pass, and the `backward` pass.
- **`Op`** (`src/ops.rs`): Defines the mathematical operations and their analytical derivatives (e.g., `MatMul`, `ReLU`, `Add`).
- **`Model`** (`src/model.rs`): A high-level wrapper combining a `Graph`, an `ExecutionContext`, and an `Optimizer` for streamlined training and inference.
- **`Optimizer`** (`src/optimizer.rs`): Defines how gradients should be applied to variables.

## Building & Testing

TAML uses the standard `cargo` toolchain.

```bash
cargo build         # Build the library
cargo test          # Run all test suites
cargo test <name>   # Run a specific test
```

Continuous integration runs `cargo build --verbose` and `cargo test --verbose` on every push to the `dev` branch.

## Motivation
This project is built from the ground up to provide a deeper understanding of how modern Machine Learning frameworks (like PyTorch or TensorFlow) operate under the hood, particularly focusing on computational graphs and backpropagation.
