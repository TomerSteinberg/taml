# TAML — Agent Guide

**taml** (Tomer's Amazing Machine Learning Library) is a single-crate Rust library using the 2024 edition.
This is a learning project. Any addition to this project by an agent is allowed only for review and time saving purposes. 
The goal of the project is to create a production grade library while also not getting stuck on every decision trying to
find the perfect solution. Your job is to help guide me through what's not important and to reinforce my learning of Rust
and Automatic Differentiation concepts. I want to come out of this project with a much deeper understanding of how ML
libraries work under the hood.

## Build & test

```bash
cargo build
cargo test            # all tests
cargo test <name>      # single test by name
```

No Makefile, Justfile, or formatter config — use `cargo` directly. No rustfmt/clippy overrides; defaults apply.

## CI

GitHub Actions runs `cargo build --verbose && cargo test --verbose` on every push/PR to `dev` (not `main`).

## Structure

| Path | Role |
|---|---|
| `src/lib.rs` | Crate root; declares `pub mod ops` and `pub mod graph` |
| `src/ops.rs` | `Op` enum (Add, Mul, MatMul) with `compute` / `backward` using `ndarray::ArrayD<f64>` |
| `src/graph.rs` | **Empty** — stub to be implemented |
| `tests/`, `examples/` | **Empty** (`.gitkeep` only) — create test/example files from scratch |

## Dependencies

`ndarray 0.17.2` is the sole dependency. N-dimensional arrays use `ArrayD<f64>` and `IxDyn`.
