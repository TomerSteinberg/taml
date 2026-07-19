use ndarray::{ArrayD, IxDyn};
use taml::graph::Graph;
use taml::initializer;
// =========================================================================
// Graph construction
// =========================================================================

#[test]
fn new_graph_round_trip_input() {
    let mut g = Graph::new();
    let x = g.input();
    let y = g.input();
    assert_ne!(x, y);
}

#[test]
fn input_and_variable_and_constant_have_unique_ids() {
    let mut g = Graph::new();
    let i = g.input();
    let v = g.variable(&[2, 3]);
    let c = g.scalar_constant(42.0);
    assert!(i.0 != v.0);
    assert!(v.0 != c.0);
}

#[test]
fn op_node_connects_inputs() {
    let mut g = Graph::new();
    let a = g.input();
    let b = g.input();
    let sum = g.add(a, b);
    // Can't inspect NodeKind directly, but can verify via forward/backward
    // Just verify it returns a valid node id
    assert!(sum.0 > 1);
}

#[test]
fn all_op_shortcuts_compile() {
    let mut g = Graph::new();
    let a = g.input();
    let b = g.input();

    // All of these should succeed without panicking
    let _ = g.add(a, b);
    let _ = g.sub(a, b);
    let _ = g.mul(a, b);
    let _ = g.div(a, b);
    let left = g.variable(&[2, 1]);
    let right = g.variable(&[1, 2]);
    let _ = g.matmul(left, right);
    let _ = g.neg(a);
    let _ = g.relu(a);
    let _ = g.exp(a);
    let _ = g.pow(a, 2.0);
    let _ = g.sum(a);
    let _ = g.mean(a);
}

// =========================================================================
// Chain builder
// =========================================================================

#[test]
fn chain_builder_produces_valid_graph() {
    let mut g = Graph::new();
    let x = g.input();
    let w = g.variable(&[3, 2]);
    let b = g.variable(&[2]);

    let y = g.chain(x).matmul(w).add(b).relu().end();
    // end() returns the last node id
    assert!(y.0 > 0);
}

#[test]
fn chain_builder_unary_ops() {
    let mut g = Graph::new();
    let x = g.input();
    let y = g.chain(x).neg().relu().exp().sum().mean().end();
    assert!(y.0 > 0);
}

#[test]
fn chain_builder_returns_to_normal_api() {
    let mut g = Graph::new();
    let x = g.input();
    let w = g.variable(&[2, 2]);
    let chained = g.chain(x).matmul(w).end();
    // Can continue using normal API after end()
    let one = g.scalar_constant(1.0);
    let y = g.add(chained, one);
    assert!(y.0 > 0);
}

// =========================================================================
// Node naming
// =========================================================================

#[test]
fn node_name_defaults_are_unique() {
    let mut g = Graph::new();
    let x = g.input();
    let w = g.variable(&[2, 3]);
    let y = g.add(x, w);
    assert_eq!(g.node_name(x), "input_0");
    assert_eq!(g.node_name(w), "var_2x3_1");
    assert_eq!(g.node_name(y), "add_2");
}

#[test]
fn set_node_name_overrides_default() {
    let mut g = Graph::new();
    let x = g.input();
    g.set_node_name(x, "features");
    assert_eq!(g.node_name(x), "features");
}

#[test]
fn with_name_chains() {
    let mut g = Graph::new();
    let x = g.input();
    g.set_node_name(x, "x");
    let w = g.variable(&[1]);
    g.set_node_name(w, "weight");
    assert_eq!(g.node_name(x), "x");
    assert_eq!(g.node_name(w), "weight");
}

// =========================================================================
// Initializers
// =========================================================================

#[test]
fn default_init_applies_to_all_vars() {
    let mut g = Graph::new();
    g.set_default_init(initializer::ones());
    let a = g.variable(&[2, 3]);
    let b = g.variable(&[4]);
    let ctx = taml::context::ExecutionContext::new(&g);
    assert!(
        ctx.value(a)
            .unwrap()
            .iter()
            .all(|&x| (x - 1.0).abs() < 1e-10)
    );
    assert!(
        ctx.value(b)
            .unwrap()
            .iter()
            .all(|&x| (x - 1.0).abs() < 1e-10)
    );
}

#[test]
fn explicit_init_overrides_default() {
    let mut g = Graph::new();
    g.set_default_init(initializer::ones());
    let a = g.variable_with(&[2], initializer::zeros());
    let ctx = taml::context::ExecutionContext::new(&g);
    assert_eq!(ctx.value(a).unwrap().as_slice().unwrap(), &[0.0, 0.0]);
}

#[test]
fn var_without_init_is_none() {
    let mut g = Graph::new();
    let v = g.variable(&[3]);
    let ctx = taml::context::ExecutionContext::new(&g);
    assert!(ctx.value(v).is_none());
}

#[test]
fn init_zeros_creates_zeros() {
    let init = initializer::zeros();
    let arr = init(&[2, 3]);
    assert_eq!(arr.shape(), &[2, 3]);
    assert!(arr.iter().all(|&x| x == 0.0));
}

#[test]
fn init_ones_creates_ones() {
    let arr = initializer::ones()(&[4]);
    assert!(arr.iter().all(|&x| x == 1.0));
}

#[test]
fn init_scalar_constant() {
    let arr = initializer::scalar(3.14)();
    assert_eq!(arr.ndim(), 0);
    assert!((arr.as_slice().unwrap()[0] - 3.14).abs() < 1e-10);
}

#[test]
fn init_constant_array() {
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let arr = initializer::constant_array(data)();
    assert_eq!(arr.shape(), &[2, 2]);
    assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn scalar_constant_is_zero_dimensional() {
    let mut g = Graph::new();
    let s = g.scalar_constant(42.0);
    let ctx = taml::context::ExecutionContext::new(&g);
    let val = ctx.value(s).unwrap();
    assert_eq!(val.ndim(), 0);
    assert!((val.as_slice().unwrap()[0] - 42.0).abs() < 1e-10);
}
