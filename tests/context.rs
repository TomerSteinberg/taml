use ndarray::{ArrayD, IxDyn};
use taml::graph::{Graph};
use taml::context::ExecutionContext;
use taml::initializer;

fn arr1(data: &[f64]) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap()
}

fn approx_eq(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-10, "{x} != {y}");
    }
}

// =========================================================================
// Context creation
// =========================================================================

#[test]
fn context_initializes_const_value() {
    let mut g = Graph::new();
    let c = g.scalar_constant(42.0);
    let ctx = ExecutionContext::new(&g);
    let val = ctx.value(c).unwrap();
    assert_eq!(val.ndim(), 0);
    approx_eq(val.as_slice().unwrap(), &[42.0]);
}

#[test]
fn context_initializes_var_with_default_init() {
    let mut g = Graph::new();
    g.set_default_init(initializer::ones());
    let v = g.variable(&[3]);
    let ctx = ExecutionContext::new(&g);
    approx_eq(ctx.value(v).unwrap().as_slice().unwrap(), &[1.0, 1.0, 1.0]);
}

#[test]
fn context_var_without_init_is_none() {
    let mut g = Graph::new();
    let v = g.variable(&[3]);
    let ctx = ExecutionContext::new(&g);
    assert!(ctx.value(v).is_none());
}

#[test]
fn context_input_is_none() {
    let mut g = Graph::new();
    let x = g.input();
    let ctx = ExecutionContext::new(&g);
    assert!(ctx.value(x).is_none());
}

// =========================================================================
// Forward pass
// =========================================================================

#[test]
fn forward_simple_mul() {
    let mut g = Graph::new();
    let x = g.input();
    let w = g.variable(&[1]);
    let y = g.mul(x, w);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(w, arr1(&[3.0]));
    ctx.set_input(x, arr1(&[2.0]));
    ctx.forward(&g, y);
    approx_eq(ctx.value(y).unwrap().as_slice().unwrap(), &[6.0]);
}

#[test]
fn forward_linear_chain() {
    let mut g = Graph::new();
    let x = g.input();
    let a = g.variable(&[1]);
    let b = g.variable(&[1]);
    let ax = g.mul(x, a);
    let y = g.add(ax, b);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(a, arr1(&[2.0]));
    ctx.set_var(b, arr1(&[1.0]));
    ctx.set_input(x, arr1(&[5.0]));
    ctx.forward(&g, y);
    approx_eq(ctx.value(y).unwrap().as_slice().unwrap(), &[11.0]);
}

#[test]
fn forward_diamond() {
    let mut g = Graph::new();
    let a = g.variable(&[1]);
    let b = g.variable(&[1]);
    let c = g.variable(&[1]);
    let ab = g.mul(a, b);
    let ac = g.mul(a, c);
    let sum = g.add(ab, ac);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(a, arr1(&[3.0]));
    ctx.set_var(b, arr1(&[2.0]));
    ctx.set_var(c, arr1(&[4.0]));
    ctx.forward(&g, sum);
    approx_eq(ctx.value(sum).unwrap().as_slice().unwrap(), &[18.0]);
}

// =========================================================================
// Backward pass
// =========================================================================

#[test]
fn backward_linear_chain() {
    let mut g = Graph::new();
    let x = g.input();
    let a = g.variable(&[1]);
    let b = g.variable(&[1]);
    let ax = g.mul(x, a);
    let y = g.add(ax, b);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(a, arr1(&[2.0]));
    ctx.set_var(b, arr1(&[1.0]));
    ctx.set_input(x, arr1(&[5.0]));
    ctx.forward(&g, y);
    ctx.backward(&g, y);

    approx_eq(ctx.grad(a).unwrap().as_slice().unwrap(), &[5.0]);
    approx_eq(ctx.grad(b).unwrap().as_slice().unwrap(), &[1.0]);
}

#[test]
fn backward_gradient_accumulation() {
    let mut g = Graph::new();
    let a = g.variable(&[1]);
    let b = g.variable(&[1]);
    let c = g.variable(&[1]);
    let ab = g.mul(a, b);
    let ac = g.mul(a, c);
    let sum = g.add(ab, ac);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(a, arr1(&[3.0]));
    ctx.set_var(b, arr1(&[2.0]));
    ctx.set_var(c, arr1(&[4.0]));
    ctx.forward(&g, sum);
    ctx.backward(&g, sum);

    approx_eq(ctx.grad(a).unwrap().as_slice().unwrap(), &[6.0]);
    approx_eq(ctx.grad(b).unwrap().as_slice().unwrap(), &[3.0]);
    approx_eq(ctx.grad(c).unwrap().as_slice().unwrap(), &[3.0]);
}

#[test]
fn backward_matmul_graph() {
    let mut g = Graph::new();
    let x = g.input();
    let w = g.variable(&[3, 2]);
    let y = g.matmul(x, w);
    let loss = g.mean(y);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(w, initializer::ones()(&[3, 2]));
    ctx.set_input(x, ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![2.0, 3.0, 4.0]).unwrap());
    ctx.forward(&g, loss);
    ctx.backward(&g, loss);

    assert_eq!(ctx.value(loss).unwrap().as_slice().unwrap()[0], 9.0);
    let grad_w = ctx.grad(w).unwrap();
    approx_eq(grad_w.as_slice().unwrap(), &[1.0, 1.0, 1.5, 1.5, 2.0, 2.0]);
}

// =========================================================================
// Zero grad
// =========================================================================

#[test]
fn zero_grad_resets_all_gradients() {
    let mut g = Graph::new();
    let a = g.variable(&[1]);
    let b = g.variable(&[1]);
    let y = g.add(a, b);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(a, arr1(&[1.0]));
    ctx.set_var(b, arr1(&[2.0]));
    ctx.forward(&g, y);
    ctx.backward(&g, y);

    assert!(ctx.grad(a).is_some());
    assert!(ctx.grad(b).is_some());

    ctx.zero_grad();
    assert!(ctx.grad(a).is_none());
    assert!(ctx.grad(b).is_none());
}

// =========================================================================
// set_input and set_var
// =========================================================================

#[test]
fn set_input_and_const_forward() {
    let mut g = Graph::new();
    let x = g.input();
    let c = g.scalar_constant(10.0);
    let y = g.add(x, c);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_input(x, arr1(&[5.0]));
    ctx.forward(&g, y);
    approx_eq(ctx.value(y).unwrap().as_slice().unwrap(), &[15.0]);
}

#[test]
fn set_var_overrides_initializer() {
    let mut g = Graph::new();
    g.set_default_init(initializer::ones());
    let v = g.variable(&[2]);
    let y = g.add(v, v);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(v, arr1(&[10.0, 20.0]));
    ctx.forward(&g, y);
    approx_eq(ctx.value(y).unwrap().as_slice().unwrap(), &[20.0, 40.0]);
}

// =========================================================================
// Error cases
// =========================================================================

#[test]
#[should_panic(expected = "forward: input has no value")]
fn forward_panics_on_missing_input() {
    let mut g = Graph::new();
    let x = g.input();
    let y = g.neg(x);

    let mut ctx = ExecutionContext::new(&g);
    ctx.forward(&g, y);
}

#[test]
#[should_panic(expected = "backward: call forward first")]
fn backward_panics_without_forward() {
    let mut g = Graph::new();
    let a = g.variable(&[1]);
    let b = g.variable(&[1]);
    let y = g.add(a, b);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(a, arr1(&[1.0]));
    ctx.set_var(b, arr1(&[2.0]));
    ctx.backward(&g, y);
}
