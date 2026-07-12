use ndarray::{ArrayD, IxDyn};
use taml::graph::Graph;
use taml::initializer;
use taml::model::Model;
use taml::optimizer::SGD;

// ---------------------------------------------------------------------------
// Chain builder
// ---------------------------------------------------------------------------

#[test]
fn chain_builder_reads_left_to_right() {
    let mut g = Graph::new();

    let x = g.input();
    let w = g.variable(&[3, 2]);
    let b = g.variable(&[2]);

    // Without chain (nested style):
    //   g.add(g.matmul(x, w), b)

    // With chain (linear style):
    let y = g.chain(x).matmul(w).add(b).end();

    let mut model = Model::compile(g, SGD::new(0.01));
    model.set_var(
        w,
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    model.set_var(
        b,
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1, 0.2]).unwrap(),
    );

    // Input must be 2D for matmul: [1, 3] x [3, 2] = [1, 2]
    model.set_input(
        x,
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![2.0, 3.0, 4.0]).unwrap(),
    );
    model.forward(y);

    // Manual: [2,3,4] . [[1,2],[3,4],[5,6]] = [31, 40]; + [0.1, 0.2] = [31.1, 40.2]
    let result: Vec<f64> = model.value(y).unwrap().iter().copied().collect();
    assert_eq!(result, vec![31.1, 40.2]);
}

// ---------------------------------------------------------------------------
// Multi-layer MLP with function composition
// ---------------------------------------------------------------------------

#[test]
fn deterministic_training_converges() {
    let mut g = Graph::new();

    let x = g.input();
    let t = g.input();
    let w = g.variable(&[1]);
    let b = g.variable(&[1]);
    let xw = g.mul(x, w);
    let y = g.add(xw, b);
    let err = g.sub(y, t);
    let sq_err = g.pow(err, 2.0);
    let loss = g.mean(sq_err);

    let mut model = Model::compile(g, SGD::new(0.02));

    // Deterministic init: w=0.5, b=0.0 (true: w=3.0, b=1.5)
    model.set_var(w, ArrayD::from_elem(IxDyn(&[1]), 0.5));
    model.set_var(b, ArrayD::from_elem(IxDyn(&[1]), 0.0));

    let data: [(f64, f64); 5] = [(0.0, 1.5), (1.0, 4.5), (2.0, 7.5), (3.0, 10.5), (4.0, 13.5)];

    let mut prev_loss = f64::INFINITY;
    for _epoch in 0..200 {
        for &(xv, tv) in &data {
            model.set_input(x, ArrayD::from_elem(IxDyn(&[1]), xv));
            model.set_input(t, ArrayD::from_elem(IxDyn(&[1]), tv));
            model.forward(loss);
            model.backward(loss);
            model.optimizer_step();
            model.zero_grad();
        }

        let mut total = 0.0;
        for &(xv, tv) in &data {
            model.set_input(x, ArrayD::from_elem(IxDyn(&[1]), xv));
            model.set_input(t, ArrayD::from_elem(IxDyn(&[1]), tv));
            model.forward(loss);
            total += model.value(loss).unwrap().as_slice().unwrap()[0];
        }
        total /= data.len() as f64;

        assert!(
            total <= prev_loss + 1e-12,
            "loss increased: {:.10} -> {:.10}",
            prev_loss,
            total
        );
        prev_loss = total;
    }

    assert!(prev_loss < 0.1, "final loss too high: {:.6}", prev_loss);
    let w_val = model.value(w).unwrap().as_slice().unwrap()[0];
    let b_val = model.value(b).unwrap().as_slice().unwrap()[0];
    assert!((w_val - 3.0).abs() < 0.2, "w={:.4} (expected 3.0)", w_val);
    assert!((b_val - 1.5).abs() < 0.2, "b={:.4} (expected 1.5)", b_val);
}

// ---------------------------------------------------------------------------
// Gradient accumulation (shared variable between two paths)
// ---------------------------------------------------------------------------

#[test]
fn gradient_accumulates_across_branches() {
    let mut g = Graph::new();

    // Build: loss = (a*b) + (a*c)  — a is shared
    let a = g.variable(&[1]);
    let b = g.variable(&[1]);
    let c = g.variable(&[1]);

    let ab = g.mul(a, b);
    let ac = g.mul(a, c);
    let sum = g.add(ab, ac);
    let loss = g.mean(sum);

    let mut model = Model::compile(g, SGD::new(0.01));
    model.set_var(a, ArrayD::from_elem(IxDyn(&[1]), 3.0));
    model.set_var(b, ArrayD::from_elem(IxDyn(&[1]), 2.0));
    model.set_var(c, ArrayD::from_elem(IxDyn(&[1]), 4.0));

    model.forward(loss);
    model.backward(loss);

    // loss = a*b + a*c = 3*2 + 3*4 = 6 + 12 = 18
    // d(loss)/da = b + c = 2 + 4 = 6
    // d(loss)/db = a = 3
    // d(loss)/dc = a = 3
    let grad_a = model.grad(a).unwrap().as_slice().unwrap()[0];
    let grad_b = model.grad(b).unwrap().as_slice().unwrap()[0];
    let grad_c = model.grad(c).unwrap().as_slice().unwrap()[0];

    assert!(
        (grad_a - 6.0).abs() < 1e-10,
        "d(loss)/da = {}, expected 6",
        grad_a
    );
    assert!(
        (grad_b - 3.0).abs() < 1e-10,
        "d(loss)/db = {}, expected 3",
        grad_b
    );
    assert!(
        (grad_c - 3.0).abs() < 1e-10,
        "d(loss)/dc = {}, expected 3",
        grad_c
    );
}

// ---------------------------------------------------------------------------
// Parameter initialization via set_default_init
// ---------------------------------------------------------------------------

#[test]
fn default_initializer_applies_to_vars() {
    let mut g = Graph::new();
    g.set_default_init(initializer::ones());

    let a = g.variable(&[2, 3]);
    let b = g.variable(&[4]);

    let ctx = taml::context::ExecutionContext::new(&g);
    let val_a = ctx.value(a).unwrap();
    let val_b = ctx.value(b).unwrap();

    // All ones
    assert!(val_a.iter().all(|&x| (x - 1.0).abs() < 1e-10));
    assert!(val_b.iter().all(|&x| (x - 1.0).abs() < 1e-10));
}

// ---------------------------------------------------------------------------
// Scalar constant
// ---------------------------------------------------------------------------

#[test]
fn scalar_constant_is_zero_dimensional() {
    let mut g = Graph::new();
    let s = g.scalar_constant(42.0);

    let ctx = taml::context::ExecutionContext::new(&g);
    let val = ctx.value(s).unwrap();
    assert_eq!(val.ndim(), 0);
    assert!((val.as_slice().unwrap()[0] - 42.0).abs() < 1e-10);
}
