use ndarray::{ArrayD, IxDyn};
use taml::graph::Graph;
use taml::model::Model;
use taml::optimizer::SGD;

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
// Predict
// =========================================================================

#[test]
fn predict_runs_forward_and_returns_output() {
    let mut g = Graph::new();
    let x = g.input();
    let w = g.variable(&[1]);
    let b = g.variable(&[1]);
    let xw = g.mul(x, w);
    let y = g.add(xw, b);

    let mut model = Model::compile(g, SGD::new(0.01));
    model.set_var(w, arr1(&[2.0]));
    model.set_var(b, arr1(&[1.0]));

    let result = model.predict(x, y, arr1(&[5.0]));
    approx_eq(result.as_slice().unwrap(), &[11.0]);
}

#[test]
fn predict_is_idempotent() {
    let mut g = Graph::new();
    let x = g.input();
    let w = g.variable(&[1]);
    let y = g.mul(x, w);

    let mut model = Model::compile(g, SGD::new(0.01));
    model.set_var(w, arr1(&[3.0]));

    let r1 = model.predict(x, y, arr1(&[2.0]));
    let r2 = model.predict(x, y, arr1(&[4.0]));
    approx_eq(r1.as_slice().unwrap(), &[6.0]);
    approx_eq(r2.as_slice().unwrap(), &[12.0]);
    approx_eq(model.value(w).unwrap().as_slice().unwrap(), &[3.0]);
}

// =========================================================================
// Training loop
// =========================================================================

#[test]
fn forward_backward_step_cycle() {
    let mut g = Graph::new();
    let x = g.input();
    let t = g.input();
    let w = g.variable(&[1]);
    let b = g.variable(&[1]);
    let xw = g.mul(x, w);
    let pred = g.add(xw, b);
    let err = g.sub(pred, t);
    let sq = g.pow(err, 2.0);
    let loss = g.mean(sq);

    let mut model = Model::compile(g, SGD::new(0.02));
    model.set_var(w, arr1(&[0.5]));
    model.set_var(b, arr1(&[0.0]));

    model.set_input(x, arr1(&[1.0]));
    model.set_input(t, arr1(&[4.5]));
    model.forward(loss);
    model.backward(loss);
    model.optimizer_step();

    let loss_before = model.value(loss).unwrap().as_slice().unwrap()[0];

    model.zero_grad();
    model.set_input(x, arr1(&[1.0]));
    model.set_input(t, arr1(&[4.5]));
    model.forward(loss);
    let loss_after = model.value(loss).unwrap().as_slice().unwrap()[0];

    assert!(loss_after <= loss_before);
}

#[test]
fn zero_grad_clears_gradients() {
    let mut g = Graph::new();
    let a = g.variable(&[1]);
    let y = g.add(a, a);

    let mut model = Model::compile(g, SGD::new(0.01));
    model.set_var(a, arr1(&[5.0]));
    model.forward(y);
    model.backward(y);
    assert!(model.grad(a).is_some());

    model.zero_grad();
    assert!(model.grad(a).is_none());
}

// =========================================================================
// Accessors
// =========================================================================

#[test]
fn graph_and_context_accessors() {
    let g = Graph::new();
    let mut model = Model::compile(g, SGD::new(0.01));
    let _g = model.graph();
    let _c = model.context();
    let _cm = model.context_mut();
}
