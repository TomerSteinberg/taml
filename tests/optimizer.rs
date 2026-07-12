use ndarray::{ArrayD, IxDyn};
use taml::graph::Graph;
use taml::context::ExecutionContext;
use taml::optimizer::{SGD, Optimizer};

fn arr1(data: &[f64]) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap()
}

#[test]
fn sgd_updates_variables() {
    let mut g = Graph::new();
    let w = g.variable(&[2]);
    let y = g.add(w, w);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(w, arr1(&[10.0, 20.0]));
    ctx.forward(&g, y);
    ctx.backward(&g, y);

    let mut optimizer = SGD::new(0.1);
    optimizer.step(&g, &mut ctx);

    let actual: Vec<f64> = ctx.value(w).unwrap().iter().copied().collect();
    assert!((actual[0] - 9.8).abs() < 1e-10);
    assert!((actual[1] - 19.8).abs() < 1e-10);
}

#[test]
fn sgd_multiple_steps() {
    let mut g = Graph::new();
    let w = g.variable(&[1]);
    let y = g.add(w, w);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(w, arr1(&[10.0]));
    let mut optimizer = SGD::new(0.5);

    ctx.forward(&g, y);
    ctx.backward(&g, y);
    optimizer.step(&g, &mut ctx);
    assert!((ctx.value(w).unwrap().as_slice().unwrap()[0] - 9.0).abs() < 1e-10);

    ctx.zero_grad();
    ctx.forward(&g, y);
    ctx.backward(&g, y);
    optimizer.step(&g, &mut ctx);
    assert!((ctx.value(w).unwrap().as_slice().unwrap()[0] - 8.0).abs() < 1e-10);
}

#[test]
fn sgd_no_grad_does_not_change_value() {
    let mut g = Graph::new();
    let w = g.variable(&[1]);
    let y = g.add(w, w);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(w, arr1(&[5.0]));
    ctx.forward(&g, y);

    let mut optimizer = SGD::new(0.1);
    optimizer.step(&g, &mut ctx);
    assert!((ctx.value(w).unwrap().as_slice().unwrap()[0] - 5.0).abs() < 1e-10);
}

#[test]
fn sgd_skips_non_var_nodes() {
    let mut g = Graph::new();
    let x = g.input();
    let w = g.variable(&[1]);
    let y = g.mul(x, w);

    let mut ctx = ExecutionContext::new(&g);
    ctx.set_var(w, arr1(&[3.0]));
    ctx.set_input(x, arr1(&[2.0]));
    ctx.forward(&g, y);
    ctx.backward(&g, y);

    let mut optimizer = SGD::new(0.1);
    optimizer.step(&g, &mut ctx);
    assert!((ctx.value(w).unwrap().as_slice().unwrap()[0] - 2.8).abs() < 1e-10);
}
