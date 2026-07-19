use ndarray::{ArrayD, IxDyn};
use taml::graph::Graph;
use taml::layer::linear;
use taml::model::Model;
use taml::optimizer::SGD;

#[test]
fn linear_forward_computes_xw_plus_b() {
    let mut g = Graph::new();
    let x = g.input();

    let (y, w, b) = linear(&mut g, x, 2, 3);

    let mut model = Model::compile(g, SGD::new(0.01));

    model.set_input(
        x,
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
    );
    model.set_var(
        w,
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    model.set_var(
        b,
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.2, 0.3]).unwrap(),
    );

    model.forward(y);

    let result = model.value(y).unwrap();
    let actual: Vec<f64> = result.iter().copied().collect();
    // x @ w = [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
    // + bias = [9.1, 12.2, 15.3]
    assert_eq!(actual, vec![9.1, 12.2, 15.3]);
}

#[test]
fn linear_forward_batch() {
    let mut g = Graph::new();
    let x = g.input();

    let (y, w, b) = linear(&mut g, x, 2, 3);

    let mut model = Model::compile(g, SGD::new(0.01));

    // Batch of 2
    model.set_input(
        x,
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    );
    model.set_var(
        w,
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
    );
    model.set_var(
        b,
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 0.5, 0.5]).unwrap(),
    );

    model.forward(y);

    let result = model.value(y).unwrap();
    // row 0: [1, 2] @ [[1,0,0],[0,1,0]] = [1, 2, 0] + [0.5, 0.5, 0.5] = [1.5, 2.5, 0.5]
    // row 1: [3, 4] @ [[1,0,0],[0,1,0]] = [3, 4, 0] + [0.5, 0.5, 0.5] = [3.5, 4.5, 0.5]
    let actual: Vec<f64> = result.iter().copied().collect();
    assert_eq!(actual, vec![1.5, 2.5, 0.5, 3.5, 4.5, 0.5]);
}

#[test]
fn linear_backward_produces_gradients() {
    let mut g = Graph::new();
    let x = g.input();

    let (y, w, b) = linear(&mut g, x, 2, 3);

    let mut model = Model::compile(g, SGD::new(0.01));

    model.set_input(
        x,
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
    );
    model.set_var(
        w,
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    model.set_var(
        b,
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.2, 0.3]).unwrap(),
    );

    model.forward(y);
    model.backward(y);

    // Gradient of y w.r.t. itself is all 1s
    // d(y)/dw = x^T @ 1 = [[1],[2]] @ [1,1,1] = [[1,1,1],[2,2,2]]
    let grad_w = model.grad(w).unwrap();
    let actual_w: Vec<f64> = grad_w.iter().copied().collect();
    assert_eq!(actual_w, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

    // d(y)/db = 1 (summed across batch)
    let grad_b = model.grad(b).unwrap();
    let actual_b: Vec<f64> = grad_b.iter().copied().collect();
    assert_eq!(actual_b, vec![1.0, 1.0, 1.0]);

    // d(y)/dx = 1 @ w^T = [1,1,1] @ [[1,4],[2,5],[3,6]]^T = [1+2+3, 4+5+6] = [6, 15]
    let grad_x = model.grad(x).unwrap();
    let actual_x: Vec<f64> = grad_x.iter().copied().collect();
    assert_eq!(actual_x, vec![6.0, 15.0]);
}

#[test]
fn linear_returns_output_weight_bias_in_order() {
    let mut g = Graph::new();
    let x = g.input();

    let (y, w, b) = linear(&mut g, x, 4, 7);

    assert_ne!(y, w);
    assert_ne!(y, b);
    assert_ne!(w, b);

    // All node IDs are distinct and valid
    let mut model = Model::compile(g, SGD::new(0.01));
    model.set_input(
        x,
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0; 4]).unwrap(),
    );
    model.forward(y);
    let val = model.value(y).unwrap();
    assert_eq!(val.shape(), &[1, 7]);
}
