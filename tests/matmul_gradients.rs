use ndarray::{ArrayD, IxDyn};
use taml::graph::Graph;
use taml::model::Model;
use taml::optimizer::SGD;

fn arr1(data: &[f64]) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap()
}

fn arr2(data: &[f64], rows: usize, cols: usize) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(&[rows, cols]), data.to_vec()).unwrap()
}

fn approx_eq(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-10, "{x} != {y}");
    }
}

// ---------------------------------------------------------------------------
// MatMul backward with non-square shapes — verify analytical gradients
// ---------------------------------------------------------------------------

#[test]
fn matmul_backward_non_square() {
    let mut g = Graph::new();

    let a = g.variable(&[2, 3]);
    let b = g.variable(&[3, 2]);
    let y = g.matmul(a, b);
    let loss = g.mean(y);

    let mut model = Model::compile(g, SGD::new(0.01));

    // A = [[1,2,3],[4,5,6]]  (2x3)
    // B = [[7,8],[9,10],[11,12]]  (3x2)
    // A@B = [[58,64],[139,154]]
    // loss = mean(A@B) = (58+64+139+154)/4 = 103.75
    model.set_var(a, arr2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3));
    model.set_var(b, arr2(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2));
    model.forward(loss);

    // Verify forward
    let loss_val = model.value(loss).unwrap().as_slice().unwrap()[0];
    assert!((loss_val - 103.75).abs() < 1e-10, "loss={}", loss_val);

    model.backward(loss);

    // d(loss)/dy = 1/4 (gradient of mean) for each element of y (2x2)
    // So grad_y = [[0.25, 0.25], [0.25, 0.25]]
    //
    // d(loss)/dA = grad_y @ B^T
    // B^T = [[7,9,11],[8,10,12]] (2x3)
    // dA = [[0.25,0.25],[0.25,0.25]] @ [[7,9,11],[8,10,12]]
    //    = [[0.25*7+0.25*8, 0.25*9+0.25*10, 0.25*11+0.25*12],
    //       [0.25*7+0.25*8, 0.25*9+0.25*10, 0.25*11+0.25*12]]
    //    = [[3.75, 4.75, 5.75],
    //       [3.75, 4.75, 5.75]]
    //
    // d(loss)/dB = A^T @ grad_y
    // A^T = [[1,4],[2,5],[3,6]] (3x2)
    // dB = [[1,4],[2,5],[3,6]] @ [[0.25,0.25],[0.25,0.25]]
    //    = [[1*0.25+4*0.25, 1*0.25+4*0.25],
    //       [2*0.25+5*0.25, 2*0.25+5*0.25],
    //       [3*0.25+6*0.25, 3*0.25+6*0.25]]
    //    = [[1.25, 1.25],
    //       [1.75, 1.75],
    //       [2.25, 2.25]]
    let grad_a = model.grad(a).unwrap();
    let grad_b = model.grad(b).unwrap();

    approx_eq(
        grad_a.as_slice().unwrap(),
        &[3.75, 4.75, 5.75, 3.75, 4.75, 5.75],
    );
    approx_eq(
        grad_b.as_slice().unwrap(),
        &[1.25, 1.25, 1.75, 1.75, 2.25, 2.25],
    );
}

// ---------------------------------------------------------------------------
// Compound chain rule through 3+ ops
// ---------------------------------------------------------------------------

#[test]
fn compound_chain_rule() {
    let mut g = Graph::new();

    // f(w,x) = (w*x + w)^2
    // df/dw = 2*(w*x + w)*(x + 1)
    // df/dx = 2*(w*x + w)*w = 2*w^2*(x + 1)
    let w = g.variable(&[1]);
    let x = g.variable(&[1]);
    let wx = g.mul(w, x);
    let sum = g.add(wx, w);
    let y = g.pow(sum, 2.0);

    let mut model = Model::compile(g, SGD::new(0.01));
    model.set_var(w, arr1(&[3.0]));
    model.set_var(x, arr1(&[2.0]));

    model.forward(y);
    model.backward(y);

    // f = (3*2 + 3)^2 = (6+3)^2 = 81
    assert!((model.value(y).unwrap().as_slice().unwrap()[0] - 81.0).abs() < 1e-10);

    // df/dw = 2*(6+3)*(2+1) = 2*9*3 = 54
    let gw = model.grad(w).unwrap().as_slice().unwrap()[0];
    assert!((gw - 54.0).abs() < 1e-10, "df/dw = {gw}, expected 54");

    // df/dx = 2*(6+3)*3 = 2*9*3 = 54
    let gx = model.grad(x).unwrap().as_slice().unwrap()[0];
    assert!((gx - 54.0).abs() < 1e-10, "df/dx = {gx}, expected 54");
}

// ---------------------------------------------------------------------------
// Unreachable node has zero gradient
// ---------------------------------------------------------------------------

#[test]
fn unreachable_node_has_no_gradient() {
    let mut g = Graph::new();

    // Build two disjoint subgraphs, only compute loss from one
    let a = g.variable(&[1]);
    let b = g.variable(&[1]);
    let _disconnected = g.mul(a, b); // not used in loss

    let c = g.variable(&[1]);
    let loss = g.add(c, c); // 2*c

    let mut model = Model::compile(g, SGD::new(0.01));
    model.set_var(a, arr1(&[1.0]));
    model.set_var(b, arr1(&[2.0]));
    model.set_var(c, arr1(&[3.0]));

    model.forward(loss);
    model.backward(loss);

    // c has a gradient
    assert!(model.grad(c).is_some());
    // a and b were not visited during backward (not ancestors of loss)
    // so their gradients should be None
    assert!(
        model.grad(a).is_none(),
        "disconnected node a should have no gradient"
    );
    assert!(
        model.grad(b).is_none(),
        "disconnected node b should have no gradient"
    );
}

// ---------------------------------------------------------------------------
// Multiple outputs from the same graph
// ---------------------------------------------------------------------------

#[test]
fn multiple_outputs_independent_backward() {
    let mut g = Graph::new();

    // Shared input, two separate outputs
    let x = g.input();
    let w1 = g.variable(&[1]);
    let w2 = g.variable(&[1]);
    let y1 = g.mul(x, w1);
    let y2 = g.mul(x, w2);

    let mut model = Model::compile(g, SGD::new(0.01));
    model.set_var(w1, arr1(&[2.0]));
    model.set_var(w2, arr1(&[3.0]));
    model.set_input(x, arr1(&[5.0]));

    // Backprop through y1 only
    model.forward(y1);
    model.backward(y1);

    // grad_w1 = x = 5
    assert!((model.grad(w1).unwrap().as_slice().unwrap()[0] - 5.0).abs() < 1e-10);
    // w2 should have no gradient (not in the y1 computation path... actually it IS because w2 is not ancestor of y1)
    // Wait: w2 is not an ancestor of y1, so its gradient should be None
    assert!(model.grad(w2).is_none());

    model.zero_grad();

    // Backprop through y2 only
    model.set_input(x, arr1(&[5.0]));
    model.forward(y2);
    model.backward(y2);

    // grad_w2 = x = 5
    assert!((model.grad(w2).unwrap().as_slice().unwrap()[0] - 5.0).abs() < 1e-10);
    // w1 should have no gradient
    assert!(model.grad(w1).is_none());
}

// ---------------------------------------------------------------------------
// Deep linear chain — 5+ ops, numerical stability
// ---------------------------------------------------------------------------

#[test]
fn deep_linear_chain_forward() {
    let mut g = Graph::new();

    // Chain: x -> exp -> mul(2) -> exp -> mul(3) -> output
    // f(x) = 3 * exp(2 * exp(x))
    let x = g.input();
    let c1 = g.scalar_constant(2.0);
    let c2 = g.scalar_constant(3.0);
    let e1 = g.exp(x);
    let m1 = g.mul(e1, c1);
    let e2 = g.exp(m1);
    let y = g.mul(e2, c2);

    let mut model = Model::compile(g, SGD::new(0.01));
    model.set_input(x, arr1(&[0.0]));

    model.forward(y);

    // f(0) = 3 * exp(2 * exp(0)) = 3 * exp(2 * 1) = 3 * exp(2) = 3 * 7.389...
    let expected = 3.0 * std::f64::consts::E * std::f64::consts::E;
    let actual = model.value(y).unwrap().as_slice().unwrap()[0];
    assert!(
        (actual - expected).abs() < 1e-10,
        "expected {expected}, got {actual}"
    );
}

// ---------------------------------------------------------------------------
// SGD converges on quadratic — numerical gradient check
// ---------------------------------------------------------------------------

#[test]
fn sgd_converges_on_quadratic() {
    let mut g = Graph::new();

    // minimize f(w) = w^2
    // df/dw = 2w, SGD: w <- w - lr * 2w = w(1 - 2*lr)
    // With w=5, lr=0.1: w <- 5 - 0.1*10 = 4
    let w = g.variable(&[1]);
    let y = g.pow(w, 2.0);

    let mut model = Model::compile(g, SGD::new(0.1));
    model.set_var(w, arr1(&[5.0]));

    for _ in 0..10 {
        model.forward(y);
        model.backward(y);
        model.optimizer_step();
        model.zero_grad();
    }

    // After 10 steps with lr=0.1: w = 5 * (0.8)^10 ≈ 5 * 0.107 = 0.536
    let final_w = model.value(w).unwrap().as_slice().unwrap()[0];
    assert!(final_w.abs() < 1.0, "w = {final_w}, should be near 0");
}
