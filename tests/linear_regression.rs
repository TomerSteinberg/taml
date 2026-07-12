use ndarray::{ArrayD, IxDyn};
use taml::graph::Graph;
use taml::model::Model;
use taml::optimizer::SGD;

/// Train a simple linear model y = Wx + b on synthetic data.
/// Verifies that the loss decreases and the parameters converge.
#[test]
fn linear_regression_converges() {
    let mut g = Graph::new();

    // True parameters
    let true_w = 3.0;
    let true_b = 1.5;

    // Build graph: y_pred = x * w + b
    let x = g.input();
    let t = g.input();
    let w = g.variable(&[1]);
    let b = g.variable(&[1]);

    // Bind intermediates to avoid nested &mut self calls
    let xw = g.mul(x, w);
    let y_pred = g.add(xw, b);
    let err = g.sub(y_pred, t);
    let sq_err = g.pow(err, 2.0);
    let loss = g.mean(sq_err);

    // Compile
    let mut model = Model::compile(g, SGD::new(0.01));

    // Set initial parameters
    model.set_var(w, ArrayD::from_elem(IxDyn(&[1]), 0.0));
    model.set_var(b, ArrayD::from_elem(IxDyn(&[1]), 0.0));

    // Train for 100 steps on a fixed dataset
    let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
    let ts: Vec<f64> = xs.iter().map(|x| true_w * x + true_b).collect();

    let mut prev_loss = f64::INFINITY;

    for _epoch in 0..100 {
        for (&xv, &tv) in xs.iter().zip(ts.iter()) {
            model.set_input(x, ArrayD::from_elem(IxDyn(&[1]), xv));
            model.set_input(t, ArrayD::from_elem(IxDyn(&[1]), tv));
            model.forward(loss);
            model.backward(loss);
            model.optimizer_step();
            model.zero_grad();
        }

        // Check loss after full batch
        let mut total_loss = 0.0;
        for (&xv, &tv) in xs.iter().zip(ts.iter()) {
            model.set_input(x, ArrayD::from_elem(IxDyn(&[1]), xv));
            model.set_input(t, ArrayD::from_elem(IxDyn(&[1]), tv));
            model.forward(loss);
            total_loss += model.value(loss).unwrap().as_slice().unwrap()[0];
        }
        total_loss /= xs.len() as f64;

        // loss should be non-increasing (this is SGD on convex problem)
        // Allow tiny numerical increases
        assert!(
            total_loss <= prev_loss + 1e-12,
            "loss increased: {:.10} -> {:.10}",
            prev_loss,
            total_loss
        );
        prev_loss = total_loss;
    }

    let loss_val = model.value(loss).unwrap().as_slice().unwrap()[0];
    assert!(loss_val < 0.1, "final loss too high: {:.6}", loss_val);

    // Parameters should be close to true values
    let w_val = model.value(w).unwrap().as_slice().unwrap()[0];
    let b_val = model.value(b).unwrap().as_slice().unwrap()[0];
    assert!(
        (w_val - true_w).abs() < 0.2,
        "w={:.4} (expected {})",
        w_val,
        true_w
    );
    assert!(
        (b_val - true_b).abs() < 0.2,
        "b={:.4} (expected {})",
        b_val,
        true_b
    );
}
