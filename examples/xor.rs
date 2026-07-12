use ndarray::{ArrayD, IxDyn};
use taml::graph::{Graph};
use taml::initializer;
use taml::model::Model;
use taml::optimizer::SGD;

fn main() {
    // XOR truth table: each input is (x1, x2) -> target
    let data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    // Build the computation graph
    let mut graph = Graph::new();

    let x = graph.input();
    let y_true = graph.input();

    // Hidden layer 1: 2 inputs -> 8 hidden units
    let w1 = graph.variable_with(&[2, 8], initializer::glorot_uniform());
    let b1 = graph.variable_with(&[8], initializer::zeros());
    let z1 = graph.matmul(x, w1);
    let a1 = graph.add(z1, b1);
    let h1 = graph.relu(a1);

    // Hidden layer 2: 8 -> 8
    let w2 = graph.variable_with(&[8, 8], initializer::glorot_uniform());
    let b2 = graph.variable_with(&[8], initializer::zeros());
    let z2 = graph.matmul(h1, w2);
    let a2 = graph.add(z2, b2);
    let h2 = graph.relu(a2);

    // Output layer: 8 hidden units -> 1 output
    let w3 = graph.variable_with(&[8, 1], initializer::glorot_uniform());
    let b3 = graph.variable_with(&[1], initializer::zeros());
    let z3 = graph.matmul(h2, w3);
    let y_pred = graph.add(z3, b3);

    // MSE loss: mean((y_pred - y_true)^2)
    let diff = graph.sub(y_pred, y_true);
    let sq = graph.pow(diff, 2.0);
    let loss = graph.mean(sq);

    let mut model = Model::compile(graph, SGD::new(0.01));

    const EPOCHS: u32 = 10_000;

    for epoch in 0.. EPOCHS {
        let mut total_loss = 0.0;

        for (xv, yv) in &data {
            let x_arr = ArrayD::from_shape_vec(IxDyn(&[1, 2]), xv.to_vec()).unwrap();
            let y_arr = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![*yv]).unwrap();

            model.set_input(x, x_arr);
            model.set_input(y_true, y_arr);

            model.forward(loss);
            total_loss += model.value(loss).unwrap().as_slice().unwrap()[0];

            model.zero_grad();
            model.backward(loss);
            model.optimizer_step();
        }

        if epoch % 1000 == 0 {
            let avg_loss = total_loss / data.len() as f64;
            println!("Epoch {epoch:>5}, loss: {avg_loss:.6}");
        }
    }

    // Test the trained model
    println!("\nTrained XOR results:");
    for (xv, yv) in &data {
        let x_arr = ArrayD::from_shape_vec(IxDyn(&[1, 2]), xv.to_vec()).unwrap();
        let result = model.predict(x, y_pred, x_arr);
        let pred = result.as_slice().unwrap()[0];
        let pass = (pred > 0.5) == (*yv > 0.5);
        println!(
            "  {} XOR {} = {:.4}  (expected {:.0}, {})",
            xv[0] as u8, xv[1] as u8, pred, yv,
            if pass { "PASS" } else { "FAIL" }
        );
    }
}
