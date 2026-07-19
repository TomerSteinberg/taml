use ndarray::{ArrayD, IxDyn};
use taml::graph::Graph;
use taml::layer::linear;
use taml::model::Model;
use taml::optimizer::SGD;

fn main() {
    // XOR truth table
    let data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    let mut g = Graph::new();
    let x = g.input();
    let y_true = g.input();

    // Hidden layer 1: 2 -> 8
    let (h1, _w1, _b1) = linear(&mut g, x, 2, 8);
    let h1 = g.relu(h1);

    // Hidden layer 2: 8 -> 8
    let (h2, _w2, _b2) = linear(&mut g, h1, 8, 8);
    let h2 = g.relu(h2);

    // Output layer: 8 -> 1
    let (y_pred, _w3, _b3) = linear(&mut g, h2, 8, 1);
    g.set_node_name(y_pred, "prediction");

    // MSE loss
    let diff = g.sub(y_pred, y_true);
    let sq = g.pow(diff, 2.0);
    let loss = g.mean(sq);
    g.set_node_name(loss, "loss");

    let mut model = Model::compile(g, SGD::new(0.01));

    const EPOCHS: u32 = 10_000;

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0;

        for (xv, yv) in &data {
            model.set_input(
                x,
                ArrayD::from_shape_vec(IxDyn(&[1, 2]), xv.to_vec()).unwrap(),
            );
            model.set_input(
                y_true,
                ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![*yv]).unwrap(),
            );

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
            xv[0] as u8,
            xv[1] as u8,
            pred,
            yv,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    println!("\nResult XOR model view:");
    println!("{}", model.to_graphviz_dot());
}
