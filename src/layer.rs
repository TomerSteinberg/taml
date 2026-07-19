use crate::graph::{Graph, NodeId};
use crate::initializer::{glorot_uniform, zeros};

/// A linear (fully-connected) layer shorthand: `y = x @ W + b`
///
/// Creates weight `(in_features, out_features)` and bias `(out_features,)` nodes,
/// appends `matmul` + `add` ops, and returns `(output, weight, bias)`.
///
/// # Example
///
/// ```rust
/// use taml::graph::Graph;
/// use taml::layer::linear;
///
/// let mut g = Graph::new();
/// let x = g.input();
///
/// let (h, _w, _b) = linear(&mut g, x, 784, 256);
/// let y = g.relu(h);
/// ```
pub fn linear(
    graph: &mut Graph,
    input: NodeId,
    in_features: usize,
    out_features: usize,
) -> (NodeId, NodeId, NodeId) {
    let w = graph.variable_with(&[in_features, out_features], glorot_uniform());
    let b = graph.variable_with(&[out_features], zeros());
    let output = graph.chain(input).matmul(w).add(b).end();
    (output, w, b)
}
