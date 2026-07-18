use crate::graph::{Graph, NodeId, NodeKind};
use ndarray::ArrayD;

/// Holds runtime values and gradients for every node in a Graph.
///
/// Created via `ExecutionContext::new(&graph)`, which evaluates all
/// Const and Var initializers stored in the graph.
pub struct ExecutionContext {
    pub(crate) values: Vec<Option<ArrayD<f64>>>,
    pub(crate) grads: Vec<Option<ArrayD<f64>>>,
}

impl ExecutionContext {
    /// Create a new context for the given graph.
    ///
    /// Evaluates all Const and Var initializers. Variables without
    /// an initializer are left as `None` — you must call `set_var`
    /// before calling `forward`.
    pub fn new(graph: &Graph) -> Self {
        let graph_size = graph.nodes.len();
        let mut values: Vec<Option<ArrayD<f64>>> = (0..graph_size).map(|_| None).collect();
        let grads: Vec<Option<ArrayD<f64>>> = (0..graph_size).map(|_| None).collect();

        for (i, node) in graph.nodes.iter().enumerate() {
            match &node.kind {
                NodeKind::Const(init) => values[i] = Some(init()),
                NodeKind::Var(meta) => values[i] = graph.resolve_init(meta),
                _ => {}
            }
        }

        ExecutionContext { values, grads }
    }

    /// Set the value of an Input node (must be called before `forward`).
    pub fn set_input(&mut self, node: NodeId, value: ArrayD<f64>) {
        self.values[node.0] = Some(value);
    }

    /// Manually set the value of a Var node (overrides the initializer).
    pub fn set_var(&mut self, node: NodeId, value: ArrayD<f64>) {
        self.values[node.0] = Some(value);
    }

    /// Compute the value of `node` and all its ancestors by traversing
    /// the graph in topological order (inputs before outputs).
    ///
    /// All Input and Var nodes must have values before calling this.
    pub fn forward(&mut self, graph: &Graph, node: NodeId) {
        let order = topological_forward_order(graph, node);

        for &node_id in &order {
            let n = &graph.nodes[node_id.0];
            if let NodeKind::Op(op) = &n.kind {
                // Clone values to avoid aliasing borrows
                let input_vals: Vec<ArrayD<f64>> = n
                    .inputs
                    .iter()
                    .map(|&i| {
                        self.values[i.0]
                            .clone()
                            .expect("forward: input has no value")
                    })
                    .collect();
                let input_refs: Vec<&ArrayD<f64>> = input_vals.iter().collect();
                self.values[node_id.0] = Some(op.compute(&input_refs));
            }
        }
    }

    /// Run backpropagation from `node` (typically the loss).
    ///
    /// Seeds `grad[node] = 1.0`, then walks the graph in reverse
    /// topological order, computing and accumulating gradients.
    pub fn backward(&mut self, graph: &Graph, node: NodeId) {
        let shape = self.values[node.0]
            .as_ref()
            .expect("backward: call forward first")
            .raw_dim();
        self.grads[node.0] = Some(ArrayD::from_elem(shape, 1.0));

        let order = topological_forward_order(graph, node);

        for &node_id in order.iter().rev() {
            let n = &graph.nodes[node_id.0];
            if let NodeKind::Op(op) = &n.kind {
                let input_vals: Vec<ArrayD<f64>> = n
                    .inputs
                    .iter()
                    .map(|&i| {
                        self.values[i.0]
                            .clone()
                            .expect("backward: input has no value")
                    })
                    .collect();
                let input_refs: Vec<&ArrayD<f64>> = input_vals.iter().collect();

                let output_grad = self.grads[node_id.0]
                    .clone()
                    .expect("backward: op missing gradient");

                let input_grads = op.backward(&input_refs, &output_grad);

                for (i, grad) in input_grads.into_iter().enumerate() {
                    let input_id = n.inputs[i].0;
                    match &mut self.grads[input_id] {
                        Some(existing) => {
                            // Accumulate gradients from multiple consumers
                            let current = existing.clone();
                            *existing = current + &grad;
                        }
                        None => self.grads[input_id] = Some(grad),
                    }
                }
            }
        }
    }

    /// Reset all gradients to zero.
    pub fn zero_grad(&mut self) {
        for gradient in &mut self.grads {
            *gradient = None;
        }
    }

    /// Get the computed value for a node, if any.
    pub fn value(&self, node: NodeId) -> Option<&ArrayD<f64>> {
        self.values[node.0].as_ref()
    }

    /// Get the computed gradient for a node, if any.
    pub fn grad(&self, node: NodeId) -> Option<&ArrayD<f64>> {
        self.grads[node.0].as_ref()
    }
}

/// Post-order DFS from `root`. Children (inputs) appear before their parents,
/// so the result is a valid forward evaluation order.
pub(crate) fn topological_forward_order(graph: &Graph, root: NodeId) -> Vec<NodeId> {
    let graph_size = graph.nodes.len();
    let mut visited = vec![false; graph_size];
    let mut order = Vec::with_capacity(graph_size);
    dfs_visit(graph, root, &mut visited, &mut order);
    order
}

fn dfs_visit(graph: &Graph, node: NodeId, visited: &mut Vec<bool>, order: &mut Vec<NodeId>) {
    if visited[node.0] {
        return;
    }
    visited[node.0] = true;
    for &input in &graph.nodes[node.0].inputs {
        dfs_visit(graph, input, visited, order);
    }
    order.push(node);
}
