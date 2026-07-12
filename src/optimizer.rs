use crate::context::ExecutionContext;
use crate::graph::{Graph, NodeId, NodeKind};

/// An optimizer updates trainable variables using their accumulated gradients.
pub trait Optimizer {
    /// Apply one step of the optimization algorithm.
    ///
    /// Iterates all Var nodes in the graph and updates their values
    /// based on `ctx.grad(node)`.
    fn step(&mut self, graph: &Graph, ctx: &mut ExecutionContext);
}

/// Stochastic Gradient Descent optimizer.
pub struct SGD {
    /// The learning rate step size.
    pub learning_rate: f64,
}

impl SGD {
    /// Create a new SGD optimizer with the given learning rate.
    pub fn new(learning_rate: f64) -> Self {
        SGD { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, graph: &Graph, context: &mut ExecutionContext) {
        for (id, node) in graph.nodes.iter().enumerate() {
            if let NodeKind::Var(_) = &node.kind {
                let node_id = NodeId(id);
                let grad = context.grad(node_id).cloned();
                if let Some(grad) = grad {
                    let mut value = context.values[id]
                        .take()
                        .expect("optimizer: variable has no value");
                    value = value - grad * self.learning_rate;
                    context.values[id] = Some(value);
                }
            }
        }
    }
}
