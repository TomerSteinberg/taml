use crate::context::ExecutionContext;
use crate::graph::{Graph, NodeId};
use crate::optimizer::Optimizer;
use ndarray::ArrayD;

/// A compiled model bundles a Graph blueprint with its ExecutionContext
/// and an Optimizer.
///
/// This is the high-level entry-point for training and inference.
/// For lower-level control, use Graph + ExecutionContext directly.
pub struct Model {
    pub(crate) graph: Graph,
    pub(crate) context: ExecutionContext,
    pub(crate) optimizer: Box<dyn Optimizer>,
}

impl Model {
    /// Compiles a graph into a runnable model.
    ///
    /// This creates the ExecutionContext (evaluating all initializers)
    /// and stores the optimizer for later `optimizer_step()` calls.
    pub fn compile(graph: Graph, optimizer: impl Optimizer + 'static) -> Self {
        let ctx = ExecutionContext::new(&graph);
        Model {
            graph,
            context: ctx,
            optimizer: Box::new(optimizer),
        }
    }

    /// Set the value of an input node.
    pub fn set_input(&mut self, node: NodeId, value: ArrayD<f64>) {
        self.context.set_input(node, value);
    }

    /// Set the value of a variable node.
    pub fn set_var(&mut self, node: NodeId, value: ArrayD<f64>) {
        self.context.set_var(node, value);
    }

    /// Run the forward pass up to the given node.
    pub fn forward(&mut self, node: NodeId) {
        self.context.forward(&self.graph, node);
    }

    /// Run the backward pass starting from the given node.
    pub fn backward(&mut self, node: NodeId) {
        self.context.backward(&self.graph, node);
    }

    /// Zero out all accumulated gradients.
    pub fn zero_grad(&mut self) {
        self.context.zero_grad();
    }

    /// Run one step of the optimizer to update variables.
    pub fn optimizer_step(&mut self) {
        self.optimizer.step(&self.graph, &mut self.context);
    }

    /// Get the computed value for a node, if any.
    pub fn value(&self, node: NodeId) -> Option<&ArrayD<f64>> {
        self.context.value(node)
    }

    /// Get the computed gradient for a node, if any.
    pub fn grad(&self, node: NodeId) -> Option<&ArrayD<f64>> {
        self.context.grad(node)
    }

    /// Set an input, run forward on `output`, and return the result.
    pub fn predict(&mut self, input: NodeId, output: NodeId, data: ArrayD<f64>) -> ArrayD<f64> {
        self.set_input(input, data);
        self.forward(output);
        self.value(output).unwrap().clone()
    }

    /// Access the underlying graph (e.g. for inspecting nodes).
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Access the underlying context (e.g. for manual gradient manipulation).
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }

    /// Access the underlying context mutably.
    pub fn context_mut(&mut self) -> &mut ExecutionContext {
        &mut self.context
    }
}
