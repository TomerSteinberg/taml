use crate::context::ExecutionContext;
use crate::graph::{Graph, NodeId, NodeKind};
use crate::optimizer::Optimizer;
use ndarray::ArrayD;
use std::fmt::Write;

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

    /// Render the model's computation graph to a Graphviz DOT string,
    /// including runtime values and gradients from the ExecutionContext.
    ///
    /// Pipe the output through `dot -Tpng > graph.png` or paste into
    /// <https://dreampuf.github.io/GraphvizOnline/>.
    /// This is for debugging use
    pub fn to_graphviz_dot(&self) -> String {
        let mut dot = String::from("digraph G {\n  rankdir=LR;\n  node [shape=record];\n");

        for (node_idx, node) in self.graph.nodes.iter().enumerate() {
            let id = NodeId(node_idx);
            let name = escape_dot_label(self.graph.node_name(id));

            let mut parts: Vec<String> = vec![name];

            match &node.kind {
                NodeKind::Input => parts.push("Input".into()),
                NodeKind::Const(_) => parts.push("Const".into()),
                NodeKind::Var(meta) => {
                    parts.push("Var".into());
                    if !meta.shape.is_empty() {
                        parts.push(
                            meta.shape
                                .iter()
                                .map(|dim| dim.to_string())
                                .collect::<Vec<_>>()
                                .join("×"),
                        );
                    }
                }
                NodeKind::Op(op) => parts.push(op.to_string()),
            }

            if let Some(val) = &self.context.values[node_idx] {
                parts.push(format!(
                    "val: {}",
                    escape_dot_label(&fmt_array_preview(val))
                ));
            }
            if let Some(grad) = &self.context.grads[node_idx] {
                parts.push(format!(
                    "grad: {}",
                    escape_dot_label(&fmt_array_preview(grad))
                ));
            }

            let _ = writeln!(
                dot,
                "  n{node_idx} [label=\"{label}\"];",
                label = parts.join(" | ")
            );

            for &input in &node.inputs {
                let _ = writeln!(dot, "  n{} -> n{node_idx};", input.0);
            }
        }

        dot.push_str("}\n");
        dot
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

/// Escape characters with special meaning in Graphviz `shape=record` labels.
/// In record shapes, `|`, `{`, `}`, `<`, `>` have structural meaning, and
/// `\` and `"` are escape/delimiter characters in the DOT language itself.
fn escape_dot_label(label_text: &str) -> String {
    let mut out = String::with_capacity(label_text.len());
    for ch in label_text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '|' => out.push_str("\\|"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '<' => out.push_str("\\<"),
            '>' => out.push_str("\\>"),
            other => out.push(other),
        }
    }
    out
}

/// Compact one-line preview of a ndarray for DOT labels.
fn fmt_array_preview(array: &ArrayD<f64>) -> String {
    const ARRAY_PREVIEW_SHOW_ALL_THRESHOLD: usize = 6;
    const ARRAY_PREVIEW_EDGE_COUNT: usize = 3;
    const ARRAY_PREVIEW_FLOAT_PRECISION: usize = 4;
    let shape: Vec<_> = array.shape().iter().map(|dim| dim.to_string()).collect();
    let shape_str = format!("[{}]", shape.join(","));

    let values: Vec<f64> = array.iter().copied().collect();
    if values.len() <= ARRAY_PREVIEW_SHOW_ALL_THRESHOLD {
        let elems: Vec<_> = values
            .iter()
            .map(|value| format!("{value:.width$}", width = ARRAY_PREVIEW_FLOAT_PRECISION))
            .collect();
        format!("{shape_str} [{}]", elems.join(", "))
    } else {
        let first: Vec<_> = values
            .iter()
            .take(ARRAY_PREVIEW_EDGE_COUNT)
            .map(|value| format!("{value:.width$}", width = ARRAY_PREVIEW_FLOAT_PRECISION))
            .collect();
        let last: Vec<_> = values
            .iter()
            .skip(values.len() - ARRAY_PREVIEW_EDGE_COUNT)
            .map(|value| format!("{value:.width$}", width = ARRAY_PREVIEW_FLOAT_PRECISION))
            .collect();
        let mut parts = first;
        parts.push("...".into());
        parts.extend(last);
        format!("{shape_str} [{}]", parts.join(", "))
    }
}
