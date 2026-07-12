use crate::chain::Chain;
use crate::initializer::{ConstInit, VarInit};
use crate::ops::Op;
use ndarray::{ArrayD, IxDyn};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) default_init: Option<VarInit>,
}

pub(crate) struct Node {
    pub name: Option<String>,
    pub kind: NodeKind,
    pub inputs: Vec<NodeId>,
}

pub(crate) enum NodeKind {
    Input,
    Const(ConstInit),
    Var(VarMeta),
    Op(Op),
}

pub(crate) struct VarMeta {
    pub shape: Vec<usize>,
    pub init: Option<VarInit>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            default_init: None,
        }
    }

    /// Set a default initializer for all Var nodes that don't have an explicit one.
    pub fn set_default_init(&mut self, init: VarInit) {
        self.default_init = Some(init);
    }

    /// Create an input placeholder node (data/target fed at runtime).
    pub fn input(&mut self) -> NodeId {
        self.push_node(NodeKind::Input, &[])
    }

    /// Create a constant node with a self-contained initializer.
    pub fn constant(&mut self, init: ConstInit) -> NodeId {
        self.push_node(NodeKind::Const(init), &[])
    }

    /// Convenience: scalar constant.
    pub fn scalar_constant(&mut self, value: f64) -> NodeId {
        self.constant(Box::new(move || ArrayD::from_elem(IxDyn(&[]), value)))
    }

    /// Create a trainable variable with an explicit shape.
    /// Initialization (in priority order):
    ///   1. Use `variable_with` for an explicit initializer
    ///   2. Uses the graph's default initializer (`set_default_init`)
    ///   3. No initializer — user must call `ctx.set_var()` before training
    pub fn variable(&mut self, shape: &[usize]) -> NodeId {
        self.push_var(shape.to_vec(), None)
    }

    /// Create a trainable variable with a shape and an explicit initializer.
    pub fn variable_with(&mut self, shape: &[usize], init: VarInit) -> NodeId {
        self.push_var(shape.to_vec(), Some(init))
    }

    /// Create an op node (the primitive computation unit).
    pub fn op(&mut self, op: Op, inputs: &[NodeId]) -> NodeId {
        self.push_node(NodeKind::Op(op), inputs)
    }

    pub fn add(&mut self, node_a: NodeId, node_b: NodeId) -> NodeId {
        self.op(Op::Add, &[node_a, node_b])
    }
    pub fn sub(&mut self, node_a: NodeId, node_b: NodeId) -> NodeId {
        self.op(Op::Sub, &[node_a, node_b])
    }
    pub fn mul(&mut self, node_a: NodeId, node_b: NodeId) -> NodeId {
        self.op(Op::Mul, &[node_a, node_b])
    }
    pub fn div(&mut self, node_a: NodeId, node_b: NodeId) -> NodeId {
        self.op(Op::Div, &[node_a, node_b])
    }
    pub fn matmul(&mut self, node_a: NodeId, node_b: NodeId) -> NodeId {
        self.op(Op::MatMul, &[node_a, node_b])
    }
    pub fn neg(&mut self, node: NodeId) -> NodeId {
        self.op(Op::Neg, &[node])
    }
    pub fn relu(&mut self, node: NodeId) -> NodeId {
        self.op(Op::ReLU, &[node])
    }
    pub fn exp(&mut self, node: NodeId) -> NodeId {
        self.op(Op::Exp, &[node])
    }
    pub fn pow(&mut self, node: NodeId, exp: f64) -> NodeId {
        self.op(Op::Pow(exp), &[node])
    }
    pub fn sum(&mut self, node: NodeId) -> NodeId {
        self.op(Op::Sum, &[node])
    }
    pub fn mean(&mut self, node: NodeId) -> NodeId {
        self.op(Op::Mean, &[node])
    }

    /// Start a left-to-right chain from `start`.
    /// Each method adds an op node; `.end()` returns the final NodeId.
    ///
    /// ```ignore
    /// let y = g.chain(x).matmul(w).add(b).relu().end();
    /// ```
    pub fn chain(&mut self, start: NodeId) -> Chain<'_> {
        Chain {
            graph: self,
            prev: start,
        }
    }

    /// Set a human-readable name for a node.
    pub fn set_node_name(&mut self, id: NodeId, name: impl Into<String>) {
        if let Some(node) = self.nodes.get_mut(id.0) {
            node.name = Some(name.into());
        }
    }

    /// Attach a name at construction time — useful inline:
    /// `let x = g.with_name(g.input(), "x");`
    pub fn with_name(&mut self, id: NodeId, name: impl Into<String>) -> NodeId {
        self.set_node_name(id, name);
        id
    }

    /// Get the display name for a node (always available — auto-generated
    /// at construction if no explicit name was given).
    pub fn node_name(&self, id: NodeId) -> &str {
        self.nodes[id.0].name.as_deref().unwrap()
    }

    pub(crate) fn resolve_init(&self, meta: &VarMeta) -> Option<ArrayD<f64>> {
        if let Some(init) = &meta.init {
            Some(init(&meta.shape))
        } else {
            self.default_init
                .as_ref()
                .map(|default| default(&meta.shape))
        }
    }

    fn default_name(&self, id: NodeId, kind: &NodeKind) -> String {
        let idx = id.0;
        match kind {
            NodeKind::Input => format!("input_{idx}"),
            NodeKind::Const(_) => format!("const_{idx}"),
            NodeKind::Var(meta) => {
                let shape_str: Vec<_> = meta.shape.iter().map(|d| d.to_string()).collect();
                if shape_str.is_empty() {
                    format!("var_{idx}")
                } else {
                    format!("var_{}_{idx}", shape_str.join("x"))
                }
            }
            NodeKind::Op(op) => format!("{op}_{idx}"),
        }
    }

    fn push_node(&mut self, kind: NodeKind, inputs: &[NodeId]) -> NodeId {
        let id = NodeId(self.nodes.len());
        let name = self.default_name(id, &kind);
        self.nodes.push(Node {
            name: Some(name),
            kind,
            inputs: inputs.to_vec(),
        });
        id
    }

    fn push_var(&mut self, shape: Vec<usize>, init: Option<VarInit>) -> NodeId {
        self.push_node(NodeKind::Var(VarMeta { shape, init }), &[])
    }
}
