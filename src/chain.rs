use crate::graph::{Graph, NodeId};
use crate::ops::Op;

/// A fluent builder for constructing sequences of operations.
pub struct Chain<'a> {
    pub(crate) graph: &'a mut Graph,
    pub(crate) prev: NodeId,
}

impl Chain<'_> {
    /// Adds an addition operation to the chain.
    pub fn add(self, other: NodeId) -> Self {
        self.unary(Op::Add, other)
    }
    /// Adds a subtraction operation to the chain.
    pub fn sub(self, other: NodeId) -> Self {
        self.unary(Op::Sub, other)
    }
    /// Adds a multiplication operation to the chain.
    pub fn mul(self, other: NodeId) -> Self {
        self.unary(Op::Mul, other)
    }
    /// Adds a division operation to the chain.
    pub fn div(self, other: NodeId) -> Self {
        self.unary(Op::Div, other)
    }
    /// Adds a matrix multiplication operation to the chain.
    pub fn matmul(self, other: NodeId) -> Self {
        self.unary(Op::MatMul, other)
    }
    /// Adds a negation operation to the chain.
    pub fn neg(self) -> Self {
        self.nullary(Op::Neg)
    }
    /// Adds a ReLU activation operation to the chain.
    pub fn relu(self) -> Self {
        self.nullary(Op::ReLU)
    }
    /// Adds an exponential operation to the chain.
    pub fn exp(self) -> Self {
        self.nullary(Op::Exp)
    }
    /// Adds a sum reduction operation to the chain.
    pub fn sum(self) -> Self {
        self.nullary(Op::Sum)
    }
    /// Adds a mean reduction operation to the chain.
    pub fn mean(self) -> Self {
        self.nullary(Op::Mean)
    }
    /// Adds a power operation to the chain.
    pub fn pow(self, n: f64) -> Self {
        self.nullary(Op::Pow(n))
    }

    /// Ends the chain and returns the final node ID.
    pub fn end(self) -> NodeId {
        self.prev
    }

    fn unary(mut self, op: Op, other: NodeId) -> Self {
        self.prev = self.graph.op(op, &[self.prev, other]);
        self
    }

    fn nullary(mut self, op: Op) -> Self {
        self.prev = self.graph.op(op, &[self.prev]);
        self
    }
}
