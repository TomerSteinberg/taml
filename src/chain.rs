use crate::graph::{Graph, NodeId};
use crate::ops::Op;

pub struct Chain<'a> {
    pub(crate) graph: &'a mut Graph,
    pub(crate) prev: NodeId,
}

impl Chain<'_> {
    pub fn add(self, other: NodeId) -> Self { self.unary(Op::Add, other) }
    pub fn sub(self, other: NodeId) -> Self { self.unary(Op::Sub, other) }
    pub fn mul(self, other: NodeId) -> Self { self.unary(Op::Mul, other) }
    pub fn div(self, other: NodeId) -> Self { self.unary(Op::Div, other) }
    pub fn matmul(self, other: NodeId) -> Self { self.unary(Op::MatMul, other) }
    pub fn neg(self) -> Self { self.nullary(Op::Neg) }
    pub fn relu(self) -> Self { self.nullary(Op::ReLU) }
    pub fn exp(self) -> Self { self.nullary(Op::Exp) }
    pub fn sum(self) -> Self { self.nullary(Op::Sum) }
    pub fn mean(self) -> Self { self.nullary(Op::Mean) }
    pub fn pow(self, n: f64) -> Self { self.nullary(Op::Pow(n)) }

    pub fn end(self) -> NodeId { self.prev }

    fn unary(mut self, op: Op, other: NodeId) -> Self {
        self.prev = self.graph.op(op, &[self.prev, other]);
        self
    }

    fn nullary(mut self, op: Op) -> Self {
        self.prev = self.graph.op(op, &[self.prev]);
        self
    }
}