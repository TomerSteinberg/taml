use ndarray::ArrayD;
use ndarray::linalg::Dot;

pub enum Op {
    Add,
    Mul,
    MatMul,
}

impl Op {
    pub fn backward(&self, inputs: &[&ArrayD<f32>], gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        match self {
            Op::MatMul => {vec![gradient.dot(&inputs[1].t()),
                                gradient.dot(&inputs[0].t())]}
            Op::Add => {vec![gradient.clone(), gradient.clone()]}
            Op::Mul => {vec![gradient * inputs[1], gradient * inputs[0]]}
        }

    }
}