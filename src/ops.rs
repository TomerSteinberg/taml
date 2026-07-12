use ndarray::{ArrayD, Ix2, IxDyn};
use ndarray::linalg::Dot;

use std::fmt;

#[derive(Debug)]
pub enum Op {
    Add,
    Mul,
    MatMul,
    Exp,
    ReLU,
    Sub,
    Neg,
    Div,
    Pow(f64),
    Sum,
    Mean,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Add => write!(f, "add"),
            Op::Sub => write!(f, "sub"),
            Op::Mul => write!(f, "mul"),
            Op::Div => write!(f, "div"),
            Op::MatMul => write!(f, "matmul"),
            Op::Neg => write!(f, "neg"),
            Op::Exp => write!(f, "exp"),
            Op::ReLU => write!(f, "relu"),
            Op::Pow(_) => write!(f, "pow"),
            Op::Sum => write!(f, "sum"),
            Op::Mean => write!(f, "mean"),
        }
    }
}

impl Op {
    pub fn compute(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        match self {
            Op::MatMul => {
                // Safely cast dynamic arrays to 2D views for matrix multiplication
                let a = inputs[0].view().into_dimensionality::<Ix2>().expect("MatMul input 0 must be 2D");
                let b = inputs[1].view().into_dimensionality::<Ix2>().expect("MatMul input 1 must be 2D");

                a.dot(&b).into_dyn()
            },
            Op::Add =>
                inputs[0] + inputs[1],
            Op::Mul =>
                inputs[0] * inputs[1],
            Op::Exp => inputs[0].mapv(f64::exp),
            Op::ReLU => inputs[0].mapv(|x| if x > 0.0 { x } else { 0.0 }),
            Op::Sub => inputs[0] - inputs[1],
            Op::Neg => -inputs[0].clone(),
            Op::Div => inputs[0] / inputs[1],
            Op::Pow(n) => inputs[0].mapv(|x| x.powf(*n)),
            Op::Sum => {
                let sum: f64 = inputs[0].iter().sum();
                ArrayD::from_elem(IxDyn(&[]), sum)
            },
            Op::Mean => {
                let sum: f64 = inputs[0].iter().sum();
                let n = inputs[0].len() as f64;
                ArrayD::from_elem(IxDyn(&[]), sum / n)
            },
        }
    }

    pub fn backward(&self, inputs: &[&ArrayD<f64>], gradient: &ArrayD<f64>) -> Vec<ArrayD<f64>>{
        match self {
            Op::MatMul => {
                // Cast inputs and the incoming gradient to 2D matrices
                let grad_2d = gradient.view().into_dimensionality::<Ix2>().expect("Gradient must be 2D");
                let a_2d = inputs[0].view().into_dimensionality::<Ix2>().expect("Input 0 must be 2D");
                let b_2d = inputs[1].view().into_dimensionality::<Ix2>().expect("Input 1 must be 2D");

                // Compute gradients w.r.t to A and B, then convert back to dynamic arrays
                let grad_a = grad_2d.dot(&b_2d.t()).into_dyn();
                let grad_b = a_2d.t().dot(&grad_2d).into_dyn();

                vec![grad_a, grad_b]
            },
            Op::Add => vec![gradient.clone(), gradient.clone()],
            Op::Mul => vec![gradient * inputs[1], gradient * inputs[0]],
            Op::Exp => vec![gradient * &inputs[0].mapv(f64::exp)],
            Op::ReLU => {
                let mask = inputs[0].mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                vec![gradient * mask]
            },
            Op::Sub => vec![gradient.clone(), -gradient],
            Op::Neg => vec![-gradient.clone()],
            Op::Div => {
                let grad_a = gradient / inputs[1];
                let grad_b = gradient * &(-inputs[0] / (inputs[1] * inputs[1]));
                vec![grad_a, grad_b]
            },
            Op::Pow(n) => {
                let factor = inputs[0].mapv(|x| *n * x.powf(n - 1.0));
                vec![gradient * factor]
            },
            Op::Sum => {
                let grad_val = gradient.iter().next().copied().unwrap_or(0.0);
                vec![ArrayD::from_elem(inputs[0].raw_dim(), grad_val)]
            },
            Op::Mean => {
                let grad_val = gradient.iter().next().copied().unwrap_or(0.0);
                let n = inputs[0].len() as f64;
                vec![ArrayD::from_elem(inputs[0].raw_dim(), grad_val / n)]
            },
        }
    }

}