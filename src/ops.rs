use ndarray::linalg::Dot;
use ndarray::{ArrayD, Axis, Ix2, IxDyn};

use std::fmt;

/// Mathematical operations supported by the computation graph.
#[derive(Debug)]
pub enum Op {
    /// Addition
    Add,
    /// Multiplication
    Mul,
    /// Matrix multiplication
    MatMul,
    /// Exponential
    Exp,
    /// Rectified Linear Unit
    ReLU,
    /// Subtraction
    Sub,
    /// Negation
    Neg,
    /// Division
    Div,
    /// Power
    Pow(f64),
    /// Sum reduction
    Sum,
    /// Mean reduction
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

/// Reduce `grad` over any dimensions that were broadcast during the forward
/// pass, producing a gradient whose shape matches `target`.
///
/// ndarray implicitly broadcasts during element-wise operations. The backward
/// pass must undo this: wherever the target has size 1 but the gradient has
/// a larger size, the gradient is summed along that axis.
pub fn unbroadcast(grad: ArrayD<f64>, target: &[usize]) -> ArrayD<f64> {
    let mut grad = grad;
    let mut padded = target.to_vec();
    while padded.len() < grad.ndim() {
        padded.insert(0, 1);
    }
    for axis in (0..padded.len()).rev() {
        if padded[axis] == 1 && grad.shape()[axis] > 1 {
            grad = grad.sum_axis(Axis(axis));
        }
    }
    grad.into_shape_with_order(IxDyn(target)).unwrap()
}

impl Op {
    /// Compute the forward pass for this operation.
    pub fn compute(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        match self {
            Op::MatMul => {
                // Safely cast dynamic arrays to 2D views for matrix multiplication
                let a = inputs[0]
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("MatMul input 0 must be 2D");
                let b = inputs[1]
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("MatMul input 1 must be 2D");

                a.dot(&b).into_dyn()
            }
            Op::Add => inputs[0] + inputs[1],
            Op::Mul => inputs[0] * inputs[1],
            Op::Exp => inputs[0].mapv(f64::exp),
            Op::ReLU => inputs[0].mapv(|x| if x > 0.0 { x } else { 0.0 }),
            Op::Sub => inputs[0] - inputs[1],
            Op::Neg => -inputs[0].clone(),
            Op::Div => inputs[0] / inputs[1],
            Op::Pow(n) => inputs[0].mapv(|x| x.powf(*n)),
            Op::Sum => {
                let sum: f64 = inputs[0].iter().sum();
                ArrayD::from_elem(IxDyn(&[]), sum)
            }
            Op::Mean => {
                let sum: f64 = inputs[0].iter().sum();
                let n = inputs[0].len() as f64;
                ArrayD::from_elem(IxDyn(&[]), sum / n)
            }
        }
    }

    /// Compute the backward pass for this operation, returning gradients for inputs.
    pub fn backward(&self, inputs: &[&ArrayD<f64>], gradient: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        match self {
            Op::MatMul => {
                // Cast inputs and the incoming gradient to 2D matrices
                let grad_2d = gradient
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("Gradient must be 2D");
                let a_2d = inputs[0]
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("Input 0 must be 2D");
                let b_2d = inputs[1]
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("Input 1 must be 2D");

                // Compute gradients w.r.t to A and B, then convert back to dynamic arrays
                let grad_a = grad_2d.dot(&b_2d.t()).into_dyn();
                let grad_b = a_2d.t().dot(&grad_2d).into_dyn();

                vec![grad_a, grad_b]
            }
            Op::Add => {
                let grad0 = gradient.clone();
                let grad1 = gradient.clone();
                vec![
                    unbroadcast(grad0, inputs[0].shape()),
                    unbroadcast(grad1, inputs[1].shape()),
                ]
            }
            Op::Mul => {
                let grad0 = gradient * inputs[1];
                let grad1 = gradient * inputs[0];
                vec![
                    unbroadcast(grad0, inputs[0].shape()),
                    unbroadcast(grad1, inputs[1].shape()),
                ]
            }
            Op::Exp => vec![gradient * &inputs[0].mapv(f64::exp)],
            Op::ReLU => {
                let mask = inputs[0].mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                vec![gradient * mask]
            }
            Op::Sub => {
                vec![
                    unbroadcast(gradient.clone(), inputs[0].shape()),
                    unbroadcast(-gradient.clone(), inputs[1].shape()),
                ]
            }
            Op::Neg => vec![-gradient.clone()],
            Op::Div => {
                let grad_a = unbroadcast(gradient / inputs[1], inputs[0].shape());
                let raw = gradient * &(-inputs[0] / (inputs[1] * inputs[1]));
                let grad_b = unbroadcast(raw, inputs[1].shape());
                vec![grad_a, grad_b]
            }
            Op::Pow(n) => {
                let factor = inputs[0].mapv(|x| *n * x.powf(n - 1.0));
                vec![gradient * factor]
            }
            Op::Sum => {
                let grad_val = gradient.iter().next().copied().unwrap_or(0.0);
                vec![ArrayD::from_elem(inputs[0].raw_dim(), grad_val)]
            }
            Op::Mean => {
                let grad_val = gradient.iter().next().copied().unwrap_or(0.0);
                let n = inputs[0].len() as f64;
                vec![ArrayD::from_elem(inputs[0].raw_dim(), grad_val / n)]
            }
        }
    }
}
