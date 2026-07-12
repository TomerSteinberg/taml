use ndarray::{ArrayD, IxDyn};
use taml::ops::Op;

fn arr1(data: &[f64]) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap()
}

fn arr2(data: &[f64], rows: usize, cols: usize) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(&[rows, cols]), data.to_vec()).unwrap()
}

fn approx_eq(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-10, "{x} != {y}");
    }
}

// =========================================================================
// Compute tests
// =========================================================================

#[test]
fn compute_add() {
    let result = Op::Add.compute(&[&arr1(&[1.0, 2.0, 3.0]), &arr1(&[4.0, 5.0, 6.0])]);
    approx_eq(result.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
}

#[test]
fn compute_sub() {
    let result = Op::Sub.compute(&[&arr1(&[5.0, 7.0, 9.0]), &arr1(&[4.0, 5.0, 6.0])]);
    approx_eq(result.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn compute_mul() {
    let result = Op::Mul.compute(&[&arr1(&[2.0, 3.0, 4.0]), &arr1(&[5.0, 6.0, 7.0])]);
    approx_eq(result.as_slice().unwrap(), &[10.0, 18.0, 28.0]);
}

#[test]
fn compute_div() {
    let result = Op::Div.compute(&[&arr1(&[10.0, 18.0]), &arr1(&[2.0, 3.0])]);
    approx_eq(result.as_slice().unwrap(), &[5.0, 6.0]);
}

#[test]
fn compute_neg() {
    let result = Op::Neg.compute(&[&arr1(&[1.0, -2.0, 3.0])]);
    approx_eq(result.as_slice().unwrap(), &[-1.0, 2.0, -3.0]);
}

#[test]
fn compute_exp() {
    let result = Op::Exp.compute(&[&arr1(&[0.0, 1.0])]);
    approx_eq(result.as_slice().unwrap(), &[1.0, std::f64::consts::E]);
}

#[test]
fn compute_relu() {
    let result = Op::ReLU.compute(&[&arr1(&[-1.0, 0.0, 2.0])]);
    approx_eq(result.as_slice().unwrap(), &[0.0, 0.0, 2.0]);
}

#[test]
fn compute_relu_all_negative() {
    let result = Op::ReLU.compute(&[&arr1(&[-3.0, -2.0, -1.0])]);
    approx_eq(result.as_slice().unwrap(), &[0.0, 0.0, 0.0]);
}

#[test]
fn compute_pow_integer() {
    let result = Op::Pow(2.0).compute(&[&arr1(&[2.0, 3.0, 4.0])]);
    approx_eq(result.as_slice().unwrap(), &[4.0, 9.0, 16.0]);
}

#[test]
fn compute_pow_fractional() {
    let result = Op::Pow(0.5).compute(&[&arr1(&[4.0, 9.0])]);
    approx_eq(result.as_slice().unwrap(), &[2.0, 3.0]);
}

#[test]
fn compute_pow_zero() {
    let result = Op::Pow(0.0).compute(&[&arr1(&[2.0, 3.0])]);
    approx_eq(result.as_slice().unwrap(), &[1.0, 1.0]);
}

#[test]
fn compute_matmul_square() {
    let a = arr2(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = arr2(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let result = Op::MatMul.compute(&[&a, &b]);
    approx_eq(result.as_slice().unwrap(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn compute_matmul_non_square() {
    let a = arr2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = arr2(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let result = Op::MatMul.compute(&[&a, &b]);
    approx_eq(result.as_slice().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn compute_matmul_vector_as_matrix() {
    let a = arr2(&[2.0, 3.0, 4.0], 1, 3);
    let b = arr2(&[1.0, 2.0, 3.0], 3, 1);
    let result = Op::MatMul.compute(&[&a, &b]);
    approx_eq(result.as_slice().unwrap(), &[20.0]);
}

#[test]
fn compute_sum() {
    let result = Op::Sum.compute(&[&arr1(&[1.0, 2.0, 3.0])]);
    assert_eq!(result.ndim(), 0);
    approx_eq(result.as_slice().unwrap(), &[6.0]);
}

#[test]
fn compute_mean() {
    let result = Op::Mean.compute(&[&arr1(&[1.0, 2.0, 3.0])]);
    assert_eq!(result.ndim(), 0);
    approx_eq(result.as_slice().unwrap(), &[2.0]);
}

// =========================================================================
// Backward tests
// =========================================================================

#[test]
fn backward_add() {
    let grads = Op::Add.backward(&[&arr1(&[1.0, 2.0]), &arr1(&[3.0, 4.0])], &arr1(&[2.0, 5.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[2.0, 5.0]);
    approx_eq(grads[1].as_slice().unwrap(), &[2.0, 5.0]);
}

#[test]
fn backward_sub() {
    let grads = Op::Sub.backward(&[&arr1(&[5.0, 7.0]), &arr1(&[4.0, 5.0])], &arr1(&[1.0, 1.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[1.0, 1.0]);
    approx_eq(grads[1].as_slice().unwrap(), &[-1.0, -1.0]);
}

#[test]
fn backward_mul() {
    let grads = Op::Mul.backward(&[&arr1(&[2.0, 3.0]), &arr1(&[5.0, 6.0])], &arr1(&[1.0, 1.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[5.0, 6.0]);
    approx_eq(grads[1].as_slice().unwrap(), &[2.0, 3.0]);
}

#[test]
fn backward_div() {
    let grads = Op::Div.backward(&[&arr1(&[10.0, 18.0]), &arr1(&[2.0, 3.0])], &arr1(&[1.0, 1.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[0.5, 1.0 / 3.0]);
    approx_eq(grads[1].as_slice().unwrap(), &[-2.5, -2.0]);
}

#[test]
fn backward_neg() {
    let grads = Op::Neg.backward(&[&arr1(&[1.0, -2.0, 3.0])], &arr1(&[1.0, 1.0, 1.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[-1.0, -1.0, -1.0]);
}

#[test]
fn backward_exp() {
    let grads = Op::Exp.backward(&[&arr1(&[0.0, 1.0])], &arr1(&[1.0, 1.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[1.0, std::f64::consts::E]);
}

#[test]
fn backward_relu_positive() {
    let grads = Op::ReLU.backward(&[&arr1(&[1.0, 2.0])], &arr1(&[3.0, 4.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[3.0, 4.0]);
}

#[test]
fn backward_relu_mixed() {
    let grads = Op::ReLU.backward(&[&arr1(&[-1.0, 0.0, 2.0])], &arr1(&[1.0, 1.0, 1.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[0.0, 0.0, 1.0]);
}

#[test]
fn backward_relu_all_negative() {
    let grads = Op::ReLU.backward(&[&arr1(&[-3.0, -2.0])], &arr1(&[5.0, 6.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[0.0, 0.0]);
}

#[test]
fn backward_pow() {
    let grads = Op::Pow(3.0).backward(&[&arr1(&[2.0, 3.0, 4.0])], &arr1(&[1.0, 1.0, 1.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[12.0, 27.0, 48.0]);
}

#[test]
fn backward_pow_non_unity_grad() {
    let grads = Op::Pow(2.0).backward(&[&arr1(&[2.0, 3.0])], &arr1(&[2.0, 3.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[8.0, 18.0]);
}

#[test]
fn backward_matmul_square() {
    let a = arr2(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = arr2(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let grad = arr2(&[1.0, 0.0, 0.0, 1.0], 2, 2);
    let grads = Op::MatMul.backward(&[&a, &b], &grad);
    approx_eq(grads[0].as_slice().unwrap(), &[5.0, 7.0, 6.0, 8.0]);
    approx_eq(grads[1].as_slice().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn backward_matmul_non_square() {
    let a = arr2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = arr2(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let grad = arr2(&[1.0, 1.0, 1.0, 1.0], 2, 2);
    let grads = Op::MatMul.backward(&[&a, &b], &grad);
    approx_eq(grads[0].as_slice().unwrap(), &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0]);
    approx_eq(grads[1].as_slice().unwrap(), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
}

#[test]
fn backward_sum() {
    let grads = Op::Sum.backward(&[&arr1(&[1.0, 2.0, 3.0])], &arr1(&[1.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[1.0, 1.0, 1.0]);
}

#[test]
fn backward_sum_scaled() {
    let grads = Op::Sum.backward(&[&arr1(&[1.0, 2.0, 3.0])], &arr1(&[5.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[5.0, 5.0, 5.0]);
}

#[test]
fn backward_mean() {
    let grads = Op::Mean.backward(&[&arr1(&[1.0, 2.0, 3.0])], &arr1(&[1.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
}

#[test]
fn backward_mean_scaled() {
    let grads = Op::Mean.backward(&[&arr1(&[1.0, 2.0, 3.0, 4.0])], &arr1(&[2.0]));
    approx_eq(grads[0].as_slice().unwrap(), &[0.5, 0.5, 0.5, 0.5]);
}

// =========================================================================
// #[should_panic] — shape errors
// =========================================================================

#[test]
#[should_panic(expected = "MatMul input 0 must be 2D")]
fn matmul_panics_on_1d_input_a() {
    let a = arr1(&[1.0, 2.0]);
    let b = arr2(&[3.0, 4.0, 5.0, 6.0], 2, 2);
    Op::MatMul.compute(&[&a, &b]);
}

#[test]
#[should_panic(expected = "MatMul input 1 must be 2D")]
fn matmul_panics_on_1d_input_b() {
    let a = arr2(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = arr1(&[1.0, 2.0]);
    Op::MatMul.compute(&[&a, &b]);
}
