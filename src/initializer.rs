use ndarray::{ArrayD, IxDyn};
use rand::RngExt;

/// Produces an ArrayD given a shape. Used for trainable variables.
pub type VarInit = Box<dyn Fn(&[usize]) -> ArrayD<f64>>;

/// Self-contained producer of an ArrayD. Used for constants.
pub type ConstInit = Box<dyn Fn() -> ArrayD<f64>>;

/// Creates a Glorot (Xavier) uniform initializer for variables.
pub fn glorot_uniform() -> VarInit {
    const GLOROT_UNIFORM_SCALE_FACTOR: f64 = 6.0;
    Box::new(|shape: &[usize]| {
        let limit = (GLOROT_UNIFORM_SCALE_FACTOR / (shape.iter().sum::<usize>() as f64)).sqrt();
        let len: usize = shape.iter().product();
        let data: Vec<f64> = (0..len)
            .map(|_| rand::rng().random_range(-limit..limit))
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
    })
}

/// Creates a zero-filled initializer for variables.
pub fn zeros() -> VarInit {
    Box::new(|shape: &[usize]| ArrayD::zeros(IxDyn(shape)))
}

/// Creates a one-filled initializer for variables.
pub fn ones() -> VarInit {
    Box::new(|shape: &[usize]| ArrayD::ones(IxDyn(shape)))
}

/// Creates an initializer for a scalar constant.
pub fn scalar(value: f64) -> ConstInit {
    Box::new(move || ArrayD::from_elem(IxDyn(&[]), value))
}

/// Creates an initializer for an arbitrary constant array.
pub fn constant_array(data: ArrayD<f64>) -> ConstInit {
    Box::new(move || data.clone())
}
