use crate::{
    util::{validate_not_empty, validate_same_length},
    Error,
};

/// Calculates the mean squared error between two vectors.
///
/// # Formula
///
/// MSE = 1/n * sum((y_hat - y)^2)
///
/// # Arguments
///
/// * `y_hat` - Vector of predicted values
/// * `y` - Vector of actual values
/// 
/// # Returns
/// 
/// * Mean squared error
pub fn mean_squared_error(y_hat: &Vec<f64>, y: &Vec<f64>) -> Result<f64, Error> {
    validate_not_empty(y_hat)?;
    validate_not_empty(y)?;
    validate_same_length(y_hat, y)?;

    Ok(_mean_squared_error(y_hat, y))
}

fn _mean_squared_error(y: &Vec<f64>, y_hat: &Vec<f64>) -> f64 {
    let sum = y_hat
        .iter()
        .zip(y)
        .map(|(y_hat, y)| (y_hat - y).powi(2))
        .collect::<Vec<f64>>()
        .iter()
        .sum::<f64>();

    sum / y.len() as f64
}

/// Calculates the gradient of the mean squared error between two vectors.
/// 
/// # Formula
/// 
/// d/dy_hat MSE = 2/n * (y_hat - y)
/// 
/// # Arguments
/// 
/// * `y_hat` - Vector of predicted values
/// * `y` - Vector of actual values
/// 
/// # Returns
/// 
/// * Vector of partial derivatives of MSE with respect to each y_hat
pub fn gradient_mse(y_hat: &Vec<f64>, y: &Vec<f64>) -> Result<Vec<f64>, Error> {
    validate_not_empty(y_hat)?;
    validate_not_empty(y)?;
    validate_same_length(y_hat, y)?;

    Ok(_gradient_mse(y_hat, y))
}

fn _gradient_mse(y_hat: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
    let n = y.len() as f64;

    y_hat.iter()
        .zip(y)
        .map(|(y_hat, y)| 2.0 / n * (y_hat - y))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_empty_vector() {
        let y: Vec<f64> = vec![];
        let y_hat: Vec<f64> = vec![];

        assert!(mean_squared_error(&y, &y_hat).is_err());
        assert_eq!(
            mean_squared_error(&y, &y_hat).unwrap_err(),
            Error::EmptyVector
        );
    }

    #[test]
    fn test_mse_dimension_mismatch() {
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y_hat: Vec<f64> = vec![1.0, 2.0];

        assert!(mean_squared_error(&y, &y_hat).is_err());
        assert_eq!(
            mean_squared_error(&y, &y_hat).unwrap_err(),
            Error::DimensionMismatch
        );
    }

    #[test]
    fn test_mse_is_0_for_same_vectors() {
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y_hat: Vec<f64> = vec![1.0, 2.0, 3.0];

        assert_eq!(mean_squared_error(&y, &y_hat).unwrap(), 0.0);
    }

    #[test]
    fn test_mse_different_vectors() {
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y_hat: Vec<f64> = vec![2.0, 3.0, 4.0];

        assert_eq!(mean_squared_error(&y, &y_hat).unwrap(), 1.0);
    }

    #[test]
    fn test_gradient_mse() {
        let y_hat: Vec<f64> = vec![2.0, 3.0, 4.0];
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];

        assert_eq!(
            gradient_mse(&y_hat, &y).unwrap(),
            vec![2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
        );
    }
}
