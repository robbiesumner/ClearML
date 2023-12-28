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
        .sum::<f64>();

    sum / y.len() as f64
}

/// Calculates the gradient of the mean squared error between two vectors.
///
/// # Formula
///
/// dMSE/dw = 2/n * sum((y_hat - y) * x)
///
/// # Arguments
///
/// * `x` - Vector of input values
/// * `y_hat` - Vector of predicted values
/// * `y` - Vector of actual values
///
/// # Returns
///
/// * Vector of partial derivatives of MSE with respect to each weight
/// * The last element of the vector is the partial derivative with respect to the intercept
pub fn gradient_mse(x: &Vec<Vec<f64>>, y_hat: &Vec<f64>, y: &Vec<f64>) -> Result<Vec<f64>, Error> {
    validate_not_empty(x)?;
    validate_not_empty(y_hat)?;
    validate_not_empty(y)?;
    validate_same_length(x, y)?;
    validate_same_length(y_hat, y)?;

    Ok(_gradient_mse(x, y_hat, y))
}

fn _gradient_mse(x: &Vec<Vec<f64>>, y_hat: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
    let n_samples = x.len() as f64;
    let n_features = x.first().unwrap().len();

    let mut gradient = vec![0.0; x.first().unwrap().len() + 1]; // + 1 for intercept
    for (i, row) in x.iter().enumerate() {
        for (j, x_ij) in row.iter().enumerate() {
            gradient[j] += 2.0 / n_samples * (y_hat[i] - y[i]) * x_ij;
        }
    }

    // intercept
    gradient[n_features] = 2.0 / n_samples * (y_hat.iter().sum::<f64>() - y.iter().sum::<f64>());

    gradient
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
    fn test_gradient_mse_slope() {
        let x: Vec<Vec<f64>> = vec![vec![10.0]];
        let y: Vec<f64> = vec![1.0];
        let y_hat: Vec<f64> = vec![2.0];

        assert_eq!(
            gradient_mse(&x, &y_hat, &y).unwrap()[0],
            20.0
        );
    }
}
