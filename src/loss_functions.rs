use std::iter::zip;

#[derive(Debug, PartialEq)]
pub enum Error {
    DimensionMismatch,
    EmptyVector,
}

/// Calculates the mean squared error between two vectors.
/// 
/// # Formula
/// 
/// MSE = 1/n * sum((y - y_hat)^2)
/// 
/// # Arguments
/// 
/// * `y` - Vector of actual values
/// * `y_hat` - Vector of predicted values 
pub fn mean_squared_error(y: &Vec<f64>, y_hat: &Vec<f64>) -> Result<f64, Error> {
    if y.is_empty() && y_hat.is_empty() {
        return Err(Error::EmptyVector);
    }

    match y.len().cmp(&y_hat.len()) {
        std::cmp::Ordering::Equal => Ok(_mean_squared_error(y, y_hat)),
        _ => Err(Error::DimensionMismatch), // not equal
    }
}

fn _mean_squared_error(y: &Vec<f64>, y_hat: &Vec<f64>) -> f64 {
    let sum = zip(y, y_hat)
        .into_iter()
        .map(|(y, y_hat)| (y - y_hat).powi(2))
        .collect::<Vec<f64>>()
        .iter()
        .sum::<f64>();

    sum / y.len() as f64
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
}
