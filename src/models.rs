use ndarray::Array;

use super::Error;

/// Linear model.
///
/// # Formula
///
/// y = a0(intercept) + a1*x1 + a2*x2 + ... + an*xn
///
#[derive(Debug)]
pub struct LinearModel {
    pub intercept: f64,
    pub coefficients: Vec<f64>,
}

impl LinearModel {
    /// Create a new linear model.
    pub fn new() -> Self {
        LinearModel {
            intercept: 0.0,
            coefficients: vec![],
        }
    }

    /// Fit linear model.
    ///
    /// # Arguments
    ///
    /// * `x` - Matrix of features
    /// * `y` - Vector of actual values
    pub fn fit(self, x: &Array<f64, ndarray::Ix2>, y: &Vec<f64>) -> Result<Self, Error> {
        if x.is_empty() && y.is_empty() {
            return Err(Error::EmptyVector);
        }

        match x.len_of(ndarray::Axis(0)).cmp(&y.len()) {
            std::cmp::Ordering::Equal => Ok(self._fit(x, y)),
            _ => Err(Error::DimensionMismatch),
        }
    }

    fn _fit(self, _x: &Array<f64, ndarray::Ix2>, _y: &Vec<f64>) -> Self {
        // TODO: implement fit
        self
    }

    /// Predict using the linear model.
    ///
    /// # Arguments
    ///
    /// * `x` - Matrix of features
    pub fn predict(&self, _x: &Vec<f64>) -> Result<Vec<f64>, Error> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_empty_vector() {
        let x: Array<f64, ndarray::Ix2> = Array::zeros((0, 0));
        let y: Vec<f64> = vec![];

        assert!(LinearModel::new().fit(&x, &y).is_err());
        assert_eq!(
            LinearModel::new().fit(&x, &y).unwrap_err(),
            Error::EmptyVector
        );
    }

    #[test]
    fn test_fit_dimension_mismatch() {
        let x: Array<f64, ndarray::Ix2> = Array::zeros((2, 2));
        let y: Vec<f64> = vec![0.0; 3];

        assert!(LinearModel::new().fit(&x, &y).is_err());
        assert_eq!(
            LinearModel::new().fit(&x, &y).unwrap_err(),
            Error::DimensionMismatch
        );
    }

    #[test]
    fn test_fit() {
        let x: Array<f64, ndarray::Ix2> = Array::zeros((3, 2));
        let y: Vec<f64> = vec![0.0; 3];

        assert!(LinearModel::new().fit(&x, &y).is_ok());
    }
}
