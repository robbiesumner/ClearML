use crate::{
    util::{validate_not_empty, validate_same_length},
    Error,
};

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
            coefficients: Vec::new(),
        }
    }

    /// Fit linear model.
    ///
    /// # Arguments
    ///
    /// * `x` - Matrix of features
    /// * `y` - Vector of actual values
    ///
    /// # Requirements
    ///
    /// * `x` must have the same length as `y`
    /// * `x`'s rows must all be same length (not implemented yet!)
    pub fn fit(self, x: &Vec<Vec<f64>>, y: &Vec<f64>) -> Result<Self, Error> {
        validate_not_empty(x)?;
        validate_not_empty(y)?;

        validate_same_length(x, y)?;

        // validate all rows have same length
        let first_row = x.first().unwrap();
        x.iter()
            .skip(1)
            .try_for_each(|row| validate_same_length(first_row, row))?;

        Ok(self._fit(x, y))
    }

    fn _fit(self, _x: &Vec<Vec<f64>>, _y: &Vec<f64>) -> Self {
        // TODO: implement fit
        self
    }

    /// Predict using the linear model.
    ///
    /// # Arguments
    ///
    /// * `x` - Matrix of features
    ///
    /// # Requirements
    ///
    /// * `x`'s rows must all match the length of the coefficients of the `LinearModel` (not implemented yet! only checks first row for now)
    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Result<Vec<f64>, Error> {
        validate_not_empty(x)?;

        // validate all rows are same length as coefficients
        x.iter()
            .try_for_each(|row| validate_same_length(row, &self.coefficients))?;

        Ok(self._predict(x))
    }

    fn _predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        x.iter()
            .map(|row| {
                row.iter()
                    .zip(&self.coefficients)
                    .map(|(x, a)| x * a)
                    .sum::<f64>()
                    + self.intercept
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_fit_empty_vector() {
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];

        assert!(LinearModel::new().fit(&x, &y).is_err());
        assert_eq!(
            LinearModel::new().fit(&x, &y).unwrap_err(),
            Error::EmptyVector
        );
    }

    #[test]
    fn test_fit_dimension_mismatch() {
        let x: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0]];
        let y: Vec<f64> = vec![0.0; 3];

        assert!(LinearModel::new().fit(&x, &y).is_err());
        assert_eq!(
            LinearModel::new().fit(&x, &y).unwrap_err(),
            Error::DimensionMismatch
        );
    }

    #[test]
    fn test_fit() {
        let x: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0]; 3];
        let y: Vec<f64> = vec![0.0; 3];

        assert!(LinearModel::new().fit(&x, &y).is_ok());
    }

    #[test]
    fn test_predict_empty_vector() {
        let x: Vec<Vec<f64>> = vec![];

        assert!(LinearModel::new().predict(&x).is_err());
        assert_eq!(
            LinearModel::new().predict(&x).unwrap_err(),
            Error::EmptyVector
        );
    }

    #[test]
    fn test_predict_different_row_length() {
        let mut model = LinearModel::new();

        model.intercept = 1.0;
        model.coefficients = vec![1.0, 1.0];

        let x: Vec<Vec<f64>> = vec![vec![1.0, 1.0], vec![1.0]];

        assert!(model.predict(&x).is_err());
        assert_eq!(model.predict(&x).unwrap_err(), Error::DimensionMismatch);
    }

    #[test]
    fn test_predict() {
        let mut model = LinearModel::new();

        model.intercept = 1.0;
        model.coefficients = vec![1.0];

        let x: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0], vec![3.0]];
        let expected: Vec<f64> = vec![2.0, 3.0, 4.0];

        assert_eq!(model.predict(&x).unwrap(), expected);
    }
}
