use crate::{
    loss_functions::gradient_mse,
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
    /// # Note
    /// 
    /// Does not fit intercept for now! # TODO
    ///
    /// # Arguments
    ///
    /// * `x` - Matrix of features
    /// * `y` - Vector of actual values
    ///
    /// # Requirements
    ///
    /// * `x` must have the same length as `y`
    /// * `x`'s rows must all be same length
    /// 
    /// # Returns
    /// 
    /// * `Ok(&Self)` if successful
    /// * `Err(Error)` if unsuccessful
    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) -> Result<&Self, Error> {
        validate_not_empty(x)?;
        validate_not_empty(y)?;

        validate_same_length(x, y)?;

        // validate all rows have same length
        let first_row = x.first().unwrap();
        x.iter()
            .skip(1)
            .try_for_each(|row| validate_same_length(first_row, row))?;

        Ok(self._fit(x, y, 1000, 0.01, 0.000001))
    }

    fn _fit(
        &mut self,
        x: &Vec<Vec<f64>>,
        y: &Vec<f64>,
        max_iter: u32,
        learning_rate: f64,
        tol: f64,
    ) -> &Self {
        // initialize coefficients to 0
        self.coefficients = vec![0.0; x.first().unwrap().len()];

        for _ in 0..max_iter {
            let y_hat = self.predict(x).unwrap();
            let gradient = gradient_mse(&y_hat, y).unwrap();
            // check if converged
            if gradient.iter().all(|g| g.abs() < tol) {
                break;
            }

            // update coefficients
            self.coefficients = self
                .coefficients
                .iter()
                .zip(&gradient)
                .map(|(a, g)| a - learning_rate * g)
                .collect();

            // update intercept
            // TODO: how to update intercept?
        }

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
    /// * `x`'s rows must all match the length of the coefficients of the `LinearModel`
    /// 
    /// # Returns
    /// 
    /// * `Ok(Vec<f64>)` if successful
    /// * `Err(Error)` if unsuccessful
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
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![1.0, 2.0];

        let mut model = LinearModel::new();

        model.fit(&x, &y).unwrap();

        println!("{:?}", model);
        println!("{:?}", model.predict(&x).unwrap());
    }

    #[test]
    fn test_fit_with_intercept() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![2.0, 4.0];

        let mut model = LinearModel::new();

        model.fit(&x, &y).unwrap();

        println!("{:?}", model);
        println!("{:?}", model.predict(&x).unwrap());
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

    #[test]
    fn test_predict_with_multiple_coefficients() {
        let mut model = LinearModel::new();

        model.intercept = 1.0;
        model.coefficients = vec![1.0, 2.0];

        let x: Vec<Vec<f64>> = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        let expected: Vec<f64> = vec![4.0, 7.0, 10.0];

        assert_eq!(model.predict(&x).unwrap(), expected);
    }
}
