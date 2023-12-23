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
        if x.is_empty() && y.is_empty() {
            return Err(Error::EmptyVector);
        }

        match x.len().cmp(&y.len()) {
            std::cmp::Ordering::Equal => Ok(self._fit(x, y)),
            _ => Err(Error::DimensionMismatch),
        }
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
        if x.is_empty() {
            return Err(Error::EmptyVector);
        }

        match x[0].len().cmp(&self.coefficients.len()) {
            std::cmp::Ordering::Equal => Ok(self._predict(x)),
            _ => Err(Error::DimensionMismatch),
        }
    }

    fn _predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        x.iter()
            .map(|row| {
                row.iter()
                    .zip(&self.coefficients)
                    .map(|(a, x)| a * x)
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
    fn test_predict() {
        let mut model = LinearModel::new();

        model.intercept = 1.0;
        model.coefficients = vec![1.0];

        let x: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0], vec![3.0]];
        let expected: Vec<f64> = vec![2.0, 3.0, 4.0];

        assert_eq!(model.predict(&x).unwrap(), expected);
    }
}
