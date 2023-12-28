use crate::{
    util::{validate_not_empty, validate_same_length},
    Error,
};

/// Feature normalization with min max.
/// Scale features to be between 0 and 1.
///
/// # Formula
///
/// x_norm = (x - min(x)) / (max(x) - min(x))
/// 
/// # Arguments
/// 
/// * `x` - Matrix of features.
/// 
/// # Example
/// 
/// ```
/// use clear_ml::preprocessing::scale_min_max;
/// 
/// let mut x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// scale_min_max(&mut x).unwrap();
/// 
/// assert_eq!(x, vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]]);
/// ```
pub fn scale_min_max(x: &mut Vec<Vec<f64>>) -> Result<(), Error> {
    validate_not_empty(x)?;
    let first_row = x.first().unwrap();
    x.iter()
        .skip(1)
        .try_for_each(|row| validate_same_length(first_row, row))?;

    _scale_min_max(x);

    Ok(())
}

pub fn _scale_min_max(x: &mut Vec<Vec<f64>>) -> () {
    // for every column
    for j in 0..x[0].len() {
        let mut min = x[0][j];
        let mut max = x[0][j];

        // min and max
        for i in 1..x.len() {
            if x[i][j] < min {
                min = x[i][j];
            }
            if x[i][j] > max {
                max = x[i][j];
            }
        }

        // normalize between 0 and 1
        for i in 0..x.len() {
            x[i][j] = (x[i][j] - min) / (max - min);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_min_max() {
        let mut x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        scale_min_max(&mut x).unwrap();

        assert_eq!(x, vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]]);
    }
}
