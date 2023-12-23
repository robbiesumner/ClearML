use crate::Error;

// functions to validate data

/// Checks if a vector is empty.
///
/// # Arguments
///
/// * `x` - Vector to check
///
/// # Returns
///
/// * `Ok(())` if vector is not empty
/// * `Err(Error::EmptyVector)` if vector is empty
pub fn validate_not_empty<T>(x: &Vec<T>) -> Result<(), Error> {
    if x.is_empty() {
        Err(Error::EmptyVector)
    } else {
        Ok(())
    }
}

/// Checks if two vectors have the same length.
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * `Ok(())` if vectors have the same length
/// * `Err(Error::DimensionMismatch)` if vectors do not have the same length
pub fn validate_same_length<T1, T2>(x: &[T1], y: &[T2]) -> Result<(), Error> {
    if x.len().cmp(&y.len()).is_ne() {
        Err(Error::DimensionMismatch)
    } else {
        Ok(())
    }
}
