pub mod loss_functions;
pub mod models;

#[derive(Debug, PartialEq)]
pub enum Error {
    DimensionMismatch,
    EmptyVector,
}
