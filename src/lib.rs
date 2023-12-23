pub mod loss_functions;
pub mod models;
mod util;

#[derive(Debug, PartialEq)]
pub enum Error {
    DimensionMismatch,
    EmptyVector,
}
