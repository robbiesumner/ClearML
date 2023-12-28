pub mod loss_functions;
pub mod models;
pub mod preprocessing;
mod util;

#[derive(Debug, PartialEq)]
pub enum Error {
    DimensionMismatch,
    EmptyVector,
}
