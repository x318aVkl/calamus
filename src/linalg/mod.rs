//!
//! Linear algebra datatypes and subroutines.
//! 
//! 
//! 


pub mod vector;
pub mod matrix;
pub mod factorize;
pub mod sparse;

pub use matrix::{Matrix, MatrixMut};
pub use vector::{Vector, VectorMut};


#[derive(Debug)]
pub enum Error {
    ErrorSingularMatrix,
    ErrorSizeInvalid,
}


impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for Error {
}


