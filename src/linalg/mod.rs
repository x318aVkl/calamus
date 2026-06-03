//!
//! Linear algebra datatypes and subroutines.
//! 
//! 
//! 


pub mod vector;
pub mod matrix;
pub mod lu;

use crate::num_traits::FloatNumber;

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




pub trait MatrixFactorization<T> {
    fn allocate(size: usize) -> Self where Self: Sized;

    fn set_matrix<M>(&mut self, matrix: &M) -> Result<(), Error> where M: matrix::Matrix<T>;

    fn factorize(&mut self) -> Result<(), Error> where T: FloatNumber;

    fn solve<X, B>(&self, x: &mut X, b: &B) where X: vector::VectorMut<T>, B: vector::Vector<T>, T: FloatNumber;
}


