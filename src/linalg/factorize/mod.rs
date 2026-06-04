pub mod lu;

use super::Error;
use super::Matrix;
use super::{Vector, VectorMut};

use crate::num_traits::FloatNumber;

pub trait MatrixFactorization<T> {
    fn allocate(size: usize) -> Self where Self: Sized;

    fn set_matrix<M>(&mut self, matrix: &M) -> Result<(), Error> where M: Matrix<T>;

    fn factorize(&mut self) -> Result<(), Error> where T: FloatNumber;
    
    fn solve<X, B>(&self, x: &mut X, b: &B) where X: VectorMut<T>, B: Vector<T>, T: FloatNumber;
}



