//!
//! Compressed sparse row matrix type
//! 
//! 
//! 
//! ```rust
//! 
//! use scixl::linalg::sparse::csr::SparseMatrixCSR;
//! 
//! let mut a = SparseMatrixCSR::<f64>::new();
//! 
//! a.push_to_row(0, 1.0);
//! a.push_to_row(1, 0.5);
//! a.close_row();
//! 
//! a.push_to_row(1, 1.0);
//! a.close_row();
//! 
//! assert!(a[[0, 1]] == 0.5);
//! 
//! ```
//! 

use std::collections::HashSet;

use crate::{linalg::{Vector, VectorMut, sparse::sparsity::Sparsity}};



pub struct SparseMatrixCSR<T> {
    sparsity: Sparsity,
    values: Vec<T>,
    tmp_buffer: Vec<(usize, T)>,
}



impl<T> SparseMatrixCSR<T> {

    pub fn new() -> Self {
        Self { sparsity: Sparsity::new(), values: vec![], tmp_buffer: vec![], }
    }


    pub fn nrows(&self) -> usize {
        self.sparsity.major_len()
    }

    pub fn ncols(&self) -> usize {
        if self.sparsity.minor_len() == 0 {
            0
        } else {
            self.sparsity.max_minor() + 1
        }
    }

    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = (usize, &T)> {
        self.sparsity.major_range_flat(row).map(|k| (self.sparsity.flat_index(k), &self.values[k]))
    }

    pub fn push_to_row(&mut self, column: usize, value: T) {
        self.sparsity.push_to_major(column);
        self.values.push(value);
    }

    pub fn close_row(&mut self) where T: Copy + From<u8> + std::fmt::Debug {
        self.sparsity.close_major_and_sort(&mut self.values, &mut self.tmp_buffer);
    }

    /// In-place multiply self with rhs, storing the result in result
    pub fn imul(&self, result: &mut impl VectorMut<T>, rhs: &impl Vector<T>) where T: From<u8> + core::ops::Mul<T, Output = T> + core::ops::AddAssign<T> + Copy + std::fmt::Debug {
        for row in 0..self.nrows() {
            let mut ri = T::from(0);

            for (j, aij) in self.iter_row(row) {
                ri += *aij * rhs[j];
            }

            result[row] = ri;
        }
    }

    // compute the transpose, costly operation
    pub fn transpose(&self) -> Self where T: Copy + From<u8> + core::fmt::Debug {

        // compute the transposed matrix
        let mut cols: Vec<HashSet<usize>> = vec![HashSet::new(); self.ncols()];

        for i in 0..self.nrows() {
            for (j, _) in self.iter_row(i) {
                cols[j].insert(i);
            }
        }

        let mut sparsity = Sparsity::new();
        let mut values = vec![];

        let mut tmp_buffer = vec![];

        for j in 0..self.ncols() {
            let start = values.len();
            for i in &cols[j] {
                sparsity.push_to_major(*i);
                values.push(self[[*i, j]])
            }
            let end = values.len();
            sparsity.close_major_and_sort(&mut values[start..end], &mut tmp_buffer);
        }

        SparseMatrixCSR {
            sparsity,
            values,
            tmp_buffer: vec![],
        }
    }

}


impl<T> std::ops::Index<[usize; 2]> for SparseMatrixCSR<T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        match self.sparsity.major_range(index[0]).binary_search(&index[1]) {
            Ok(k) => &self.values[self.sparsity.major_start(index[0]) + k],
            Err(_) => panic!("Error in SparseMatrixCSR<T>::index(), index ({}, {}) not found", index[0], index[1])
        }
    }
}





