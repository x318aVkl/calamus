use crate::linalg::vector::{Vector, VectorMut};



/// An immutable view window that acts like a row major matrix
#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a, T> {
    data: &'a [T],
    ncolumns: usize,
}


/// An mutable view window that acts like a row major matrix
#[derive(Debug)]
pub struct MatrixViewMut<'a, T> {
    data: &'a mut [T],
    ncolumns: usize,
}

/// a matrix with owned data, of runtime size
#[derive(Debug)]
pub struct DynamicMatrix<T> {
    data: Vec<T>,
    ncolumns: usize,
}

/// a matrix with owned data, of compile-time size
pub struct StaticMatrix<T, const NR: usize, const NC: usize> {
    data: [[T; NC]; NR],
}


impl<T> DynamicMatrix<T> {
    pub fn new(value: T, shape: [usize; 2]) -> Self where T: Copy {
        Self { data: vec![value; shape[0]*shape[1]], ncolumns: shape[1] }
    }
    pub fn eye(shape: [usize; 2]) -> Self where T: Copy + From<u8> {
        let mut m = Self::new(T::from(0), shape);
        for i in 0..shape[0].min(shape[1]) {
            m[[i, i]] = T::from(1);
        }
        m
    }
}


impl<T> From<([usize; 2], Vec<T>)> for DynamicMatrix<T> {
    fn from(value: ([usize; 2], Vec<T>)) -> Self {
        Self { data: value.1, ncolumns: value.0[1] }
    }
}


pub trait Matrix<T>: core::ops::Index<[usize; 2], Output = T> {
    fn shape(&self) -> [usize; 2];

    fn data(&self) -> &[T];

    fn view<'a>(&'a self) -> MatrixView<'a, T> {
        MatrixView { data: self.data(), ncolumns: self.shape()[1] }
    }


    fn imul<'a, A, B>(&'a self, result: &mut A, rhs: &B) where A: VectorMut<T>, B: Vector<T>, T: core::ops::AddAssign + core::ops::Mul<Output = T> + Copy + From<u8> {
        assert_eq!(self.shape()[0], result.len());
        assert_eq!(self.shape()[1], rhs.len());

        for i in 0..result.len() {
            let mut ai = T::from(0);
            for j in 0..rhs.len() {
                ai += self[[i, j]] * rhs[j];
            }
            result[i] = ai;
        }
    }


    fn imul_left<'a, A, B>(&'a self, result: &mut A, lhs: &B) where A: VectorMut<T>, B: Vector<T>, T: core::ops::AddAssign + core::ops::Mul<Output = T> + Copy + From<u8> {
        assert_eq!(self.shape()[0], lhs.len());
        assert_eq!(self.shape()[1], result.len());

        for i in 0..self.shape()[0] {
            for j in 0..self.shape()[1] {
                result[j] += lhs[i] * self[[i, j]];
            }
        }
    }
}

pub trait MatrixMut<T>: Matrix<T> + core::ops::IndexMut<[usize; 2]> {

    fn data_mut(&mut self) -> &mut [T];

    fn view_mut<'a>(&'a mut self) -> MatrixViewMut<'a, T> {
        let ncolumns = self.shape()[1];
        MatrixViewMut { data: self.data_mut(), ncolumns, }
    }
}



impl<T, const NR: usize, const NC: usize> core::ops::Index<[usize; 2]> for StaticMatrix<T, NR, NC> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[index[0]][index[1]]
    }
}


impl<T, const NR: usize, const NC: usize> core::ops::IndexMut<[usize; 2]> for StaticMatrix<T, NR, NC> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.data[index[0]][index[1]]
    }
}

impl<T, const NR: usize, const NC: usize> Matrix<T> for StaticMatrix<T, NR, NC> {
    fn shape(&self) -> [usize; 2] {
        [NR, NC]
    }
    fn data(&self) -> &[T] {
        self.data.as_flattened()
    }
}

impl<T, const NR: usize, const NC: usize> MatrixMut<T> for StaticMatrix<T, NR, NC> {
    fn data_mut(&mut self) -> &mut [T] {
        self.data.as_flattened_mut()
    }
}



impl<T> core::ops::Index<[usize; 2]> for DynamicMatrix<T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[index[0]*self.ncolumns + index[1]]
    }
}


impl<T> core::ops::IndexMut<[usize; 2]> for DynamicMatrix<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.data[index[0]*self.ncolumns + index[1]]
    }
}

impl<T> Matrix<T> for DynamicMatrix<T> {
    fn shape(&self) -> [usize; 2] {
        [self.data.len() / self.ncolumns, self.ncolumns]
    }
    fn data(&self) -> &[T] {
        &self.data
    }
}

impl<T> MatrixMut<T> for DynamicMatrix<T> {
    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}


impl<'a, T> std::ops::Index<[usize; 2]> for MatrixView<'a, T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[index[0]*self.ncolumns + index[1]]
    }
}


impl<'a, T> std::ops::Index<[usize; 2]> for MatrixViewMut<'a, T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[index[0]*self.ncolumns + index[1]]
    }
}


impl<'a, T> std::ops::IndexMut<[usize; 2]> for MatrixViewMut<'a, T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.data[index[0]*self.ncolumns + index[1]]
    }
}



