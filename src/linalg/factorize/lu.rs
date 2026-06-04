


use crate::{linalg::{matrix::DynamicMatrix, vector::VectorMut}, num_traits::FloatNumber};

use crate::linalg::{Error, Vector, matrix::{Matrix, MatrixMut}};

use super::MatrixFactorization;


pub struct DynamicLu<T> {
    pub lu: DynamicMatrix<T>,
    pub p: Vec<usize>,
}


impl<T> MatrixFactorization<T> for DynamicLu<T> where T: From<u8> + Copy {
    fn allocate(size: usize) -> Self where Self: Sized {
        Self {
            lu: DynamicMatrix::new(T::from(0), [size, size]),
            p: vec![0; size + 1],
        }
    }
    fn set_matrix<M>(&mut self, matrix: &M) -> Result<(), Error> where M: Matrix<T> {
        let n = matrix.shape()[0];
        if matrix.shape()[0] != self.lu.shape()[0] {
            return Err(Error::ErrorSizeInvalid);
        }
        if matrix.shape()[1] != self.lu.shape()[1] {
            return Err(Error::ErrorSizeInvalid);
        }
        for i in 0..n {
            for j in 0..n {
                self.lu[[i, j]] = matrix[[i, j]];
            }
        }
        Ok(())
    }
    fn factorize(&mut self) -> Result<(), Error> where T: FloatNumber {
        plu_decompose(&mut self.lu, &mut self.p)
    }
    fn solve<X, B>(&self, x: &mut X, b: &B) where X: VectorMut<T>, B: Vector<T>, T: FloatNumber {
        plu_solve(x, b, &self.lu, &self.p);
    }
}






// performs the plu decomposition inplace
fn plu_decompose<T>(
    a: &mut impl MatrixMut<T>,
    p: &mut [usize],
) -> Result<(), Error> 
where T: FloatNumber,
{

    let tol = T::EPSILON;

    let n = a.shape()[0];

    if (n + 1) != p.len() {
        return Err(Error::ErrorSizeInvalid);
    }
    if a.shape()[0] != a.shape()[1] {
        return Err(Error::ErrorSizeInvalid);
    }

    for i in 0..n {
        p[i] = i;
    }
    p[n] = n;

    for i in 0..n {

        let mut maxa = T::from(0);
        let mut imax = i;

        for k in i..n {
            let absa = a[[k, i]].float_abs();
            if absa > maxa {
                maxa = absa;
                imax = k;
            }
        }

        if maxa < tol {
            return Err(Error::ErrorSingularMatrix);
        }

        if imax != i {
            // permutation
            let j = p[i];
            p[i] = p[imax];
            p[imax] = j;

            // swap rows
            for j in 0..n {
                let atmp = a[[i, j]];
                a[[i, j]] = a[[imax, j]];
                a[[imax, j]] = atmp;
            }

            p[n] += 1;
        }

        for j in (i+1)..n {
            let aii = a[[i, i]];
            a[[j, i]] /= aii;

            for k in (i+1)..n {
                let v = a[[j, i]] * a[[i, k]];
                a[[j, k]] -= v;
            }
        }

    }


    Ok(())
}



fn plu_solve<T>(
    x: &mut impl VectorMut<T>, 
    b: &impl Vector<T>, 
    lu: &impl Matrix<T>, 
    p: &[usize]
)
where T: FloatNumber
{

    let n = x.len();

    for i in 0..n {

        x[i] = b[p[i]];

        for k in 0..i {
            let v = lu[[i, k]] * x[k];
            x[i] -= v;
        }
    }

    for i in (0..n).rev() {
        for k in (i+1)..n {
            let v = lu[[i, k]] * x[k];
            x[i] -= v;
        }
        x[i] /= lu[[i, i]];
    }

}




