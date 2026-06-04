//!
//! 
//! ```rust
//! 
//! use calamus::functions::lagrange::*;
//! 
//! let lagrange = LagrangePolynomialBasis::new([0.0, 1.0, 2.0]);
//! 
//! let basis = lagrange.basis(0);
//! 
//! assert!((lagrange_polynomial(1.5, 0, &[0.0, 1.0, 2.0]) + 1.0 / 8.0).abs() < 1e-14);
//! assert!((basis(1.5) + 1.0 / 8.0).abs() < 1e-14);
//! 
//! ```
//! 
//! 
use crate::num_traits::FloatNumber;



pub fn lagrange_polynomial<T>(x: T, i: usize, xp: &[T]) -> T where T: FloatNumber {

    let mut prod = T::ONE;

    for j in 0..xp.len() {
        if i == j {
            continue;
        }
        prod *= (x - xp[j]) / (xp[i] - xp[j]);
    }

    prod
}



// Unidimensional lagrange polynomial basis
pub struct LagrangePolynomialBasis<T, const N: usize> {
    points: [T; N],
    weights: [T; N],
}


impl<T, const N: usize> LagrangePolynomialBasis<T, N> where T: FloatNumber {

    pub fn new(points: [T; N]) -> Self {

        let mut weights = [T::ONE; N];

        for i in 0..N {
            for j in 0..N {
                if i == j {continue}
                weights[i] *= T::ONE / (points[i] - points[j]);
            }
        }

        Self { points, weights }
    }

    pub fn basis(&self, i: usize) -> impl Fn(T) -> T {
        move |x| {
            // evaluate a single basis function at point x
            if (x - self.points[i]).float_abs() < T::EPSILON {
                return T::ONE;
            }
            let mut sum = T::ZERO;
            for j in 0..N {
                let dx = x - self.points[j];
                if dx.float_abs() < T::EPSILON {
                    return T::ZERO;
                }
                sum += self.weights[j] / dx;
            }
            (self.weights[i] / (x - self.points[i])) / sum
        }
    }

    pub fn basis_derivative(&self, i: usize) -> impl Fn(T) -> T {
        move |x| {
            // evaluate a single basis function's derivative at point x
            let mut prod = T::ONE;
            for j in 0..N {
                if i == j {continue}
                prod *= (x - self.points[j]) / (self.points[i] - self.points[j]);
            }
            let mut sum = T::ZERO;
            for j in 0..N {
                if i == j {continue}
                sum += prod / (self.points[i] - self.points[j]);
            }
            sum
        }
    }

    pub fn interpolant(&self, yp: &[T]) -> impl Fn(T) -> T {
        move |x| {
            let mut num = T::ZERO;
            let mut denom = T::ZERO;
            for i in 0..N {
                let xi = self.points[i];
                if (x - xi).float_abs() < T::EPSILON {
                    return yp[i];
                }
                let t = self.weights[i] / (x - xi);
                num += t * yp[i];
                denom += t;
            }
            num / denom
        }
    }

}



