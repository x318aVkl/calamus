use std::marker::PhantomData;

use crate::{linalg::{matrix::MatrixMut, vector::{DynamicVector, Vector, VectorMut, VectorViewMut}}, num_traits::FloatNumber, optimize::root::RootProblem};




const MAX_SIZE_STATIC_JACOBIAN: usize = 16;



pub trait MinimizeProblem<T: FloatNumber> {
    type Error;

    fn size(&self) -> usize;

    // the cost function that must be minimized
    fn cost(&self, solution: &impl Vector<T>) -> Result<T, Self::Error>;

    // the gradient of the cost function, by default a finite difference approach
    fn gradient(&self, gradient: &mut impl VectorMut<T>, solution: &impl Vector<T>) -> Result<(), Self::Error> {

        let y = solution;

        if y.len() > MAX_SIZE_STATIC_JACOBIAN {
            panic!("Error, problem size must be smaller or equal to {}", MAX_SIZE_STATIC_JACOBIAN);
        }

        // by default, use a difference quotient approach

        let n = self.size();

        // allocate temporary vectors or use static sized arrays
        let mut yd_alloc = [T::from(0); MAX_SIZE_STATIC_JACOBIAN];

        let yd_a = &mut yd_alloc[0..y.len()];

        #[allow(unused_assignments)]
        let mut yd_ad = None;

        let mut yd = if n <= MAX_SIZE_STATIC_JACOBIAN {
            let yd = VectorViewMut::from(yd_a);
            yd
        } else {
            // use dynamic allocation
            yd_ad = Some(DynamicVector::new(T::from(0), n));
            yd_ad.as_mut().unwrap().view_mut()
        };

        for i in 0..n {
            yd[i] = y[i];
        }

        // base f
        let c0 = self.cost(y)?;

        let scale = T::EPSILON.float_sqrt() * T::from(100);
        

        for i in 0..n {
            let y0 = y[i];
            let dy = scale.float_max(y0.float_abs() * scale);
            yd[i] = y0 + dy;

            let c1 = self.cost(&yd)?;

            gradient[i] = (c1 - c0) / dy;

            yd[i] = y0;
        }

        Ok(())
    }


    // hessian matrix of the problem, by default uses a finite difference approach
    fn hessian(&self, hessian: &mut impl MatrixMut<T>, solution: &impl Vector<T>) -> Result<(), Self::Error> {

        let y = solution;

        if y.len() > MAX_SIZE_STATIC_JACOBIAN {
            panic!("Error, problem size must be smaller or equal to {}", MAX_SIZE_STATIC_JACOBIAN);
        }

        // by default, use a difference quotient approach

        let n = self.size();

        // allocate temporary vectors or use static sized arrays
        let mut yd_alloc = [T::from(0); MAX_SIZE_STATIC_JACOBIAN];

        let yd_a = &mut yd_alloc[0..y.len()];

        #[allow(unused_assignments)]
        let mut yd_ad = None;

        let mut yd = if n <= MAX_SIZE_STATIC_JACOBIAN {
            let yd = VectorViewMut::from(yd_a);
            yd
        } else {
            // use dynamic allocation
            yd_ad = Some(DynamicVector::new(T::from(0), n));
            yd_ad.as_mut().unwrap().view_mut()
        };

        for i in 0..n {
            yd[i] = y[i];
        }

        // base f
        let c00 = self.cost(y)?;

        let scale = T::EPSILON.float_sqrt() * T::from(100);
        

        for i in 0..n {
            for j in 0..n {
                let y0i = y[i];
                let dyi = scale.float_max(y0i.float_abs() * scale);
                yd[i] = y0i + dyi;

                let cpi = self.cost(&yd)?;

                yd[i] = y0i;

                let y0j = y[j];
                let dyj = scale.float_max(y0j.float_abs() * scale);
                yd[j] = y0j + dyj;

                let cpj = self.cost(&yd)?;

                yd[j] = y0j;

                yd[i] += dyi;
                yd[j] += dyj;

                let cpij = self.cost(&yd)?;

                yd[i] = y0i;
                yd[j] = y0j;

                hessian[[i, j]] = (cpij - cpi - cpj + c00) / (dyi * dyj);

            }
        }

        Ok(())

    }

}



pub struct NewtonMinimizationProblem<T: FloatNumber, P> where P: MinimizeProblem<T> {
    problem: P,
    td: PhantomData<T>,
}

impl<T: FloatNumber, P: MinimizeProblem<T>> NewtonMinimizationProblem<T, P> {
    pub fn new(problem: P) -> Self {
        Self {
            problem,
            td: PhantomData,
        }
    }
}



impl<T, F> RootProblem<T> for NewtonMinimizationProblem<T, F> where F: MinimizeProblem<T>, T: FloatNumber, <F as MinimizeProblem<T>>::Error: std::fmt::Debug {
    type Error = F::Error;
    fn size(&self) -> usize {
        self.problem.size()
    }
    fn residual(&self, residual: &mut impl VectorMut<T>, solution: &impl Vector<T>) -> Result<(), Self::Error> {
        self.problem.gradient(residual, solution)
    }
    fn jacobian(&self, jacobian: &mut impl MatrixMut<T>, solution: &impl Vector<T>) -> Result<(), Self::Error> {
        self.problem.hessian(jacobian, solution)
    }
}

