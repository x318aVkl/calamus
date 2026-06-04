use std::marker::PhantomData;

use crate::{linalg::{matrix::MatrixMut, vector::{DynamicVector, Vector, VectorMut, VectorView, VectorViewMut}}, num_traits::FloatNumber, optimize::root::RootProblem};




const MAX_SIZE_STATIC_JACOBIAN: usize = 16;



pub trait MinimizationProblem<T: FloatNumber> {
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
            for j in i..n {
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

                hessian[[j, i]] = hessian[[i, j]];

            }
        }

        Ok(())

    }

}



pub struct NewtonMinimizationProblem<T: FloatNumber, P> where P: MinimizationProblem<T> {
    problem: P,
    td: PhantomData<T>,
}

impl<T: FloatNumber, P: MinimizationProblem<T>> NewtonMinimizationProblem<T, P> {
    pub fn new(problem: P) -> Self {
        Self {
            problem,
            td: PhantomData,
        }
    }
}



impl<T, F> RootProblem<T> for NewtonMinimizationProblem<T, F> where F: MinimizationProblem<T>, T: FloatNumber, <F as MinimizationProblem<T>>::Error: std::fmt::Debug {
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



// Solver which uses gradient descent instead of newton method
// usefull to find a rough estimate before using a newton type method
pub struct GradientDescentSolver<T, P> {
    problem: P,
    solution: DynamicVector<T>,
    gradient: DynamicVector<T>,
    pub step_size: T,
    last_residual: T,
    steps: usize,
    pub max_steps: usize,
    pub tolerance: T,
}


#[derive(Debug)]
pub struct GradientDescentSolverResult<T> {
    #[allow(dead_code)]
    iterations: usize,
    #[allow(dead_code)]
    residual: T,
    #[allow(dead_code)]
    cost: T,
}



impl<T, P> GradientDescentSolver<T, P> where T: FloatNumber, P: MinimizationProblem<T> {

    pub fn new(problem: P) -> Self {
        let size = problem.size();
        Self {
            problem,
            solution: DynamicVector::new(T::ZERO, size),
            gradient: DynamicVector::new(T::ZERO, size),
            step_size: T::fraction(1, 1000),
            last_residual: T::ONE,
            steps: 0,
            max_steps: 10000,
            tolerance: T::EPSILON.float_sqrt() * T::from(1000),
        }
    }

    pub fn solution<'a>(&'a self) -> VectorView<'a, T> {
        self.solution.view()
    }


    fn step(&mut self) -> Result<(), P::Error> {

        // compute the gradient
        self.problem.gradient(&mut self.gradient, &self.solution)?;

        // compute the current residual as the gradient norm
        let mut residual: T = T::ZERO;
        for i in 0..self.gradient.len() {
            residual += self.gradient[i].float_powi(2);
        }
        residual = residual.float_sqrt();

        if self.steps == 0 {
            self.last_residual = residual;
        }

        // compute the current step size
        self.step_size *= (residual / self.last_residual).float_min(T::fraction(12, 10)).float_max(T::fraction(5, 10));

        // move
        for i in 0..self.gradient.len() {
            self.solution[i] -= self.gradient[i] * self.step_size;
        }

        // done!
        self.last_residual = residual;

        Ok(())
    }


    pub fn solve(&mut self) -> Result<GradientDescentSolverResult<T>, P::Error> {

        loop {
            self.step()?;

            //println!("{} {:?} {:?} {:?}", self.steps, self.step_size, self.last_residual, self.solution);

            self.steps += 1;

            if self.last_residual <= self.tolerance {
                break;
            }
            if self.steps >= self.max_steps {
                break;
            }
        }

        Ok(GradientDescentSolverResult { iterations: self.steps, residual: self.last_residual, cost: self.problem.cost(&self.solution())? })
    }

}







