use std::marker::PhantomData;

use crate::{linalg::{Matrix, matrix::{DynamicMatrix, MatrixMut}, vector::{DynamicVector, Vector, VectorMut, VectorView, VectorViewMut}}, num_traits::FloatNumber, optimize::root::RootProblem};



const MAX_SIZE_STATIC_GRADIENT: usize = 256;
const MAX_SIZE_STATIC_JACOBIAN: usize = 16;



pub trait MinimizationProblem<T: FloatNumber> {
    type Error;

    fn size(&self) -> usize;

    // the cost function that must be minimized
    fn cost(&self, solution: &impl Vector<T>) -> Result<T, Self::Error>;

    // the gradient of the cost function, by default a finite difference approach
    fn gradient(&self, gradient: &mut impl VectorMut<T>, solution: &impl Vector<T>) -> Result<(), Self::Error> {

        let y = solution;

        // by default, use a difference quotient approach

        let n = self.size();

        // allocate temporary vectors or use static sized arrays
        let mut yd_alloc = [T::from(0); MAX_SIZE_STATIC_GRADIENT];

        let yd_a = &mut yd_alloc[0..y.len()];

        #[allow(unused_assignments)]
        let mut yd_ad = None;

        let mut yd = if n <= MAX_SIZE_STATIC_GRADIENT {
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
    pub fn problem(&self) -> &P {
        &self.problem
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
    pub step_decrease: T,
}


#[derive(Debug)]
pub struct MinimizationSolverResult<T> {
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
            step_decrease: T::fraction(99, 100),
        }
    }

    pub fn solution<'a>(&'a self) -> VectorView<'a, T> {
        self.solution.view()
    }

    pub fn set_initial_guess(&mut self, guess: &impl Vector<T>) {
        for i in 0..self.solution.len() {
            self.solution[i] = guess[i];
        }
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
        self.step_size *= (self.step_decrease * self.last_residual / residual.float_max(T::float_from_f64(1e-6))).float_min(T::fraction(15, 10)).float_max(T::fraction(5, 10));

        let extra_fact = if self.steps < 10 {
            T::fraction(self.steps as i32, 10).float_powi(2)
        } else {
            T::ONE
        };

        // move
        for i in 0..self.gradient.len() {
            self.solution[i] -= self.gradient[i] * self.step_size * extra_fact;
        }

        // done!
        self.last_residual = residual;

        Ok(())
    }


    pub fn solve(&mut self) -> Result<MinimizationSolverResult<T>, P::Error> {

        loop {
            self.step()?;

            //println!("{} {:?} {:?} {:?}", self.steps, self.step_size, self.last_residual, self.solution);

            self.steps += 1;

            if self.last_residual <= self.tolerance {
                break;
            }
            if self.last_residual != self.last_residual {
                break;
            }
            if self.steps >= self.max_steps {
                break;
            }
        }

        Ok(MinimizationSolverResult { iterations: self.steps, residual: self.last_residual, cost: self.problem.cost(&self.solution())? })
    }

}




pub struct BFGSSolver<T, P> where T: FloatNumber, P: MinimizationProblem<T> {
    problem: P,
    solution: DynamicVector<T>,
    gradient: DynamicVector<T>,
    last_gradient: DynamicVector<T>,
    direction: DynamicVector<T>,
    inverse_hessian: DynamicMatrix<T>,
    tmp_new_hessian: DynamicMatrix<T>,
    tmp_new_solution: DynamicVector<T>,
    hk_y: DynamicVector<T>,
    yt_hk: DynamicVector<T>,
    pub step_size: T,
    alpha_last: T,
    last_residual: T,
    steps: usize,
    pub max_steps: usize,
    pub tolerance: T,
    pub step_decrease: T,
    flips: usize,
}






impl<T, P> BFGSSolver<T, P> where T: FloatNumber, P: MinimizationProblem<T> {

    pub fn new(problem: P) -> Self {
        let size = problem.size();
        Self {
            problem,
            solution: DynamicVector::new(T::ZERO, size),
            gradient: DynamicVector::new(T::ZERO, size),
            last_gradient: DynamicVector::new(T::ZERO, size),
            direction: DynamicVector::new(T::ZERO, size),
            inverse_hessian: DynamicMatrix::eye([size, size]),
            tmp_new_hessian: DynamicMatrix::new(T::ZERO, [size, size]),
            tmp_new_solution: DynamicVector::new(T::ZERO, size),
            hk_y: DynamicVector::new(T::ZERO, size),
            yt_hk: DynamicVector::new(T::ZERO, size),
            step_size: T::fraction(1, 1000),
            alpha_last: T::fraction(1, 100),
            last_residual: T::ONE,
            steps: 0,
            max_steps: 10000,
            tolerance: T::EPSILON.float_sqrt() * T::from(1000),
            step_decrease: T::fraction(99, 100),
            flips: 0,
        }
    }

    pub fn set_initial_guess(&mut self, guess: &impl Vector<T>) {
        for i in 0..self.solution.len() {
            self.solution[i] = guess[i];
        }
    }

    pub fn solution<'a>(&'a self) -> VectorView<'a, T> {
        self.solution.view()
    }

    pub fn line_search(&mut self) -> Result<T, P::Error> {

        let c0 = self.problem.cost(&self.solution)?;

        // try twice the last alpha
        let mut alpha_guess = self.alpha_last * T::from(10);

        if self.flips > 5 {
            alpha_guess /= T::from(9);
        }

        let mut c1last = c0;
        let mut iter = 0;
        loop {
            for i in 0..self.solution.len() {
                self.tmp_new_solution[i] = self.solution[i] - self.direction[i] * alpha_guess;
            }

            let c1 = self.problem.cost(&self.tmp_new_solution)?;

            if (iter == 0) && (c1 < c1last) {
                self.alpha_last = alpha_guess;
                return Ok(alpha_guess);
            } else if (iter > 0) && (c1 > c1last) {
                self.alpha_last = alpha_guess * T::from(10);
                return Ok(self.alpha_last);
            } else {
                // reduce alpha guess
                alpha_guess /= T::from(10);
            }

            c1last = c1;

            iter += 1;
            if iter > 30 {
                return Ok(alpha_guess);
            }
        }
    }

    fn reset_hessian(&mut self) {
        for i in 0..self.inverse_hessian.shape()[0] {
            for j in 0..self.inverse_hessian.shape()[1] {
                self.inverse_hessian[[i, j]] = if i == j {T::ONE} else {T::ZERO};
            }
        }
    }


    fn step(&mut self) -> Result<(), P::Error> {
        
        // compute the direction
        self.inverse_hessian.imul(&mut self.direction, &self.gradient);

        if self.direction.dot(&self.gradient) < T::EPSILON {
            self.flips += 1;
            // flip the direction
            for i in 0..self.solution.len() {
                self.direction[i] = - self.direction[i];
            }
            // also reset the hessian
            self.reset_hessian();
        }

        // compute the current step size
        self.step_size = self.line_search()?;

        // move and set direction = - direction * step_size
        for i in 0..self.gradient.len() {
            self.direction[i] = - self.direction[i] * self.step_size;
            self.solution[i] += self.direction[i];
        }


        // compute the new gradient
        for i in 0..self.gradient.len() {
            self.last_gradient[i] = self.gradient[i];
        }
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

        // store the gradient change in the last gradient array
        for i in 0..self.gradient.len() {
            self.last_gradient[i] = self.gradient[i] - self.last_gradient[i];
        }

        // update the hessian matrix
        // costs one matrix multiplication
        self.inverse_hessian.imul(&mut self.hk_y, &self.last_gradient);
        self.inverse_hessian.imul_left(&mut self.yt_hk, &self.last_gradient);
        let s_dot_y = self.direction.dot(&self.last_gradient);
        let y_dot_hky = self.last_gradient.dot(&self.hk_y);

        for i in 0..self.gradient.len() {
            for j in i..self.gradient.len() {
                let sst_ij = self.direction[i] * self.direction[j];

                let hk_y_st_ij = self.hk_y[i] * self.direction[j];
                let s_yt_hk_ij = self.direction[i] * self.yt_hk[j];

                let hij = self.inverse_hessian[[i, j]];

                self.tmp_new_hessian[[i, j]] = hij + (s_dot_y + y_dot_hky) * sst_ij / (s_dot_y * s_dot_y) - (hk_y_st_ij + s_yt_hk_ij) / s_dot_y;
                // exploit the fact that the hessian matrix is symmetric
                self.tmp_new_hessian[[j, i]] = self.tmp_new_hessian[[i, j]];
            }
        }


        for i in 0..self.gradient.len() {
            for j in 0..self.gradient.len() {
                self.inverse_hessian[[i, j]] = self.tmp_new_hessian[[i, j]];
            }
        }

        // done!
        self.last_residual = residual;

        Ok(())
    }


    pub fn solve(&mut self) -> Result<MinimizationSolverResult<T>, P::Error> {

        //start the method by computing the gradient
        self.problem.gradient(&mut self.gradient, &self.solution)?;

        loop {
            self.step()?;

            self.steps += 1;

            if self.last_residual <= self.tolerance {
                break;
            }
            if self.last_residual != self.last_residual {
                break;
            }
            if self.steps >= self.max_steps {
                break;
            }
        }

        Ok(MinimizationSolverResult { iterations: self.steps, residual: self.last_residual, cost: self.problem.cost(&self.solution())? })
    }

}


