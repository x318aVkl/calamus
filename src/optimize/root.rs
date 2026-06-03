use crate::{linalg::{self, MatrixFactorization, lu::DynamicLu, matrix::{DynamicMatrix, MatrixMut}, vector::{DynamicVector, Vector, VectorMut, VectorView, VectorViewMut}}, num_traits::FloatNumber};




pub trait RootProblem<T> where T: FloatNumber {
    type Error: std::fmt::Debug;

    // get the number of variables of the problem
    fn size(&self) -> usize;

    fn residual(&self, residual: &mut impl VectorMut<T>, solution: &impl Vector<T>) -> Result<(), Self::Error>;

    // by default, implement a finite difference jacobian
    fn jacobian(&self, jacobian: &mut impl MatrixMut<T>, solution: &impl Vector<T>) -> Result<(), Self::Error> {

        let y = solution;
        let jac = jacobian;

        if y.len() > MAX_SIZE_STATIC_JACOBIAN {
            panic!("Error, problem size must be smaller or equal to {}", MAX_SIZE_STATIC_JACOBIAN);
        }

        // by default, use a difference quotient approach

        let n = self.size();

        // allocate temporary vectors or use static sized arrays
        let mut f_alloc = [T::from(0); MAX_SIZE_STATIC_JACOBIAN];
        let mut fb_alloc = [T::from(0); MAX_SIZE_STATIC_JACOBIAN];
        let mut yd_alloc = [T::from(0); MAX_SIZE_STATIC_JACOBIAN];

        let f_a = &mut f_alloc[0..y.len()];
        let fb_a = &mut fb_alloc[0..y.len()];
        let yd_a = &mut yd_alloc[0..y.len()];

        #[allow(unused_assignments)]
        let mut f_ad = None;
        #[allow(unused_assignments)]
        let mut f_bd = None;
        #[allow(unused_assignments)]
        let mut yd_ad = None;

        let (mut f, mut fb, mut yd) = if n <= MAX_SIZE_STATIC_JACOBIAN {
            let f = VectorViewMut::from(f_a);
            let fb = VectorViewMut::from(fb_a);
            let yd = VectorViewMut::from(yd_a);
            (f, fb, yd)
        } else {
            // use dynamic allocation
            f_ad = Some(DynamicVector::new(T::from(0), n));
            f_bd = Some(DynamicVector::new(T::from(0), n));
            yd_ad = Some(DynamicVector::new(T::from(0), n));

            (
                f_ad.as_mut().unwrap().view_mut(), 
                f_bd.as_mut().unwrap().view_mut(),
                yd_ad.as_mut().unwrap().view_mut(),
            )
        };

        for i in 0..n {
            yd[i] = y[i];
        }

        // base f
        self.residual(&mut f, y)?;

        let scale = T::EPSILON.float_sqrt() * T::from(100);
        

        for i in 0..n {
            let y0 = y[i];
            let dy = scale.float_max(y0.float_abs() * scale);
            yd[i] = y0 + dy;

            self.residual(&mut fb, &yd)?;

            for j in 0..n {
                jac[[j, i]] = (fb[j] - f[j]) / dy;
            }

            yd[i] = y0;
        }

        Ok(())
    }
}



// find the root of functions using a newton raphson algorithm
pub struct NewtonSolver<T, P, F = DynamicLu<T>> where F: MatrixFactorization<T>, P: RootProblem<T>, T: FloatNumber {
    problem: P,
    jacobian: DynamicMatrix<T>,
    residual: DynamicVector<T>,
    solution: DynamicVector<T>,
    delta: DynamicVector<T>,
    factorization: F,
    niters: usize,
    jacobian_frequency: usize,
    stab: T,
    residual_norm: T,
    last_residual: T,
    tolerance: T,
    max_iterations: usize,
}


#[derive(Debug)]
pub enum NewtonSolverError<E: std::fmt::Debug> {
    ProblemError(E),
    LinalgError(linalg::Error),
    MaxIterations,
}

impl<E: std::fmt::Debug> NewtonSolverError<E> {
    fn handle_problem(r: Result<(), E>) -> Result<(), Self> {
        match r {
            Ok(()) => Ok(()),
            Err(e) => Err(Self::ProblemError(e))
        }
    }

    fn handle_linalg(r: Result<(), linalg::Error>) -> Result<(), Self> {
        match r {
            Ok(()) => Ok(()),
            Err(e) => Err(Self::LinalgError(e))
        }
    }
}



pub struct NewtonSolverResult<T> {
    iterations: usize,
    final_residual: T,
}

impl<T> std::fmt::Debug for NewtonSolverResult<T> where T: std::fmt::Debug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NewtonSolverResult(iterations={}, residual={:?})", self.iterations, self.final_residual)
    }
}


impl<T, P, F> NewtonSolver<T, P, F> where F: MatrixFactorization<T>, T: FloatNumber, P: RootProblem<T> {


    pub fn problem(&self) -> &P {
        &self.problem
    }

    pub fn problem_mut(&mut self) -> &mut P {
        &mut self.problem
    }


    pub fn len(&self) -> usize {
        self.solution.len()
    }


    pub fn new(problem: P) -> Self {
        let size = problem.size();

        Self {
            problem,
            jacobian: DynamicMatrix::new(T::ZERO, [size, size]),
            residual: DynamicVector::new(T::ZERO, size),
            solution: DynamicVector::new(T::ZERO, size),
            delta: DynamicVector::new(T::ZERO, size),
            factorization: F::allocate(size),
            niters: 0,
            jacobian_frequency: 3,
            stab: T::ONE,
            residual_norm: T::ONE,
            last_residual: T::ONE,
            tolerance: T::EPSILON.float_sqrt() * T::from(100),
            max_iterations: 500,
        }
    }


    pub fn set_initial_guess(&mut self, guess: &impl Vector<T>) {
        for i in 0..self.len() {
            self.solution[i] = guess[i];
        }
        self.niters = 0;
    }

    pub fn solution<'a>(&'a self) -> VectorView<'a, T> {
        self.solution.view()
    }


    pub fn solve(&mut self) -> Result<NewtonSolverResult<T>, NewtonSolverError<P::Error>> {
        
        self.niters = 0;
        self.residual_norm = T::ONE;
        self.last_residual = T::ONE;

        loop {
            
            self.attempt_step()?;

            self.niters += 1;
            if self.residual_norm < self.tolerance {
                break;
            }

            if self.niters >= self.max_iterations {
                return Err(NewtonSolverError::MaxIterations);
            }

            // adjust the stabilization factor
            if self.residual_norm > (self.last_residual * T::from(10)) {
                let rlast = self.last_residual;
                let rnorm = self.residual_norm;
                self.stab *= 
                    (T::ONE * (rlast / rnorm).float_powi(1)).float_min(T::fraction(12, 10))
                ;
                self.stab = self.stab.float_min(T::ONE).float_max(T::fraction(1, 10));
            }

            self.last_residual = self.residual_norm;
        }

        Ok(NewtonSolverResult { 
            iterations: self.niters, 
            final_residual: self.residual_norm 
        })
    }


    fn attempt_step(&mut self) -> Result<(), NewtonSolverError<P::Error>> {

        // check if we need to solve the jacobian
        if (self.niters == 0) || (self.niters % self.jacobian_frequency == 0) {
            // recompute the jacobian matrix
            NewtonSolverError::handle_problem(
                self.problem.jacobian(&mut self.jacobian, &self.solution)
            )?;

            // set the factorization solver matrix
            NewtonSolverError::handle_linalg(
                self.factorization.set_matrix(&self.jacobian)
            )?;

            // factorize the jacobian
            NewtonSolverError::handle_linalg(
                self.factorization.factorize()
            )?;
        }

        // assemble the residual
        NewtonSolverError::handle_problem(
            self.problem.residual(&mut self.residual, &self.solution)
        )?;

        // solve the problem
        self.factorization.solve(&mut self.delta, &self.residual);

        let mut rnorm = T::ZERO;
        for i in 0..self.len() {
            rnorm += self.delta[i].float_powi(2);
        }
        rnorm = rnorm.float_sqrt();
        self.residual_norm = rnorm;

        for i in 0..self.len() {
            self.solution[i] -= self.delta[i] * self.stab;
        }

        Ok(())
    }





}






const MAX_SIZE_STATIC_JACOBIAN: usize = 256;
