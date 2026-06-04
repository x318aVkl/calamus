

use crate::{linalg::{self, factorize::lu::DynamicLu, vector::{DynamicVector, Vector, VectorMut, VectorView, VectorViewMut}}, num_traits::FloatNumber, ode::{Ode, solvers::Solver}, optimize::root::{NewtonSolver, NewtonSolverError, RootProblem}};







pub struct Bdf2SubProblem<T, F> where F: Ode<T> {
    ode: F,
    last_solution_0: DynamicVector<T>,
    last_solution_1: DynamicVector<T>,
    time: T,
    dt: T,
    c0: T,
    c1: T,
    c2: T,
}



impl<T, F> RootProblem<T> for Bdf2SubProblem<T, F> 
where T: FloatNumber,
F: Ode<T>,
<F as Ode<T>>::Error: std::fmt::Debug
{
    type Error = F::Error;

    fn size(&self) -> usize {
        self.ode.problem_size()
    }

    fn residual(&self, residual: &mut impl VectorMut<T>, solution: &impl Vector<T>) -> Result<(), Self::Error> {
        self.ode.eval_f(residual, solution, self.time)?;

        for i in 0..self.ode.problem_size() {
            let fi = residual[i];

            residual[i] = self.c0 * solution[i] + self.c1 * self.last_solution_0[i] + self.c2 * self.last_solution_1[i] - self.dt * fi;
        }

        Ok(())
    }

    fn jacobian(&self, jacobian: &mut impl linalg::matrix::MatrixMut<T>, solution: &impl Vector<T>) -> Result<(), Self::Error> {

        self.ode.eval_jacobian(jacobian, solution, self.time)?;
    
        for i in 0..jacobian.shape()[0] {
            for j in 0..jacobian.shape()[1] {
                let aij = jacobian[[i, j]];
                jacobian[[i, j]] = - self.dt * aij;
            }
            jacobian[[i, i]] += self.c0;
        }
        
        Ok(())
    }
}






// Implicit Bdf-2 solver with adaptive step size
// Good for stiff problems
pub struct Bdf2<T: FloatNumber, F: Ode<T>> where <F as Ode<T>>::Error: std::fmt::Debug {

    // root finder to solve the sub time step problem
    root_finder: NewtonSolver<T, Bdf2SubProblem<T, F>, DynamicLu<T>>,
    
    // used for error estimation
    last_solution_2: DynamicVector<T>,

    t: T,
    dt: T,

    initial_dt: T,

    last_dt: T,
    last_dt_2: T,

    n_steps: usize,

    min_dt: T,
    te_tolerance: T,
}



#[derive(Debug)]
pub enum Bdf2Error<FE> {
    EulerMaxIterations,
    EulerAlphaDivergence,
    Bdf2MaxIterations,
    Linalg(linalg::Error),
    Ode(FE)
}

impl<FE> std::fmt::Display for Bdf2Error<FE> where FE: std::fmt::Debug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EulerMaxIterations => write!(f, "EulerMaxIterations"),
            Self::EulerAlphaDivergence => write!(f, "EulerAlphaDivergence"),
            Self::Bdf2MaxIterations => write!(f, "Bdf2MaxIterations"),
            Self::Linalg(l) => write!(f, "{:?}", l),
            Self::Ode(ode) => write!(f, "{:?}", ode),
        }
    }
}

impl<FE: std::error::Error> std::error::Error for Bdf2Error<FE> {
}


impl<T, F: Ode<T>> Bdf2<T, F> where <F as Ode<T>>::Error: std::fmt::Debug, T: FloatNumber {

    pub fn new(ode: F) -> Self {
        let size = ode.problem_size();
        let sqreps = T::EPSILON.float_sqrt();
        let dt0 = T::fraction(1, 1_000_000);
        Self {
            root_finder: NewtonSolver::new(Bdf2SubProblem{ 
                ode, 
                last_solution_0: DynamicVector::new(T::ZERO, size),
                last_solution_1: DynamicVector::new(T::ZERO, size),
                time: T::ZERO,
                dt: dt0,
                c0: T::ONE,
                c1: T::ONE,
                c2: T::ONE,
            }),
            
            last_solution_2: DynamicVector::new(T::ZERO, size),
           
            t: T::ZERO,
            dt: dt0,
            initial_dt: dt0,
            last_dt: dt0,
            last_dt_2: dt0,
            n_steps: 0,
            min_dt: T::EPSILON * T::from(1000),
            te_tolerance: sqreps * T::from(100),
        }
    }


    pub fn with_t0(mut self, t0: T) -> Self {
        self.t = t0;
        self
    }

    pub fn with_dt0(mut self, dt0: T) -> Self {
        self.initial_dt = dt0;

        self.dt = dt0;
        self.last_dt = dt0;
        self.last_dt_2 = dt0;
        self
    }

    pub fn with_initial_guess(mut self, guess: &impl Vector<T>) -> Self {

        self.root_finder.set_initial_guess(guess);

        for i in 0..self.root_finder.solution().len() {
            let yi = guess[i];
            self.root_finder.problem_mut().last_solution_0[i] = yi;
            self.root_finder.problem_mut().last_solution_1[i] = yi;
            self.last_solution_2[i] = yi;
        }

        self
    }

    pub fn with_min_dt(mut self, min_dt: T) -> Self {
        self.min_dt = min_dt;
        self
    }

    pub fn with_tolerance(mut self, tol: T) -> Self {
        self.te_tolerance = tol;
        self
    }


    fn bdf_d0(x: T) -> T {
        (T::ONE + T::TWO * x) / (T::ONE + x)
    }

    fn bdf_d1(x: T) -> T {
        - x / (T::ONE + x)
    }


    pub fn solution<'a>(&'a self) -> VectorView<'a, T> {
        self.root_finder.solution()
    }

    fn last_solution_0<'a>(&'a self) -> VectorView<'a, T> {
        self.root_finder.problem().last_solution_0.view()
    }
    fn last_solution_1<'a>(&'a self) -> VectorView<'a, T> {
        self.root_finder.problem().last_solution_1.view()
    }

    fn last_solution_0_mut<'a>(&'a mut self) -> VectorViewMut<'a, T> {
        self.root_finder.problem_mut().last_solution_0.view_mut()
    }
    fn last_solution_1_mut<'a>(&'a mut self) -> VectorViewMut<'a, T> {
        self.root_finder.problem_mut().last_solution_1.view_mut()
    }
    fn last_solution_2_mut<'a>(&'a mut self) -> VectorViewMut<'a, T> {
        self.last_solution_2.view_mut()
    }



    fn step_backward(&mut self) -> Result<(), Bdf2Error<F::Error>> {
        
        let h = self.dt;

        let rk = self.dt / self.last_dt;

        let bd0 = Self::bdf_d0(rk);
        let bd1 = Self::bdf_d1(rk);

        let c0 =  bd0         ;
        let c1 =  - bd0 + bd1 ;
        let c2 =  - bd1    ;


        self.root_finder.problem_mut().c0 = c0;
        self.root_finder.problem_mut().c1 = c1;
        self.root_finder.problem_mut().c2 = c2;

        self.root_finder.problem_mut().time = self.t;
        self.root_finder.problem_mut().dt = h;

        

        // solve the problem
        match self.root_finder.solve() {
            Ok(_) => Ok(()),
            Err(e) => match e {
                NewtonSolverError::LinalgError(e) => Err(Bdf2Error::Linalg(e)),
                NewtonSolverError::MaxIterations => Err(Bdf2Error::Bdf2MaxIterations),
                NewtonSolverError::ProblemError(e) => Err(Bdf2Error::Ode(e))
            }
        }
    }


    fn truncation_error(&self) -> T {

        let mut emax: T = T::ZERO;
        for i in 0..self.solution().len() {
            // approximate the truncation error as the difference between euler and Bdf2

            let dy0 = (self.solution()[i] - self.last_solution_0()[i]) / self.dt;

            let dy1 = (self.last_solution_0()[i] - self.last_solution_1()[i]) / self.last_dt;

            let dy2 = (self.last_solution_1()[i] - self.last_solution_2[i]) / self.last_dt;

            let dt0 = (self.dt + self.last_dt) * T::HALF;
            let dt1 = (self.last_dt + self.last_dt_2) * T::HALF;

            let dyy0 = (dy0 - dy1) / dt0;
            let dyy1 = (dy1 - dy2) / dt1;

            let dyyy = (dyy0 - dyy1) / ((dt0 + dt1) * T::HALF);

            let e = dyyy.float_abs() * self.dt.float_powi(3);

            emax = emax.float_max(e.float_abs());
        }

        emax
    }

}






impl<T, F: Ode<T>> Solver<T> for Bdf2<T, F> where <F as Ode<T>>::Error: std::fmt::Debug, T: FloatNumber + std::fmt::Debug {
    type Error = <F as Ode<T>>::Error;

    fn solution<'a>(&'a self) -> VectorView<'a, T> {
        self.root_finder.solution()
    }

    fn time(&self) -> T {
        self.t
    }

    fn step_size(&self) -> T {
        self.dt
    }

    fn set_step_size(&mut self, dt: T) {
        self.dt = dt;
    }

    fn steps(&self) -> usize {
        self.n_steps
    }


    fn reinit(&mut self, initial_solution: &impl Vector<T>, time: T) {

        self.t = time;

        self.dt = self.initial_dt;
        self.last_dt = self.dt;
        self.last_dt_2 = self.dt;


        self.root_finder.set_initial_guess(initial_solution);

        for i in 0..self.solution().len() {
            self.last_solution_0_mut()[i] = initial_solution[i];
            self.last_solution_1_mut()[i] = initial_solution[i];
            self.last_solution_2_mut()[i] = initial_solution[i];
        }

        // done
    }


    fn step(&mut self) -> Result<(), Self::Error> {
        match self.step_backward() {
            Ok(()) => {},
            Err(e) => {

                if self.dt < T::EPSILON * T::EPSILON {
                    panic!("solver failed, dt = {:?}, time = {:?}, steps = {:?}, error: {}", self.dt, self.time(), self.steps(), e)
                }

                self.dt /= T::TWO;

                if self.n_steps == 0 {
                    self.last_dt = self.dt;
                    self.last_dt_2 = self.dt;
                }
                self.step()?;

                return Ok(());
            }
        }


        let truncmax = self.truncation_error();

        let epsilon = self.te_tolerance;
        let hnew = T::fraction(9, 10) * self.dt * (epsilon / truncmax.float_max(T::EPSILON).float_min(T::from(100_000))).float_powf(T::fraction(1, 3)).float_min(T::fraction(12, 10));
        let dtlast = self.dt;
        self.dt = hnew;
        if self.n_steps == 0 {
            self.last_dt = hnew;
        }

        let mut ignore_terror = false;
        if self.dt < self.min_dt {
            self.dt = self.min_dt;
            ignore_terror = true;
        }

        if (truncmax > epsilon) && !ignore_terror {
            // reject step and do it again
            self.step()?;
        } else {
            // accept the step
            //println!("step is goo!\n");

            self.n_steps += 1;

            self.last_dt_2 = self.last_dt;
            self.last_dt = dtlast;
            self.t += dtlast;

            for i in 0..self.solution().len() {
                self.last_solution_2_mut()[i] = self.last_solution_1()[i];
                self.last_solution_1_mut()[i] = self.last_solution_0()[i];
                self.last_solution_0_mut()[i] = self.solution()[i];
            }

        }
        
        Ok(())
    }


    // fn step(&mut self) -> Result<(), Self::Error> {
    //     let t0 = self.t;

    //     // first do the crank nicolson step
    //     match self.attempt_step() {
    //         Ok(()) => {
    //             // we good
    //         },
    //         Err(e) => {
    //             if self.dt < 1e-24 {
    //                 panic!("Error, time step too small and solve attempt failed, {}", e);
    //             }
    //             for i in 0..self.solution.len() {
    //                 self.solution[i] = self.last_solution_0[i];
    //             }

    //             //println!("step failed {:?} {} {:?}", self.dt, e, self.solution);

    //             self.t = t0;
    //             self.dt /= 2.0;

    //             self.step()?;
    //         }
    //     }

    //     // update step size based on truncation error
    //     let truncmax = self.truncation_error().max(1e-16);

    //     let epsilon = 1e-2;
    //     let hnew = 0.9 * self.dt * (epsilon / truncmax).powf(0.333333333);
    //     let hnew = hnew.min(1.5 * self.dt);
    //     let dtlast = self.dt;
    //     self.dt = hnew;

    //     if hnew < 1e-24 {
    //         panic!("Error, time step too small, truncation error too large {:.3e} {:.3e} {:.3e}", hnew, truncmax, epsilon);
    //     }

    //     //println!("{} {:.3e} {:.3e}", self.time(), truncmax, hnew);

    //     if truncmax > epsilon {
    //         // try again
    //         for i in 0..self.solution.len() {
    //             self.solution[i] = self.last_solution_0[i];
    //         }
    //         self.t = t0;
    //         // do the step again
    //         self.step()?;
    //     } else {
    //         // good! 
    //         self.n_steps += 1;

    //         self.last_dt = dtlast;
    //         self.t += dtlast;

    //         for i in 0..self.solution.len() {
    //             self.last_solution_1[i] = self.last_solution_0[i];
    //             self.last_solution_0[i] = self.solution[i];
    //         }
    //     }

    //     //println!("{:.3e} {:.3e}, {:?}", self.time(), self.step_size(), self.solution());

    //     Ok(())
    // }
}



