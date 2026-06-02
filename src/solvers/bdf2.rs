use crate::{Ode, linalg::{self, plu_decompose, plu_solve}, ode::OdeJacobian, solvers::Solver};








// Implicit Bdf-2 solver with adaptive step size
// Good for stiff problems
pub struct Bdf2<F: Ode + OdeJacobian> {
    ode: F,
    solution: Vec<f64>,
    last_solution_0: Vec<f64>,
    last_solution_1: Vec<f64>,
    last_solution_2: Vec<f64>,
    jacobian: Vec<f64>,
    residual: Vec<f64>,
    p: Vec<usize>,
    dy: Vec<f64>,

    t: f64,
    dt: f64,

    initial_dt: f64,

    last_dt: f64,
    last_dt_2: f64,

    n_steps: usize,

    min_dt: f64,
    te_tolerance: f64,
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


impl<F: Ode + OdeJacobian> Bdf2<F> where <F as Ode>::Error: std::fmt::Debug {

    pub fn new(ode: F) -> Self {
        let size = ode.problem_size();
        Self {
            ode,
            solution: vec![0.0; size],
            last_solution_0: vec![0.0; size],
            last_solution_1: vec![0.0; size],
            last_solution_2: vec![0.0; size],
            jacobian: vec![0.0; size*size],
            residual: vec![0.0; size],
            p: vec![0; size + 1],
            dy: vec![0.0; size],
            t: 0.0,
            dt: 1e-6,
            initial_dt: 1e-6,
            last_dt: 1e-6,
            last_dt_2: 1e-6,
            n_steps: 0,
            min_dt: 1e-12,
            te_tolerance: 1e-4,
        }
    }


    pub fn with_t0(mut self, t0: f64) -> Self {
        self.t = t0;
        self
    }

    pub fn with_dt0(mut self, dt0: f64) -> Self {
        self.initial_dt = dt0;

        self.dt = dt0;
        self.last_dt = dt0;
        self.last_dt_2 = dt0;
        self
    }

    pub fn with_initial_guess(mut self, guess_function: impl Fn(usize) -> f64) -> Self {

        for i in 0..self.solution.len() {
            let yi = guess_function(i);
            self.solution[i] = yi;
            self.last_solution_0[i] = yi;
            self.last_solution_1[i] = yi;
            self.last_solution_2[i] = yi;
        }

        self
    }

    pub fn with_min_dt(mut self, min_dt: f64) -> Self {
        self.min_dt = min_dt;
        self
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.te_tolerance = tol;
        self
    }

    fn handle_error(error: Result<(), F::Error>) -> Result<(), Bdf2Error<F::Error>> {
        match error {
            Ok(()) => Ok(()),
            Err(e) => Err(Bdf2Error::Ode(e))
        }
    }


    fn bdf_d0(x: f64) -> f64 {
        (1.0 + 2.0 * x) / (1.0 + x)
    }

    fn bdf_d1(x: f64) -> f64 {
        - x / (1.0 + x)
    }



    fn step_backward(&mut self) -> Result<(), Bdf2Error<F::Error>> {

        let tol = 1e-7;
        
        let n = self.ode.problem_size();
        let h = self.dt;

        let mut iter = 0;

        let rk = self.dt / self.last_dt;

        let bd0 = Self::bdf_d0(rk);
        let bd1 = Self::bdf_d1(rk);

        let c0 =  bd0         ;
        let c1 =  - bd0 + bd1 ;
        let c2 =  - bd1    ;

        let mut stab: f64 = 1.0;
        let mut rlast: f64 = 1.0;

        let mut solnorm: f64 = 0.0;
        for i in 0..self.solution.len() {
            solnorm += self.solution[i].powi(2);
        }
        solnorm = solnorm.sqrt();


        loop {

            // assemble the  residual
            Self::handle_error(self.ode.eval_f(&mut self.residual, &self.solution, self.t + h))?;

            for i in 0..n {
                let fi = self.residual[i];
                self.residual[i] = c0*self.solution[i] + c1*self.last_solution_0[i] + c2*self.last_solution_1[i] - h * fi;
            }

            // assemble the jacobian
            Self::handle_error(self.ode.eval_jacobian(&mut self.jacobian, &self.solution, self.t + h))?;
            //println!("{:?} {:?}", self.jacobian, self.dt);
            for i in 0..n {
                // adjust the jacobian
                for j in 0..n {
                    self.jacobian[i*n + j] = - h * self.jacobian[i*n + j];
                }
                self.jacobian[i*n + i] = c0 + self.jacobian[i*n + i];
            }

            //println!("{:?} {:?}", self.jacobian, self.dt);


            // solve the problem using lu decomposition
            match plu_decompose(&mut self.jacobian, &mut self.p) {
                Ok(()) => {},
                Err(e) => {
                    return Err(Bdf2Error::Linalg(e))
                },
            };
            plu_solve(&mut self.dy, &self.residual, &self.jacobian, &self.p);


            let mut rnorm = 0.0;
            for i in 0..n {
                self.solution[i] -= self.dy[i] * stab;
                rnorm += self.dy[i].powi(2);
            }
            rnorm = rnorm.sqrt() / solnorm;

            if rnorm < tol {
                break;
            }
            if iter == 0 {
                rlast = rnorm;
            }

            
            stab *= if iter == 0 {1.0} else {
                (1.0 * (rlast / rnorm).powi(2)).min(1.2)
            };
            stab = stab.min(1.0).max(0.1);


            rlast = rnorm;

            iter += 1;

            //println!("{} {:?} {:?}", iter, rnorm, stab);

            if iter >= 1000 {
                //println!("{} {:?}", iter, rnorm);
                return Err(Bdf2Error::Bdf2MaxIterations);
            }
        }

        Ok(())
    }



    fn truncation_error(&self) -> f64 {

        let mut emax: f64 = 0.0;
        for i in 0..self.solution.len() {
            // approximate the truncation error as the difference between euler and Bdf2

            let dy0 = (self.solution[i] - self.last_solution_0[i]) / self.dt;

            let dy1 = (self.last_solution_0[i] - self.last_solution_1[i]) / self.last_dt;

            let dy2 = (self.last_solution_1[i] - self.last_solution_2[i]) / self.last_dt;

            let dt0 = (self.dt + self.last_dt) * 0.5;
            let dt1 = (self.last_dt + self.last_dt_2) * 0.5;

            let dyy0 = (dy0 - dy1) / dt0;
            let dyy1 = (dy1 - dy2) / dt1;

            let dyyy = (dyy0 - dyy1) / ((dt0 + dt1) * 0.5);

            let e = dyyy.abs() * self.dt.powi(3);

            emax = emax.max(e.abs());
        }

        emax
    }

}






impl<F: Ode + OdeJacobian> Solver for Bdf2<F> where <F as Ode>::Error: std::fmt::Debug {
    type Error = <F as Ode>::Error;

    fn solution(&self) -> &[f64] {
        &self.solution
    }

    fn time(&self) -> f64 {
        self.t
    }

    fn step_size(&self) -> f64 {
        self.dt
    }

    fn set_step_size(&mut self, dt: f64) {
        self.dt = dt;
    }

    fn steps(&self) -> usize {
        self.n_steps
    }


    fn reinit(&mut self, initial_solution: &[f64], time: f64) {

        self.t = time;

        self.dt = self.initial_dt;
        self.last_dt = self.dt;
        self.last_dt_2 = self.dt;


        for i in 0..self.solution.len() {
            self.solution[i] = initial_solution[i];
            self.last_solution_0[i] = initial_solution[i];
            self.last_solution_1[i] = initial_solution[i];
            self.last_solution_2[i] = initial_solution[i];
        }

        // done
    }


    fn step(&mut self) -> Result<(), Self::Error> {
        match self.step_backward() {
            Ok(()) => {},
            Err(e) => {

                if self.dt < 1e-24 {
                    panic!("solver failed, dt = {:?}, time = {:?}, steps = {:?}, error: {}", self.dt, self.time(), self.steps(), e)
                }

                self.dt /= 2.0;

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
        let hnew = 0.9 * self.dt * (epsilon / truncmax.max(1e-16).min(1e5)).powf(0.333333333).min(1.2);
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

            for i in 0..self.solution.len() {
                self.last_solution_2[i] = self.last_solution_1[i];
                self.last_solution_1[i] = self.last_solution_0[i];
                self.last_solution_0[i] = self.solution[i];
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



