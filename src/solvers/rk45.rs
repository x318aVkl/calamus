use crate::{ode::Ode, solvers::Solver};



// Explicit adaptive Runge-Kutta 5 solver
// Good for non-stiff problems
pub struct ExplicitRk45<F: Ode> {
    ode: F,
    solution: Vec<f64>,
    solution_tmp: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    k5: Vec<f64>,
    k6: Vec<f64>,
    t: f64,
    dt: f64,
    n_steps: usize,

    min_dt: f64,
    te_tolerance: f64,
}


impl<F: Ode> ExplicitRk45<F> {

    pub fn new(ode: F) -> Self {
        let size = ode.problem_size();
        Self {
            ode,
            solution: vec![0.0; size],
            solution_tmp: vec![0.0; size],
            k1: vec![0.0; size],
            k2: vec![0.0; size],
            k3: vec![0.0; size],
            k4: vec![0.0; size],
            k5: vec![0.0; size],
            k6: vec![0.0; size],
            t: 0.0,
            dt: 1e-6,
            n_steps: 0,
            min_dt: 1e-20,
            te_tolerance: 1e-6,
        }
    }

    pub fn with_t0(mut self, t0: f64) -> Self {
        self.t = t0;
        self
    }

    pub fn with_dt0(mut self, dt0: f64) -> Self {
        self.dt = dt0;
        self
    }

    pub fn with_initial_guess(mut self, guess_function: impl Fn(usize) -> f64) -> Self {

        for i in 0..self.solution.len() {
            self.solution[i] = guess_function(i);
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

}


impl<F: Ode> Solver for ExplicitRk45<F> where <F as Ode>::Error: std::fmt::Debug {
    
    type Error = F::Error;

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
        for i in 0..self.solution.len() {
            self.solution[i] = initial_solution[i];
        }
        self.t = time;
    }

    

    fn step(&mut self) -> Result<(), F::Error> {

        let alpha = [0.0, 2.0/9.0, 1.0/3.0, 3.0/4.0, 1.0, 5.0/6.0];
        let beta = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0/9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0/12.0, 1.0/4.0, 0.0, 0.0, 0.0, 0.0],
            [69.0/128.0, -243.0/128.0, 135.0/64.0, 0.0, 0.0, 0.0],
            [-17.0/12.0, 27.0/4.0, -27.0/5.0, 16.0/15.0, 0.0, 0.0],
            [65.0/432.0, -5.0/16.0, 13.0/16.0, 4.0/27.0, 5.0/144.0, 0.0]
        ];

        let c_4 = [1.0/9.0, 0.0, 9.0/20.0, 16.0/45.0, 1.0/12.0, 0.0];
        let c_5 = [47.0/450.0, 0.0, 12.0/25.0, 32.0/225.0, 1.0/30., 6.0/25.0];

        let t = self.t;
        let h = self.dt;

        // step 0, compute k1
        self.ode.eval_f(&mut self.k1, &self.solution, t)?;
        for i in 0..self.solution_tmp.len() {
            self.k1[i] *= h;
        }

        // step 1, compute k2
        let tk = t + alpha[1] * h;
        for i in 0..self.solution_tmp.len() {
            self.solution_tmp[i] = self.solution[i] + self.k1[i] * beta[1][0];
        }
        self.ode.eval_f(&mut self.k2, &self.solution_tmp, tk)?;
        for i in 0..self.solution_tmp.len() {
            self.k2[i] *= h;
        }

        // step 2, compute k3
        let tk = t + alpha[2] * h;
        for i in 0..self.solution_tmp.len() {
            self.solution_tmp[i] = self.solution[i] + self.k1[i] * beta[2][0] + self.k2[i] * beta[2][1];
        }
        self.ode.eval_f(&mut self.k3, &self.solution_tmp, tk)?;
        for i in 0..self.solution_tmp.len() {
            self.k3[i] *= h;
        }

        // step 3, compute k4
        let tk = t + alpha[3] * h;
        for i in 0..self.solution_tmp.len() {
            self.solution_tmp[i] = self.solution[i] + self.k1[i] * beta[3][0] + self.k2[i] * beta[3][1] + self.k3[i] * beta[3][2];
        }
        self.ode.eval_f(&mut self.k4, &self.solution_tmp, tk)?;
        for i in 0..self.solution_tmp.len() {
            self.k4[i] *= h;
        }

        // step 4, compute k5
        let tk = t + alpha[4] * h;
        for i in 0..self.solution_tmp.len() {
            self.solution_tmp[i] = self.solution[i] + self.k1[i] * beta[4][0] + self.k2[i] * beta[4][1] + self.k3[i] * beta[4][2] + self.k4[i] * beta[4][3];
        }
        self.ode.eval_f(&mut self.k5, &self.solution_tmp, tk)?;
        for i in 0..self.solution_tmp.len() {
            self.k5[i] *= h;
        }

        // step 5, compute k6
        let tk = t + alpha[5] * h;
        for i in 0..self.solution_tmp.len() {
            self.solution_tmp[i] = self.solution[i] + self.k1[i] * beta[5][0] + self.k2[i] * beta[5][1] + self.k3[i] * beta[5][2] + self.k4[i] * beta[5][3] + self.k5[i] * beta[5][4];
        }
        self.ode.eval_f(&mut self.k6, &self.solution_tmp, tk)?;
        for i in 0..self.solution_tmp.len() {
            self.k6[i] *= h;
        }

        let mut truncmax: f64 = 1e-20;
        for i in 0..self.solution.len() {
            let k = [self.k1[i], self.k2[i], self.k3[i], self.k4[i], self.k5[i], self.k6[i]];

            let mut truncerror = 0.0;
            self.solution_tmp[i] = self.solution[i];
            for j in 0..k.len() {
                self.solution_tmp[i] += k[j] * c_5[j];
                truncerror += (c_5[j] - c_4[j]) * k[j];
            }
            let truncerror = truncerror.abs();

            truncmax = truncmax.max(truncerror);
        }

        let epsilon = 1e-8;
        let hnew = 0.9 * h * (epsilon / truncmax).powf(0.2);
        self.dt = hnew;

        let mut ignore_terror = false;
        if self.dt < self.min_dt {
            self.dt = self.min_dt;
            ignore_terror = true;
        }

        if (truncmax > epsilon) && !ignore_terror {
            self.step()?;
        } else {
            // good! swap solution
            for i in 0..self.solution.len() {
                self.solution[i] = self.solution_tmp[i];
            }
            self.t += h;
            self.n_steps += 1;
        }

        Ok(())
    }

}



