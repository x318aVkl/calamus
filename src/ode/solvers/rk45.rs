use crate::{linalg::vector::{DynamicVector, Vector, VectorView}, num_traits::FloatNumber, ode::Ode, ode::solvers::Solver};



// Explicit adaptive Runge-Kutta 5 solver
// Good for non-stiff problems
pub struct ExplicitRk45<T, F: Ode<T>> {
    ode: F,
    solution: DynamicVector<T>,
    solution_tmp: DynamicVector<T>,
    k1: DynamicVector<T>,
    k2: DynamicVector<T>,
    k3: DynamicVector<T>,
    k4: DynamicVector<T>,
    k5: DynamicVector<T>,
    k6: DynamicVector<T>,
    t: T,
    dt: T,
    n_steps: usize,

    min_dt: T,
    te_tolerance: T,
}


impl<T, F: Ode<T>> ExplicitRk45<T, F> where T: FloatNumber {

    pub fn new(ode: F) -> Self {
        let size = ode.problem_size();
        Self {
            ode,
            solution: DynamicVector::new(T::ZERO, size),
            solution_tmp: DynamicVector::new(T::ZERO, size),
            k1: DynamicVector::new(T::ZERO, size),
            k2: DynamicVector::new(T::ZERO, size),
            k3: DynamicVector::new(T::ZERO, size),
            k4: DynamicVector::new(T::ZERO, size),
            k5: DynamicVector::new(T::ZERO, size),
            k6: DynamicVector::new(T::ZERO, size),
            t: T::ZERO,
            dt: T::fraction(1, 1_000_000),
            n_steps: 0,
            min_dt: T::EPSILON * T::fraction(1, 1_000_000),
            te_tolerance: T::fraction(1, 1_000_000),
        }
    }

    pub fn with_t0(mut self, t0: T) -> Self {
        self.t = t0;
        self
    }

    pub fn with_dt0(mut self, dt0: T) -> Self {
        self.dt = dt0;
        self
    }

    pub fn with_initial_guess(mut self, guess_function: impl Fn(usize) -> T) -> Self {

        for i in 0..self.solution.len() {
            self.solution[i] = guess_function(i);
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

}


impl<T, F: Ode<T>> Solver<T> for ExplicitRk45<T, F> where <F as Ode<T>>::Error: std::fmt::Debug, T: FloatNumber {
    
    type Error = F::Error;

    fn solution<'a>(&'a self) -> VectorView<'a, T> {
        self.solution.view()
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
        for i in 0..self.solution.len() {
            self.solution[i] = initial_solution[i];
        }
        self.t = time;
    }

    

    fn step(&mut self) -> Result<(), F::Error> {

        let alpha = [T::ZERO, T::fraction(2, 9), T::fraction(1, 3), T::fraction(3, 4), T::from(1), T::fraction(5, 6)];
        let beta = [
            [T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO],
            [T::fraction(2, 9), T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO],
            [T::fraction(1, 12), T::fraction(1, 4), T::ZERO, T::ZERO, T::ZERO, T::ZERO],
            [T::fraction(69, 128), T::fraction(-243, 128), T::fraction(135, 64), T::ZERO, T::ZERO, T::ZERO],
            [T::fraction(-17, 12), T::fraction(27, 4), T::fraction(-27, 5), T::fraction(16, 15), T::ZERO, T::ZERO],
            [T::fraction(65, 432), T::fraction(-5, 16), T::fraction(13, 16), T::fraction(4, 27), T::fraction(5, 144), T::ZERO]
        ];

        let c_4 = [T::fraction(1, 9), T::ZERO, T::fraction(9, 20), T::fraction(16, 45), T::fraction(1, 12), T::ZERO];
        let c_5 = [T::fraction(47, 450), T::ZERO, T::fraction(12, 25), T::fraction(32, 225), T::fraction(1, 30), T::fraction(6, 25)];

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

        let mut truncmax = T::EPSILON;
        for i in 0..self.solution.len() {
            let k = [self.k1[i], self.k2[i], self.k3[i], self.k4[i], self.k5[i], self.k6[i]];

            let mut truncerror = T::ZERO;
            self.solution_tmp[i] = self.solution[i];
            for j in 0..k.len() {
                self.solution_tmp[i] += k[j] * c_5[j];
                truncerror += (c_5[j] - c_4[j]) * k[j];
            }
            let truncerror = truncerror.float_abs();

            truncmax = truncmax.float_max(truncerror);
        }

        let epsilon = T::EPSILON.float_sqrt();
        let hnew = T::fraction(9, 10) * h * (epsilon / truncmax).float_powf(T::fraction(2, 10));
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



