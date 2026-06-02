

// Odes have the form dy/dt = f(t,y)



pub trait Ode {
    type Error;
    fn eval_f(&self, f: &mut [f64], y: &[f64], t: f64) -> Result<(), Self::Error>;
    fn problem_size(&self) -> usize;
}


const MAX_SIZE: usize = 256;


pub trait OdeJacobian: Ode {

    // evaluate the problem jacobian
    // jac is in row-first ordering
    // [row_1, row_2, row_3 ...]
    // [a_00, a_01, a_02 ... a_10, a_11, a_12 ...]
    // jac[i*size + j] must be derivative of f[i] over variable j
    // default implementation uses a difference quotient automatic formulation
    fn eval_jacobian(&self, jac: &mut [f64], y: &[f64], t: f64) -> Result<(), <Self as Ode>::Error> {
        if y.len() > MAX_SIZE {
            panic!("Error, problem size must be smaller or equal to {}", MAX_SIZE);
        }

        // by default, use a difference quotient approach

        let n = self.problem_size();

        // allocate temporary vectors
        let mut f_alloc = [0.0; MAX_SIZE];
        let mut fb_alloc = [0.0; MAX_SIZE];
        let mut yd_alloc = [0.0; MAX_SIZE];

        let f = &mut f_alloc[0..y.len()];
        let fb = &mut fb_alloc[0..y.len()];
        let yd = &mut yd_alloc[0..y.len()];
        for i in 0..n {
            yd[i] = y[i];
        }

        // base f
        self.eval_f(f, y, t)?;
        
        for i in 0..n {
            let y0 = y[i];
            let dy = 1e-6_f64.max(y[i].abs() * 1e-6);
            yd[i] = y0 + dy;

            self.eval_f(fb, &yd, t)?;

            for j in 0..n {
                jac[j*n + i] = (fb[j] - f[j]) / dy;
            }

            yd[i] = y0;
        }

        Ok(())
    }
}



