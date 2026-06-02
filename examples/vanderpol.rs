
use lwode::{traits::*, Ode, OdeJacobian, solvers::Bdf2};

struct VanDerPol {
    kappa: f64,
}

impl Ode for VanDerPol {
    type Error = ();

    fn eval_f(&self, f: &mut [f64], y: &[f64], t: f64) -> Result<(), Self::Error> {
        let x = y[0];
        let y = y[1];

        f[0] = 2.0 * self.kappa * y;
        f[1] = 2.0 * self.kappa.powi(2) * (1.0 - x.powi(2)) * y - 2.0 * self.kappa * x;

        Ok(())
    }
    fn problem_size(&self) -> usize {
        2
    }
}

impl OdeJacobian for VanDerPol {
    // define the analytic jacobian
    fn eval_jacobian(&self, jac: &mut [f64], y: &[f64], t: f64) -> Result<(), <Self as Ode>::Error> {

        let x = y[0];
        let y = y[1];
        let k = self.kappa;

        jac[0] = 0.0;
        jac[1] = 2.0 * k;

        jac[2] = - 4.0 * k.powi(2) * x * y - 2.0 * k;
        jac[3] = 2.0 * k.powi(2) * (1.0 - x.powi(2));

        Ok(())
    }
}


fn main() {

    let ode =  VanDerPol {kappa : 200.0};

    let mut solver = Bdf2::new(ode)
        .with_dt0(1e-7)
        .with_initial_guess(|i| {
            let a = [2.0, 0.0];
            a[i]
        })
        .with_min_dt(1e-8)
        .with_tolerance(1e-3)
        ;

    for i in 1..=300 {
        solver.solve_to((i as f64) * 0.01).unwrap();
        let y = solver.solution();
        println!("{} {}", solver.time(), y[0]);
    }

}

