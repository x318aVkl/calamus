// solve a stiff system of 5 species and temperature, with 3 reactions


use calamus::prelude::*;




struct Reaction;

impl Ode<f64> for Reaction {
    type Error = ();

    fn eval_f(&self, f: &mut impl calamus::linalg::vector::VectorMut<f64>, y: &impl calamus::linalg::vector::Vector<f64>, _t: f64) -> Result<(), Self::Error> {
        let a = y[0];
        let b = y[1];
        let c = y[2];
        let d = y[3];
        let e = y[4];
        let temp = y[5];
        
        // reaction 1: a + b = c + e
        // reaction 2: 2 * c + b = a + d
        // reaction 3: e + d = b

        // initial condition
        // a = 2, b = 1, c,d,e = 0, temp = 1

        // steady state results in
        // a = 1.5, b = 0, c = 1.7e-8, d = 0, e = 0.5, temp = 5.5

        // reaction 1 is moderately fast, happens at high temperature
        // reaction 2 is very fast, happens at low temperature
        // reacction 3 is slow, endothermic and decreases in strength with temperature

        let r1 = 2e5 * a * b * (- 10.0 / temp).exp();
        let r2 = 1e20 * c.powi(2) * b * (- 1.0 / temp).exp();
        let r3 = 1e3 * e * d * (0.5 / temp).exp();

        let s = 0.0; //if t > 5e-3 {1e2 * (0.5 - temp)} else {0.0};

        f[0] = -r1 + r2;
        f[1] = -r1 - r2 + r3;
        f[2] = r1 - 2.0 * r2;
        f[3] = r2 - r3;
        f[4] = r1 - r3;
        f[5] = 5.0 * r1 + 0.5 * r2 - 1.5 * r3 + s;

        Ok(())
    }
    fn problem_size(&self) -> usize {
        6
    }
}




fn main() {

    let mut solver = Bdf2::new(Reaction)
        .with_dt0(1e-7)
        .with_initial_guess(&[2.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        .with_min_dt(1e-12)
        .with_tolerance(1e-3)
        ;

    for i in 1..=200 {
        solver.solve_to((i as f64) * 0.00002).unwrap();
        let y = solver.solution();
        println!("{} {} {} {} {} {} {}", solver.time(), y[0], y[1], y[2], y[3], y[4], y[5]);
    }
}
