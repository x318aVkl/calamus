

use calamus::prelude::*;


struct NBodyProblem {
    masses: Vec<f64>,
}

impl Ode<f64> for NBodyProblem {
    type Error = ();

    fn eval_f(&self, f: &mut impl calamus::linalg::vector::VectorMut<f64>, y: &impl calamus::linalg::vector::Vector<f64>, _t: f64) -> Result<(), Self::Error> {
        //let g = 6.6743e-11;
        let g = 0.1;

        for i in 0..self.masses.len() {
            
            let mi = self.masses[i];

            let xi = y[4*i];
            let yi = y[4*i+1];

            let vx = y[4*i+2];
            let vy = y[4*i+3];

            let mut ax = 0.0;
            let mut ay = 0.0;
            for j in 0..self.masses.len() {
                if i == j {continue;}
                let mj = self.masses[j];

                let xj = y[4*j];
                let yj = y[4*j+1];

                let r2 = (xi - xj).powi(2) + (yi - yj).powi(2);

                let fij = g * mi * mj / r2;

                let r = r2.sqrt();

                let fijx = fij * (xj - xi) / r;
                let fijy = fij * (yj - yi) / r;

                ax += fijx / mi;
                ay += fijy / mi;
            }

            f[4*i + 0] = vx;
            f[4*i + 1] = vy;
            f[4*i + 2] = ax;
            f[4*i + 3] = ay;
        }

        Ok(())
    }

    fn problem_size(&self) -> usize {
        // x, y position for all objects
        // vx, vy velocities for all objects
        self.masses.len() * 4
    }
}




fn main() {


    let ode = NBodyProblem {
        masses: vec![1.0, 1.0, 1.0]
    };

    let v = 0.1;

    let s3 = 3.0_f64.sqrt() / 2.0;

    let x0 = [
        s3, -0.5, v*0.5, v*s3,
        -s3, -0.5, v*0.5, -v*s3,
        0.0, 1.0, -v, 0.0,
    ];

    let mut solver = ExplicitRk45::new(ode)
        .with_initial_guess(|i| {
            x0[i]
        });
    
    let mut time = 0.0;
    let dt = 0.1;
    
    for _i in 0..=200 {
        solver.solve_to(time + dt).unwrap();
        time += dt;

        let y = solver.solution();

        print!("{} ", solver.time());
        for k in 0..y.len() {
            print!("{} ", y[k]);
        }
        print!("\n");
    }

}

