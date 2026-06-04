use calamus::prelude::*;


struct Problem;

impl RootProblem<f64> for Problem {
    type Error = ();
    fn size(&self) -> usize {
        3
    }

    fn residual(&self, residual: &mut impl calamus::linalg::vector::VectorMut<f64>, solution: &impl calamus::linalg::vector::Vector<f64>) -> Result<(), Self::Error> {

        let x = solution[0];
        let y = solution[1];
        let z = solution[2];

        residual[0] = x.powi(2) - y + z - 1.0;
        residual[1] = 2.0 * x + y.powi(2) - z - 2.0;
        residual[2] = x + y + z.powi(3) - 4.0;
                
        Ok(())
    }
}


fn main() {

    let mut solver = NewtonSolver::<_, _, DynamicLu<f64>>::new(Problem);

    solver.set_initial_guess(&[1.0, 1.0, 1.0]);

    let result = solver.solve().unwrap();

    println!("{:?}", result);

    let y = solver.solution();
    println!("solution = {:?}", &y[0..y.len()]);

}