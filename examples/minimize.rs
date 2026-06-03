use calamus::prelude::*;


struct Problem;


impl MinimizeProblem<f64> for Problem {
    type Error = ();

    fn size(&self) -> usize {
        3
    }
    fn cost(&self, solution: &impl calamus::linalg::vector::Vector<f64>) -> Result<f64, Self::Error> {
        let x = solution[0];
        let y = solution[1];
        let l = solution[2];

        // minimize the function x^2 + (y - 0.5)^2 subject to the constraint 1 = x^2 - y^2
        // lagrange multiplier method

        // cost function
        let f = x.powi(2) + (y - 0.5).powi(2);

        // constraint
        let g = 1.0 - x.powi(2) + y.powi(2);

        Ok(f + l * g)
    }
}


fn main() {

    let mut solver = NewtonSolver::<_, _, DynamicLu<f64>>::new(NewtonMinimizationProblem::new(Problem));

    solver.set_initial_guess(&[1.0, 1.0, 1.0]);

    println!("initial cost = {:?}", Problem.cost(&solver.solution()).unwrap());

    let result = solver.solve().unwrap();

    println!("{:?}", result);

    let y = solver.solution();
    print!("solution = (");
    for i in 0..y.len() {
        print!("{:?}, ", y[i]);
    }
    print!(")\n");

    println!("final cost = {:?}", Problem.cost(&solver.solution()).unwrap());

}