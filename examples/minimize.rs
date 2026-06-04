use calamus::{optimize::minimize::GradientDescentSolver, prelude::*};


struct Problem;


impl MinimizationProblem<f64> for Problem {
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
        let f = x + y;

        // constraint, penalty method
        let g = 1.0 - x.powi(2) - y.powi(2);

        Ok(f + l * g)
    }
}


fn main() {


    let mut solver = GradientDescentSolver::new(Problem);

    solver.step_size = 0.1;
    solver.max_steps = 500;
    solver.tolerance = 0.05;

    let result = solver.solve().unwrap();

    let y = solver.solution();

    println!("{:?}", result);

    let mut solver = NewtonSolver::<_, _, DynamicLu<f64>>::new(NewtonMinimizationProblem::new(Problem));

    solver.set_initial_guess(&y);

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