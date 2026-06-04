use calamus::{optimize::minimize::GradientDescentSolver, prelude::*};


#[derive(Clone, Copy)]
struct Problem { kappa: f64 }


impl Problem {
    fn cost(&self, x: f64, y: f64) -> f64 {
        x + y
    }
    fn constraint(&self, x: f64, y: f64) -> f64 {
        1.0 - x.powi(2) - (self.kappa * y).powi(2)
    }
}


struct Penalty(Problem, f64);

impl MinimizationProblem<f64> for Penalty {
    type Error = ();

    fn size(&self) -> usize {
        2
    }
    fn cost(&self, solution: &impl calamus::linalg::vector::Vector<f64>) -> Result<f64, Self::Error> {
        let x = solution[0];
        let y = solution[1];

        // use a penalty method
        Ok(self.0.cost(x, y) + self.0.constraint(x, y).powi(2) * self.1)
    }
}


struct LagrangeMultiplier(Problem);


impl MinimizationProblem<f64> for LagrangeMultiplier {
    type Error = ();
    fn size(&self) -> usize {
        3
    }
    fn cost(&self, solution: &impl Vector<f64>) -> Result<f64, Self::Error> {
        let x = solution[0];
        let y = solution[1];
        let l = solution[2];

        // use lagrange multiplier method

        Ok(self.0.cost(x, y) + l * self.0.constraint(x, y))
    }
}


fn main() {

    // create the problem definition with parameters
    // here, kappa defines the shape of the constraint
    // kappa = 1.0, circle
    // kappa = 2.0, ellipse of height = width/2
    let problem = Problem { kappa: 2.0 };
    let penalty = 10.0;

    // Use the gradient descent solver to find a rough spot for the solution
    // lets the newton solver start close to a minima, and not a maxima
    let mut solver = GradientDescentSolver::new(Penalty(problem, penalty));

    solver.step_size = 0.05;
    solver.max_steps = 500;
    solver.tolerance = 0.1;

    let result = solver.solve().unwrap();

    let y = solver.solution();

    println!("{:?}", result);
    println!("intermediate solution = {:?}", &y[0..2]);

    // Fine tune the solution using the more costly newton solver
    // since we start close to the solution, this should not take a lot of iterations
    let mut solver = NewtonSolver::<_, _, DynamicLu<f64>>::new(NewtonMinimizationProblem::new(LagrangeMultiplier(problem)));
    solver.set_initial_guess(&[y[0], y[1], 1.0]);

    println!("intermediate cost = {:?}", problem.cost(y[0], y[1]));
    println!("intermediate constraint = {:?}", problem.constraint(y[0], y[1]));
     println!();

    let result = solver.solve().unwrap();

    println!("{:?}", result);

    let y = solver.solution();
    println!("solution = {:?}", &y[0..2]);

    println!("final cost = {:?}", problem.cost(y[0], y[1]));
    println!("final constraint = {:?}", problem.constraint(y[0], y[1]));

}