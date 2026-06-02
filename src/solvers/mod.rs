

pub mod rk45;
pub mod bdf2;


pub use rk45::ExplicitRk45;
pub use bdf2::Bdf2;

pub trait Solver {

    type Error: std::fmt::Debug;

    fn step(&mut self) -> Result<(), Self::Error>;

    fn solution(&self) -> &[f64];

    fn time(&self) -> f64;

    fn steps(&self) -> usize;

    fn step_size(&self) -> f64;

    fn set_step_size(&mut self, dt: f64);


    /// Re-initialize the solver, cleans up data, will work as if the solver was just created
    /// will also erase memory of multi-steps solvers
    /// use this per example when coupling with another solver for CFD and running on many cells to avoid re-allocation
    fn reinit(&mut self, initial_solution: &[f64], time: f64);


    fn solve_to(&mut self, t1: f64) -> Result<(), Self::Error> {

        if (self.time() + self.step_size()) >= t1 {
            self.set_step_size((t1 - self.time()) * 0.5);
        }

        let mut savestepsize = self.step_size();

        loop {

            self.step()?;
            
            if (self.time() - t1) / self.step_size() > -1e-9 {
                break;
            }

            if (self.time() + self.step_size() - t1) / self.step_size() > -1e-9 {
                savestepsize = self.step_size();
                self.set_step_size(t1 - self.time());
            }
        }

        self.set_step_size(savestepsize);

        Ok(())
    }

}

