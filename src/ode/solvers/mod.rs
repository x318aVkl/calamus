

pub mod rk45;
pub mod bdf2;


pub use rk45::ExplicitRk45;
pub use bdf2::Bdf2;

use crate::{linalg::vector::{Vector, VectorView}, num_traits::FloatNumber};

pub trait Solver<T> {

    type Error: std::fmt::Debug;

    /// step forward the solution
    fn step(&mut self) -> Result<(), Self::Error>;

    fn solution<'a>(&'a self) -> VectorView<'a, T>;

    fn time(&self) -> T;

    fn steps(&self) -> usize;

    fn step_size(&self) -> T;

    fn set_step_size(&mut self, dt: T);


    /// Re-initialize the solver, cleans up data, will work as if the solver was just created
    /// will also erase memory of multi-steps solvers
    /// use this per example when coupling with another solver for CFD and running on many cells to avoid re-allocation
    fn reinit(&mut self, initial_solution: &impl Vector<T>, time: T);


    /// Solve up to the given time
    fn solve_to(&mut self, t1: T) -> Result<(), Self::Error> where T: FloatNumber {

        if (self.time() + self.step_size()) >= t1 {
            self.set_step_size((t1 - self.time()) * T::fraction(1, 2));
        }

        let mut savestepsize = self.step_size();

        let tol = T::EPSILON.float_sqrt();

        loop {

            self.step()?;
            
            if (self.time() - t1) / self.step_size() > - tol {
                break;
            }

            if (self.time() + self.step_size() - t1) / self.step_size() > - tol {
                savestepsize = self.step_size();
                self.set_step_size(t1 - self.time());
            }
        }

        self.set_step_size(savestepsize);

        Ok(())
    }

}

