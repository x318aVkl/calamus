//! A light-weight fast ODE solver for possibly stiff problems.
//! Includes a [Ode] trait to define user system of differential equations of the form:
//! dy/dt = f(t,y,p)
//! with p solver-constant user defined parameters contained in the Ode struct.
//! Includes two solvers:
//! - [ExplicitRk45], an implementation of the Runge-Kutta Faulberg method, suitable for non-stiff problems
//! - [Bdf2], an adaptive implicit second order BDF-2 solver build for stiff problems
//!
//! Due to the pure Rust nature of the crate, the compromise made is using dense Jacobians in [Bdf2], so [Bdf2] is not suitable for large sparse problems.
//!
//! Per example, solving the VanDerPol stiff equation:
//! ```rust
//! use scixl::prelude::*;
//! 
//! // Struct defining the Ode problem's constant data
//! struct VanDerPol {
//!     kappa: f64,
//! }
//! 
//! // Implementation of the Ode trait for our problem
//! impl Ode<f64> for VanDerPol {
//!     type Error = ();
//!     
//!     // evaluate the f term: dy/dt = f(t,y,p)
//!     fn eval_f(&self, f: &mut impl VectorMut<f64>, y: &impl Vector<f64>, _t: f64) -> Result<(), Self::Error> {
//!         let x = y[0];
//!         let y = y[1];
//! 
//!         f[0] = 2.0 * self.kappa * y;
//!         f[1] = 2.0 * self.kappa.powi(2) * (1.0 - x.powi(2)) * y - 2.0 * self.kappa * x;
//! 
//!         Ok(())
//!     }
//! 
//!     fn problem_size(&self) -> usize {
//!         2
//!     }
//! }
//! 
//! 
//! fn main() -> Result<(), ()> {
//!     
//!     // Create the problem data
//!     let ode = VanDerPol { kappa: 200.0 };
//!        
//!     // Create the solver and set parameters
//!     let mut solver = Bdf2::new(ode)
//!         .with_tolerance(1e-3)
//!         .with_min_dt(1e-8)
//!         .with_initial_guess(&[2.0, 0.0]);
//!     
//!     // First, solve to a time of 0.4, solution should still be positive
//!     solver.solve_to(0.4)?;
//!     assert!(solver.solution()[0] > 0.0);    
//!     
//!     // Then, move to 0.42, solution should flip to negative
//!     solver.solve_to(0.42)?;
//!     assert!(solver.solution()[0] < 0.0);
//!
//!     Ok(()) 
//! }
//! 
//! ```
//! 



pub mod ode;
pub mod solvers;


pub use ode::Ode;

pub use solvers::{
    ExplicitRk45, Bdf2,
};


pub mod traits {
    pub use super::ode::Ode;
    pub use super::solvers::Solver;
}


