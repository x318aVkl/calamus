

pub mod num_traits;
pub mod linalg;
pub mod optimize;
pub mod ode;


pub mod traits {
    use super::*;


    pub use ode::traits::*;
}





pub mod prelude {


    use super::*;



    pub use num_traits::FloatNumber;

    pub use linalg::{
        vector::{Vector, VectorMut, DynamicVector},
        matrix::{Matrix, MatrixMut, DynamicMatrix},
        lu::DynamicLu,
    };

    pub use optimize::{
        root::{NewtonSolver, RootProblem},
        minimize::{MinimizeProblem, NewtonMinimizationProblem},
    };

    pub use ode::{
        Ode, Bdf2, ExplicitRk45, traits::*,
    };
}

