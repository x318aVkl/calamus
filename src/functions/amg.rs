//!
//! Computes the Arithmetic-geometric mean
//! 
//! ```rust
//! 
//! use scixl::functions::amg::amg;
//! 
//! let x = amg(24.0, 6.0);
//! 
//! assert!((x - 13.458171481725616).abs() < 1e-14);
//! 
//! ```
//! 
//! 
use crate::num_traits::FloatNumber;



pub fn amg<T: FloatNumber>(x: T, y: T) -> T {

    let mut a = x;
    let mut b = y;

    loop {

        let an = (a + b) * T::HALF;
        b = (a * b).float_sqrt();

        a = an;

        if (a - b).float_abs() < T::EPSILON {
            break;
        }
    }


    (a + b) * T::HALF
}





