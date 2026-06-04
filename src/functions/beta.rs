//!
//! 
//! ```rust
//! 
//! 
//! use calamus::functions::beta::{beta, beta_incomplete};
//! 
//! assert!((beta(0.5, 1.0) - 2.0).abs() < 1e-14);
//! assert!((beta_incomplete(0.5, 1.5, 2.2) - 0.15427120198153296).abs() < 1e-14);
//! 
//! ```
//! 
//! 

use crate::num_traits::FloatNumber;


use super::gamma::gamma;


pub fn beta<T>(x: T, y: T) -> T where T: FloatNumber {
    // definition from gamma function
    gamma(x) * gamma(y) / gamma(x + y)
}


pub fn beta_incomplete<T>(x: T, a: T, b: T) -> T where T: FloatNumber {
    // definition using continued fraction expansion

    let mut anm1_maj = T::ONE;
    let mut bnm1_maj = T::ZERO;

    let mut n = 1;

    let mut an_maj = T::ONE;
    let mut bn_maj = T::ONE;

    loop {

        let an = if n % 2 == 1 {
            let mt = (T::from(n) - T::ONE) / T::TWO;
            - (a + mt) * (a + b + mt) * x / ((a + T::TWO*mt) * (a + T::TWO*mt + T::ONE))
        } else {
            let mt = T::from(n) / T::TWO;
            mt * (b - mt) * x / ((a + T::TWO*mt - T::ONE)*(a + T::TWO*mt))
        };
        //let an = T::ONE;
        let bn = T::ONE;

        let an_save = an_maj;
        let bn_save = bn_maj;

        an_maj = bn * an_save + an * anm1_maj;
        bn_maj = bn * bn_save + an * bnm1_maj;

        anm1_maj = an_save;
        bnm1_maj = bn_save;

        if ((an_maj / bn_maj) - (anm1_maj / bnm1_maj)).float_abs() < (T::EPSILON * T::from(1000)) {
            break;
        }
        n += 1;

        if n > 100 {break}
    }

    let denom = T::TWO * an_maj / bn_maj - anm1_maj / bnm1_maj;

    x.float_powf(a) * (T::ONE - x).float_powf(b) / (a * denom)
    //denom
}



