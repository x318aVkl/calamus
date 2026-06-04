//!
//! 
//! ```rust
//! 
//! use calamus::functions::bessel::{bessel_first_kind, bessel_second_kind};
//! 
//! let x = bessel_first_kind(5.0, 1.0);
//! 
//! assert!((x + 0.3275791375914697).abs() < 1e-8);
//! 
//! let x = bessel_second_kind(5.0, 1e-10);
//! 
//! assert!((x + 0.30850055179078834).abs() < 1e-8);
//! 
//! ```
//! 
//! 
use crate::{functions::gamma::gamma, num_traits::FloatNumber};



pub fn bessel_first_kind<T>(x: T, alpha: T) -> T where T: FloatNumber {

    let mut m = 0;

    let mut sum = T::ZERO;

    let mut mi = 1;

    let mut lsum = sum;


    let mut mfact = T::ONE;
    loop {

        sum += T::from(mi) / (mfact * gamma(T::from(m) + alpha + T::ONE)) * (x / T::TWO).float_powf(T::TWO * T::from(m) + alpha);

        if (sum - lsum).float_abs() / sum.float_abs().float_max(T::ONE) < T::EPSILON {
            break;
        }

        lsum = sum;

        mi *= -1;

        m += 1;
        mfact *= T::from(m);

        if m > 1000 {break;}

    }

    T::TWO * sum - lsum
}


pub fn bessel_second_kind<T>(x: T, alpha: T) -> T where T: FloatNumber {


    let ja = bessel_first_kind(x, alpha);
    let jma = bessel_first_kind(x, - alpha);

    let alpha_pi = alpha * T::PI;

    (ja * alpha_pi.float_cos() - jma) / alpha_pi.float_sin()
}

