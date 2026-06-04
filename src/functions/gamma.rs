//!
//! 
//! ```rust
//! 
//! use calamus::functions::gamma::gamma;
//! use calamus::complex::Complex;
//! 
//! let x = gamma(5.0);
//! 
//! assert!((x - 24.0).abs() < 1e-12);
//! 
//! let x = Complex::<f64>::new(1.0, 0.5);
//! let sol = Complex::<f64>::new(0.8016940970697171, -0.19963973816459632);
//! 
//! assert!((gamma(x) - sol).abs() < 1e-12);
//! 
//! 
//! ```
//! 
//! 
use crate::num_traits::FloatNumber;







pub fn gamma<T>(z: T) -> T where T: FloatNumber {

    // Lanczos method for gamma function
    // about 1e-13 accuratte

    if z < T::HALF {
        return T::PI / (T::PI * z).float_sin() * gamma(T::ONE - z);
    }

    let mut z = z;

    let g = T::float_from_f64(4.7421875);
    let p = [
        0.99999999999999709182,
        57.156235665862923517,
        -59.597960355475491248,
        14.136097974741747174,
        -0.49191381609762019978,
        0.33994649984811888699e-4,
        0.46523628927048575665e-4,
        -0.98374475304879564677e-4,
        0.15808870322491248884e-3,
        -0.21026444172410488319e-3,
        0.21743961811521264320e-3,
        -0.16431810653676389022e-3,
        0.84418223983852743293e-4,
        -0.26190838401581408670e-4,
        0.36899182659531622704e-5, 
    ];

    z -= T::ONE;

    let mut x = T::float_from_f64(p[0]);
    for i in 1..(p.len() as i32) {

        x += T::float_from_f64(p[i as usize]) / (z + T::from(i));

    }

    let t = z + T::from(g) + T::HALF;
    (T::TWO * T::PI).float_sqrt() * t.float_powf(z + T::HALF) * (-t).float_exp() * x
}

