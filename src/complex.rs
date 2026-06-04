//!
//! ```rust
//! 
//! use calamus::complex::Complex;
//! 
//! // test complex multiplication
//! // with pure rotation
//! let a = Complex::<f64>::new(2.0, 0.25);
//! let b = Complex::<f64>::new(0.0, -1.0);
//! 
//! let c = a * b;
//! 
//! // check the norm of c is the same as a
//! assert!((a.abs() - c.abs()).abs() < 1e-14);
//! 
//! ```
//! 

use crate::num_traits::FloatNumber;



#[derive(Clone, Copy, Debug)]
pub struct Complex<T> {
    real: T,
    imag: T,
}


impl<T> core::fmt::Display for Complex<T> where T: core::fmt::Display + FloatNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        core::fmt::Display::fmt(&self.real, f)?;
        if self.imag > T::ZERO {
            write!(f, "+")?;
        } else {
            write!(f, "-")?;
        }
        core::fmt::Display::fmt(&self.imag.float_abs(), f)?;
        write!(f, "i")?;
        Ok(())
    }
}

impl<T> core::fmt::LowerExp for Complex<T> where T: core::fmt::LowerExp + FloatNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        core::fmt::LowerExp::fmt(&self.real, f)?;
        if self.imag > T::ZERO {
            write!(f, "+")?;
        } else {
            write!(f, "-")?;
        }
        core::fmt::LowerExp::fmt(&self.imag.float_abs(), f)?;
        write!(f, "i")?;
        Ok(())
    }
}


impl<T> Complex<T> where T: FloatNumber {
    pub fn new(real: T, imag: T) -> Self {
        Self { real, imag }
    }

    pub fn real(self) -> T {
        self.real
    }

    pub fn imag(self) -> T {
        self.imag
    }

    pub fn abs_squared(self) -> T {
        self.real*self.real + self.imag*self.imag
    }

    pub fn abs(self) -> T {
        self.abs_squared().float_sqrt()
    }

    pub fn exp(self) -> Self {
        let x = self.real;
        let y = self.imag;

        let ex = x.float_exp();

        Self { real: y.float_cos() * ex, imag: y.float_sin() * ex }
    }

    pub fn principal_sqrt(self) -> Self {
        let c = self.real;
        let d = self.imag;

        let c2d2sqrt = (c*c + d*d).float_sqrt();

        Self {
            real: ((c + c2d2sqrt) / T::TWO).float_sqrt(),
            imag: d / d.float_abs().float_max(T::EPSILON) * ((- c + c2d2sqrt) / T::TWO).float_sqrt(),
        }
    }

    pub fn min(self, rhs: Self) -> Self {
        if self.abs_squared() <= rhs.abs_squared() {
            self
        } else {
            rhs
        }
    }
    pub fn max(self, rhs: Self) -> Self {
        if self.abs_squared() >= rhs.abs_squared() {
            self
        } else {
            rhs
        }
    }

    pub fn mag_angle(self) -> (T, T) {
        let a = self.real;
        let b = self.imag;
        (
            (a*a + b*b).float_sqrt(),
            b.float_atan2(a)
        )
    }

    pub fn powi(self, rhs: i32) -> Self {
        let (r, theta) = self.mag_angle();
        let rn = r.float_powi(rhs);
        let tn = theta * T::from(rhs);
        Self {
            real: rn * tn.float_cos(),
            imag: rn * tn.float_sin(),
        }
    }

    pub fn powf(self, rhs: Self) -> Self {
        let (r, theta) = self.mag_angle();
        let c = rhs.real;
        let d = rhs.imag;
        let lnr = r.float_ln();
        let arg = Self {
            real: lnr*c - theta*d,
            imag: lnr*d + theta*c
        };
        arg.exp()
    }

    pub fn cos(self) -> Self {
        let a = self.real;
        let b = self.imag;

        let bexp = b.float_exp();
        let mbexp = (-b).float_exp();

        let coshb = (bexp + mbexp) / T::TWO;
        let sinhb = (bexp - mbexp) / T::TWO;

        Self {
            real: a.float_cos() * coshb,
            imag: - a.float_sin() * sinhb,
        }
    }

    pub fn sin(self) -> Self {
        let a = self.real;
        let b = self.imag;

        let bexp = b.float_exp();
        let mbexp = (-b).float_exp();

        let coshb = (bexp + mbexp) / T::TWO;
        let sinhb = (bexp - mbexp) / T::TWO;

        Self {
            real: a.float_sin() * coshb,
            imag: a.float_cos() * sinhb,
        }
    }

    pub fn principal_ln(self) -> Self {
        let (r, theta) = self.mag_angle();
        Self {
            real: r.float_ln(),
            imag: theta,
        }
    }
}


impl<T> PartialEq for Complex<T> where T: PartialEq {
    fn eq(&self, other: &Self) -> bool {
        self.real.eq(&other.real)
    }
}

impl<T> PartialOrd for Complex<T> where T: PartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(&other.real)
    }
}


impl<T> core::ops::Neg for Complex<T> where T: core::ops::Neg<Output = T> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        self.real = -self.real;
        self.imag = -self.imag;
        self
    }
}

impl<T> From<T> for Complex<T> where T: FloatNumber {
    fn from(value: T) -> Self {
        Self { real: value, imag: T::ZERO }
    }
}

impl<T> From<u8> for Complex<T> where T: From<u8> {
    fn from(value: u8) -> Self {
        Self { real: T::from(value), imag: T::from(0) }
    }
}

impl<T> From<i32> for Complex<T> where T: From<i32> {
    fn from(value: i32) -> Self {
        Self { real: T::from(value), imag: T::from(0) }
    }
}



impl<T> core::ops::AddAssign for Complex<T> where T: FloatNumber {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.imag += rhs.imag;
    }
}

impl<T> core::ops::Add<Self> for Complex<T> where T: FloatNumber {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}


impl<T> core::ops::SubAssign for Complex<T> where T: FloatNumber {
    fn sub_assign(&mut self, rhs: Self) {
        self.real -= rhs.real;
        self.imag -= rhs.imag;
    }
}

impl<T> core::ops::Sub<Self> for Complex<T> where T: FloatNumber {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}


impl<T> core::ops::MulAssign for Complex<T> where T: FloatNumber {
    fn mul_assign(&mut self, rhs: Self) {
        let a = self.real;
        let b = self.imag;
        let c = rhs.real;
        let d = rhs.imag;

        self.real = a*c - b*d;
        self.imag = a*d + b*c;
    }
}

impl<T> core::ops::Mul<Self> for Complex<T> where T: FloatNumber {
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}


impl<T> core::ops::DivAssign for Complex<T> where T: FloatNumber {
    fn div_assign(&mut self, rhs: Self) {
        let a = self.real;
        let b = self.imag;
        let c = rhs.real;
        let d = rhs.imag;

        let c2d2 = c*c + d*d;

        self.real = (a*c + b*d) / c2d2;
        self.imag = (b*c - a*d) / c2d2;
    }
}

impl<T> core::ops::Div<Self> for Complex<T> where T: FloatNumber {
    type Output = Self;
    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}




impl<T> FloatNumber for Complex<T> where T: FloatNumber {
    const EPSILON: Self = Self { real: T::EPSILON, imag: T::ZERO };
    const ZERO: Self = Self { real: T::ZERO, imag: T::ZERO };
    const ONE: Self = Self { real: T::ONE, imag: T::ZERO };
    const TWO: Self = Self { real: T::TWO, imag: T::ZERO };
    const HALF: Self = Self { real: T::HALF, imag: T::ZERO };

    const PI: Self = Self { real: T::PI, imag: T::ZERO };

    fn float_abs(self) -> Self {
        Self { real: self.abs(), imag: T::ZERO }
    }

    fn float_sqrt(self) -> Self {
        self.principal_sqrt()
    }

    fn float_exp(self) -> Self {
        self.exp()
    }

    fn float_max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    fn float_min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    fn fraction(a: i32, b: i32) -> Self {
        Self { real: T::from(a) / T::from(b), imag: T::ZERO }
    }

    fn float_powf(self, exponent: Self) -> Self {
        self.powf(exponent)
    }
    fn float_powi(self, exponent: i32) -> Self {
        self.powi(exponent)
    }

    fn float_from_f64(x: f64) -> Self {
        Self { real: T::float_from_f64(x), imag: T::ZERO }
    }

    fn float_sin(self) -> Self {
        self.sin()
    }
    fn float_cos(self) -> Self {
        self.cos()
    }
    fn float_atan2(self, rhs: Self) -> Self {
        Self {
            real: (self / rhs).mag_angle().1,
            imag: T::ZERO,
        }
    }
    fn float_ln(self) -> Self {
        self.principal_ln()
    }
}




