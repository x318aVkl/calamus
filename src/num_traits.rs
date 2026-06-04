
// Number (possibly complex), internally represented by a/multiple floating point number(s)
pub trait FloatNumber: 
core::cmp::PartialEq 
+ core::cmp::PartialOrd 
+ core::ops::Add<Self, Output = Self>
+ core::ops::Sub<Self, Output = Self>
+ core::ops::Mul<Self, Output = Self>
+ core::ops::Div<Self, Output = Self>
+ core::ops::AddAssign<Self>
+ core::ops::SubAssign<Self>
+ core::ops::MulAssign<Self>
+ core::ops::DivAssign<Self>
+ core::ops::Neg<Output = Self>
+ From<u8>
+ From<i32>
+ Copy
+ std::fmt::Debug
+ std::fmt::Display
{

    const EPSILON: Self;
    
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const HALF: Self;

    const PI: Self;

    fn float_abs(self) -> Self;

    fn float_sqrt(self) -> Self;

    fn float_exp(self) -> Self;

    fn float_max(self, rhs: Self) -> Self;

    fn float_min(self, rhs: Self) -> Self;

    fn fraction(a: i32, b: i32) -> Self;

    fn float_powf(self, exponent: Self) -> Self;
    fn float_powi(self, exponent: i32) -> Self;

    fn float_from_f64(x: f64) -> Self;

    fn float_sin(self) -> Self;
    fn float_cos(self) -> Self;
    fn float_atan2(self, rhs: Self) -> Self;

    fn float_ln(self) -> Self;
}


impl FloatNumber for f64 {
    const EPSILON: Self = 1e-16;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;

    const PI: Self = core::f64::consts::PI;

    fn float_abs(self) -> Self {
        self.abs()
    }

    fn float_sqrt(self) -> Self {
        self.sqrt()
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
        (a as Self) / (b as Self)
    }

    fn float_powf(self, exponent: Self) -> Self {
        self.powf(exponent)
    }
    fn float_powi(self, exponent: i32) -> Self {
        self.powi(exponent)
    }

    fn float_from_f64(x: f64) -> Self {
        x
    }

    fn float_sin(self) -> Self {
        self.sin()
    }
    fn float_cos(self) -> Self {
        self.cos()
    }
    fn float_atan2(self, rhs: Self) -> Self {
        self.atan2(rhs)
    }

    fn float_ln(self) -> Self {
        self.ln()
    }
}
