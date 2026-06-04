

pub mod gamma;
pub mod bessel;
pub mod amg;



pub fn factorial(x: u64) -> u64 {
    let sols = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800];

    if x < 10 {
        return sols[x as usize];
    }

    let mut r = sols[x as usize];
    for i in 10..x {
        r *= i;
    }
    r
}
