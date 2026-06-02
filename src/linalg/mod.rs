


#[derive(Debug)]
pub enum Error {
    ErrorSingularMatrix,
    ErrorSizeInvalid,
}


impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for Error {
}


// performs the plu decomposition inplace
pub fn plu_decompose(
    a: &mut [f64],
    p: &mut [usize],
) -> Result<(), Error> {

    let tol = 1e-16;

    let n = (a.len() as f64).sqrt() as usize;

    if n != (p.len() - 1) {
        return Err(Error::ErrorSizeInvalid);
    }
    if (n * n) != a.len() {
        return Err(Error::ErrorSizeInvalid);
    }

    for i in 0..n {
        p[i] = i;
    }
    p[n] = n;

    for i in 0..n {

        let mut maxa = 0.0;
        let mut imax = i;

        for k in i..n {
            let absa = a[k*n + i].abs();
            if absa > maxa {
                maxa = absa;
                imax = k;
            }
        }

        if maxa < tol {
            return Err(Error::ErrorSingularMatrix);
        }

        if imax != i {
            // permutation
            let j = p[i];
            p[i] = p[imax];
            p[imax] = j;

            // swap rows
            for j in 0..n {
                let atmp = a[i*n + j];
                a[i*n + j] = a[imax*n + j];
                a[imax*n + j] = atmp;
            }

            p[n] += 1;
        }

        for j in (i+1)..n {
            a[j*n + i] /= a[i*n + i];

            for k in (i+1)..n {
                a[j*n + k] -= a[j*n + i] * a[i*n + k];
            }
        }

    }


    Ok(())
}



pub fn plu_solve(x: &mut [f64], b: &[f64], lu: &[f64], p: &[usize]) {

    let n = x.len();

    for i in 0..n {

        x[i] = b[p[i]];

        for k in 0..i {
            x[i] -= lu[i*n + k] * x[k];
        }
    }

    for i in (0..n).rev() {
        for k in (i+1)..n {
            x[i] -= lu[i*n + k] * x[k];
        }
        x[i] /= lu[i*n + i];
    }

}


