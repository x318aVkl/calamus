

// Odes have the form dy/dt = f(t,y)

use crate::{linalg::{matrix::MatrixMut, vector::{DynamicVector, Vector, VectorMut, VectorViewMut}}, num_traits::FloatNumber};


const MAX_SIZE_STATIC_JACOBIAN: usize = 256;


pub trait Ode<T> {
    type Error;
    fn eval_f(&self, f: &mut impl VectorMut<T>, y: &impl Vector<T>, t: T) -> Result<(), Self::Error>;
    fn problem_size(&self) -> usize;


    // evaluate the problem jacobian
    // jac is in row-first ordering
    // [row_1, row_2, row_3 ...]
    // [a_00, a_01, a_02 ... a_10, a_11, a_12 ...]
    // jac[i*size + j] must be derivative of f[i] over variable j
    // default implementation uses a difference quotient automatic formulation
    fn eval_jacobian(&self, jac: &mut impl MatrixMut<T>, y: &impl Vector<T>, t: T) -> Result<(), <Self as Ode<T>>::Error> where T: FloatNumber {
        if y.len() > MAX_SIZE_STATIC_JACOBIAN {
            panic!("Error, problem size must be smaller or equal to {}", MAX_SIZE_STATIC_JACOBIAN);
        }

        // by default, use a difference quotient approach

        let n = self.problem_size();

        // allocate temporary vectors or use static sized arrays
        let mut f_alloc = [T::from(0); MAX_SIZE_STATIC_JACOBIAN];
        let mut fb_alloc = [T::from(0); MAX_SIZE_STATIC_JACOBIAN];
        let mut yd_alloc = [T::from(0); MAX_SIZE_STATIC_JACOBIAN];

        let f_a = &mut f_alloc[0..y.len()];
        let fb_a = &mut fb_alloc[0..y.len()];
        let yd_a = &mut yd_alloc[0..y.len()];

        #[allow(unused_assignments)]
        let mut f_ad = None;
        #[allow(unused_assignments)]
        let mut f_bd = None;
        #[allow(unused_assignments)]
        let mut yd_ad = None;

        let (mut f, mut fb, mut yd) = if n <= MAX_SIZE_STATIC_JACOBIAN {
            let f = VectorViewMut::from(f_a);
            let fb = VectorViewMut::from(fb_a);
            let yd = VectorViewMut::from(yd_a);
            (f, fb, yd)
        } else {
            // use dynamic allocation
            f_ad = Some(DynamicVector::new(T::from(0), n));
            f_bd = Some(DynamicVector::new(T::from(0), n));
            yd_ad = Some(DynamicVector::new(T::from(0), n));


            (
                f_ad.as_mut().unwrap().view_mut(), 
                f_bd.as_mut().unwrap().view_mut(),
                yd_ad.as_mut().unwrap().view_mut(),
            )
        };

        for i in 0..n {
            yd[i] = y[i];
        }

        // base f
        self.eval_f(&mut f, y, t)?;

        let scale = T::EPSILON.float_sqrt();
        
        for i in 0..n {
            let y0 = y[i];
            let dy = scale.float_max(y[i].float_abs() * scale);
            yd[i] = y0 + dy;

            self.eval_f(&mut fb, &yd, t)?;

            for j in 0..n {
                jac[[j, i]] = (fb[j] - f[j]) / dy;
            }

            yd[i] = y0;
        }

        Ok(())
    }
}



