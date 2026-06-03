



/// An immutable view window that acts like a vector
#[derive(Debug, Clone, Copy)]
pub struct VectorView<'a, T> {
    data: &'a [T],
}

/// A mutable view window that acts like a vector
#[derive(Debug)]
pub struct VectorViewMut<'a, T> {
    data: &'a mut [T],
}


/// a vector with owned data, of runtime size
#[derive(Debug, Clone)]
pub struct DynamicVector<T> {
    data: Vec<T>,
}

/// a vector with owned data, of compile-time size
#[derive(Debug, Clone, Copy)]
pub struct StaticVector<T, const N: usize> {
    data: [T; N],
}


/// Trait that defines common vector methods
pub trait Vector<T>: core::ops::Index<usize, Output = T> {
    fn len(&self) -> usize;

    fn data(&self) -> &[T];

    fn view<'a>(&'a self) -> VectorView<'a, T> {
        VectorView { data: self.data() }
    }
}

/// Trait that defines common mutable vector methods
pub trait VectorMut<T>: Vector<T> + core::ops::IndexMut<usize, Output = T> {
    fn data_mut(&mut self) -> &mut [T];

    fn view_mut<'a>(&'a mut self) -> VectorViewMut<'a, T> {
        VectorViewMut { data: self.data_mut() }
    }

    fn scope<'a>(&'a mut self, op: impl Fn(&'a mut Self) -> ()) {
        op(self)
    } 
}



impl<T> core::ops::Index<usize> for DynamicVector<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<T> core::ops::IndexMut<usize> for DynamicVector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T> Vector<T> for DynamicVector<T> {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn data(&self) -> &[T] {
        &self.data
    }
}
impl<T> VectorMut<T> for DynamicVector<T> {
    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<'a, T> Into<VectorView<'a, T>> for &'a DynamicVector<T> {
    fn into(self) -> VectorView<'a, T> {
        VectorView { data: &self.data }
    }
}
impl<'a, T, const N: usize> Into<VectorView<'a, T>> for &'a StaticVector<T, N> {
    fn into(self) -> VectorView<'a, T> {
        VectorView { data: &self.data }
    }
}



impl<'a, T> Into<VectorViewMut<'a, T>> for &'a mut DynamicVector<T> {
    fn into(self) -> VectorViewMut<'a, T> {
        VectorViewMut { data: &mut self.data }
    }
}
impl<'a, T, const N: usize> Into<VectorViewMut<'a, T>> for &'a mut StaticVector<T, N> {
    fn into(self) -> VectorViewMut<'a, T> {
        VectorViewMut { data: &mut self.data }
    }
}



impl<'a, T> core::ops::Index<usize> for VectorView<'a, T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, T> Vector<T> for VectorView<'a, T> {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn data(&self) -> &[T] {
        &self.data
    }
}

impl<'a, T> core::ops::Index<usize> for VectorViewMut<'a, T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<'a, T> core::ops::IndexMut<usize> for VectorViewMut<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<'a, T> Vector<T> for VectorViewMut<'a, T> {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn data(&self) -> &[T] {
        &self.data
    }
}
impl<'a, T> VectorMut<T> for VectorViewMut<'a, T> {
    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}



impl<'a, T, Rhs> std::ops::AddAssign<Rhs> for VectorViewMut<'a, T> where Rhs: Vector<T> + Copy, T: core::ops::AddAssign<T> + Copy {
    fn add_assign(&mut self, rhs: Rhs) {
        for i in 0..self.data.len() {
            self[i] += rhs[i];
        }
    }
}


impl<'a, T, Rhs> std::ops::SubAssign<Rhs> for VectorViewMut<'a, T> where Rhs: Vector<T> + Copy, T: core::ops::SubAssign<T> + Copy {
    fn sub_assign(&mut self, rhs: Rhs) {
        for i in 0..self.data.len() {
            self[i] -= rhs[i];
        }
    }
}




impl<T> DynamicVector<T> {


    pub fn new(value: T, size: usize) -> Self where T: Copy {
        Self { data: vec![value; size] }
    }

}



impl<'a, T> From<&'a mut [T]> for VectorViewMut<'a, T> {
    fn from(value: &'a mut [T]) -> Self {
        Self { data: value }
    }
}



impl<T> Vector<T> for [T] {
    fn len(&self) -> usize {
        self.len()
    }
    fn data(&self) -> &[T] {
        &self
    }
}

impl<T, const N: usize> Vector<T> for [T; N] {
    fn len(&self) -> usize {
        N
    }
    fn data(&self) -> &[T] {
        &self[0..N]
    }
}

impl<'a, T> Into<VectorView<'a, T>> for &'a [T] {
    fn into(self) -> VectorView<'a, T> {
        VectorView { data: self }
    }
}


