

pub struct Sparsity {
    minors: Vec<usize>,
    major_starts: Vec<usize>,
    max_minor: usize,
}



impl Sparsity {

    pub fn new() -> Sparsity {
        Self { minors: vec![], major_starts: vec![0], max_minor: 0, }
    }

    pub fn major_len(&self) -> usize {
        self.major_starts.len() - 1
    }

    pub fn minor_len(&self) -> usize {
        self.minors.len()
    }

    pub fn max_minor(&self) -> usize {
        self.max_minor
    }

    pub fn major_start(&self, major: usize) -> usize {
        self.major_starts[major]
    }


    pub fn push_to_major(&mut self, minor: usize) {
        self.minors.push(minor);
        self.max_minor = self.max_minor.max(minor);
    }

    pub fn close_major(&mut self) {
        self.major_starts.push(self.minors.len())
    }

    pub fn close_major_and_sort<T: Copy>(&mut self, values: &mut [T], tmp_buffer: &mut Vec<(usize, T)>) where T: From<u8> + core::fmt::Debug {

        let a = self.major_starts[self.major_starts.len() - 1];
        let b = self.minors.len();

        tmp_buffer.resize(b - a, (0, T::from(0)));

        for i in 0..(b - a) {
            tmp_buffer[i].0 = self.minors[a + i];
            tmp_buffer[i].1 = values[a + i];
        }
        tmp_buffer.sort_by(|a, b| a.0.cmp(&b.0));

        for i in 0..tmp_buffer.len() {
            self.minors[a + i] = tmp_buffer[i].0;
            values[a + i] = tmp_buffer[i].1;
        }

        self.major_starts.push(self.minors.len())
    }

    pub fn major_range(&self, major: usize) -> &[usize] {
        &self.minors[self.major_starts[major]..self.major_starts[major + 1]]
    }

    pub fn major_range_flat(&self, major: usize) -> std::ops::Range<usize> {
        self.major_starts[major]..self.major_starts[major + 1]
    }

    pub fn flat_index(&self, k: usize) -> usize {
        self.minors[k]
    }

}







