pub mod cwise_ops;
mod matrix;
pub use crate::matrix::{IntoMatrix, Matrix};

pub type Index = i64;

pub trait MatrixShape: Sized + IntoIterator<Item = Self::Scalar> {
    type Scalar;
    fn rows(&self) -> Index;
    fn cols(&self) -> Index;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let rows = 5;
        let cols = 3;
        let capacity = (rows * cols) as usize;
        let mut data1 = Vec::with_capacity(capacity);
        data1.resize(capacity, 3.0f64);
        let m1 = Matrix::from_slice(rows, cols, &data1);

        let mut data2 = Vec::with_capacity(capacity);
        data2.resize(capacity, 1.0f64);
        let m2 = Matrix::from_slice(rows, cols, &data2);

        let m = -(m1 + m2);
        println!("{:?}", m.into_matrix());
    }
}
