pub mod cwise_ops;
mod matrix;
pub use crate::matrix::Matrix;

pub type Index = i64;

pub trait MatrixShape: Sized + IntoIterator<Item = Self::Scalar> {
    type Scalar;
    fn rows(&self) -> Index;
    fn cols(&self) -> Index;
}

pub trait IntoMatrix<T>: MatrixShape<Scalar = T> {
    fn into_matrix(self) -> Matrix<T> {
        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            data: self.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let rows = 5;
        let cols = 3;
        let m1: Matrix<f64> = Matrix::random(rows, cols);
        let m2: Matrix<f64> = Matrix::random(rows, cols);

        let m = -(m1 + m2);
        println!("{:?}", m.into_matrix());
    }
}
