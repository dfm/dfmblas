use crate::{cwise_ops, Index, MatrixShape};
use num_traits::Zero;
use std::marker::PhantomData;

pub trait IntoMatrix<T>: MatrixShape<Scalar = T> {
    fn into_matrix(self) -> Matrix<T> {
        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            data: self.into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    rows: Index,
    cols: Index,
    data: Vec<T>,
}

impl<T: Clone + Zero> Matrix<T> {
    pub fn new(rows: Index, cols: Index) -> Self {
        if rows * cols < 0 {
            panic!("Invalid dimensions");
        }
        let capacity = (rows * cols) as usize;
        let mut data = Vec::with_capacity(capacity);
        data.resize(capacity, T::zero());
        Self { rows, cols, data }
    }

    pub fn from_slice(rows: Index, cols: Index, slice: &[T]) -> Self {
        if rows * cols != slice.len() as Index {
            panic!("Invalid dimensions");
        }
        Self {
            rows,
            cols,
            data: slice.to_vec(),
        }
    }
}

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<T> MatrixShape for Matrix<T> {
    type Scalar = T;

    fn rows(&self) -> Index {
        self.rows
    }
    fn cols(&self) -> Index {
        self.cols
    }
}

impl<T> IntoMatrix<T> for Matrix<T> {
    fn into_matrix(self) -> Self {
        self
    }
}

impl<T, B> std::ops::Add<B> for Matrix<T>
where
    B: MatrixShape<Scalar = <Self as MatrixShape>::Scalar>,
    T: std::ops::Add<T, Output = T>,
{
    type Output = cwise_ops::CwiseBinaryOp<cwise_ops::Add<T>, Self, B>;
    fn add(self, other: B) -> Self::Output {
        cwise_ops::CwiseBinaryOp {
            a: self,
            b: other,
            op: PhantomData,
        }
    }
}
