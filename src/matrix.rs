use crate::{Index, IntoMatrix, MatrixShape};
use dfmblas_macro::{CwiseAdd, CwiseMul, CwiseNeg, CwiseSub};
use num_traits::Zero;

#[derive(Debug, Clone, CwiseAdd, CwiseMul, CwiseSub, CwiseNeg)]
pub struct Matrix<T> {
    pub(crate) rows: Index,
    pub(crate) cols: Index,
    pub(crate) data: Vec<T>,
}

impl<T: Clone> Matrix<T> {
    pub fn new(rows: Index, cols: Index) -> Self
    where
        T: Zero,
    {
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

    #[cfg(feature = "random")]
    pub fn random(rows: Index, cols: Index) -> Self
    where
        rand::distributions::Standard: rand::distributions::Distribution<T>,
    {
        use rand::Rng;
        if rows * cols < 0 {
            panic!("Invalid dimensions");
        }
        let capacity = (rows * cols) as usize;
        let mut data = Vec::with_capacity(capacity);
        let mut rng = rand::thread_rng();
        data.resize_with(capacity, || rng.gen::<T>());
        Self { rows, cols, data }
    }
}

impl<T> MatrixShape for Matrix<T> {
    type Scalar = T;

    #[inline]
    fn rows(&self) -> Index {
        self.rows
    }

    #[inline]
    fn cols(&self) -> Index {
        self.cols
    }
}

impl<T> IntoMatrix<T> for Matrix<T> {
    #[inline]
    fn into_matrix(self) -> Self {
        self
    }
}

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

// impl<T, B> std::ops::Add<B> for Matrix<T>
// where
//     B: MatrixShape<Scalar = <Self as MatrixShape>::Scalar>,
//     T: std::ops::Add<T, Output = T>,
// {
//     type Output = cwise_ops::CwiseBinaryOp<cwise_ops::Add<T>, Self, B>;
//     fn add(self, other: B) -> Self::Output {
//         cwise_ops::CwiseBinaryOp::new(self, other)
//     }
// }
