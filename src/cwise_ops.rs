use crate::{Index, IntoMatrix, MatrixShape};
use std::marker::PhantomData;

pub trait UnaryOp<T> {
    fn call(a: T) -> T;
}

pub struct Neg<Scalar>(PhantomData<Scalar>);
impl<Scalar> UnaryOp<Scalar> for Neg<Scalar>
where
    Scalar: std::ops::Neg<Output = Scalar>,
{
    fn call(a: Scalar) -> Scalar {
        -a
    }
}

pub trait BinaryOp<Scalar> {
    fn call(a: Scalar, b: Scalar) -> Scalar;
}

pub struct Add<Scalar>(PhantomData<Scalar>);
impl<Scalar> BinaryOp<Scalar> for Add<Scalar>
where
    Scalar: std::ops::Add<Output = Scalar>,
{
    fn call(a: Scalar, b: Scalar) -> Scalar {
        a + b
    }
}

pub struct Sub<Scalar>(PhantomData<Scalar>);
impl<Scalar> BinaryOp<Scalar> for Sub<Scalar>
where
    Scalar: std::ops::Sub<Output = Scalar>,
{
    fn call(a: Scalar, b: Scalar) -> Scalar {
        a - b
    }
}

pub struct Mul<Scalar>(PhantomData<Scalar>);
impl<Scalar> BinaryOp<Scalar> for Mul<Scalar>
where
    Scalar: std::ops::Mul<Output = Scalar>,
{
    fn call(a: Scalar, b: Scalar) -> Scalar {
        a * b
    }
}

pub struct CwiseUnaryOp<F, A>
where
    F: UnaryOp<<A as MatrixShape>::Scalar>,
    A: MatrixShape,
{
    a: A,
    op: PhantomData<F>,
}

impl<F, A> CwiseUnaryOp<F, A>
where
    F: UnaryOp<<A as MatrixShape>::Scalar>,
    A: MatrixShape,
{
    pub fn new(a: A) -> Self {
        Self { a, op: PhantomData }
    }
}

impl<F, A> IntoMatrix<<Self as MatrixShape>::Scalar> for CwiseUnaryOp<F, A>
where
    F: UnaryOp<<A as MatrixShape>::Scalar>,
    A: MatrixShape,
{
}

impl<F, A> IntoIterator for CwiseUnaryOp<F, A>
where
    F: UnaryOp<<A as MatrixShape>::Scalar>,
    A: MatrixShape,
{
    type Item = <A as MatrixShape>::Scalar;
    type IntoIter = std::iter::Map<<A as IntoIterator>::IntoIter, fn(Self::Item) -> Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.a.into_iter().map(|a| F::call(a))
    }
}

impl<F, A> MatrixShape for CwiseUnaryOp<F, A>
where
    F: UnaryOp<<A as MatrixShape>::Scalar>,
    A: MatrixShape,
{
    type Scalar = <A as MatrixShape>::Scalar;

    fn rows(&self) -> Index {
        self.a.rows()
    }

    fn cols(&self) -> Index {
        self.a.cols()
    }
}

pub struct CwiseBinaryOp<F, A, B>
where
    F: BinaryOp<<A as MatrixShape>::Scalar>,
    A: MatrixShape,
    B: MatrixShape<Scalar = <A as MatrixShape>::Scalar>,
{
    pub(crate) a: A,
    pub(crate) b: B,
    pub(crate) op: PhantomData<F>,
}

impl<F, A, B> CwiseBinaryOp<F, A, B>
where
    F: BinaryOp<<A as MatrixShape>::Scalar>,
    A: MatrixShape,
    B: MatrixShape<Scalar = <A as MatrixShape>::Scalar>,
{
    pub fn new(a: A, b: B) -> Self {
        Self {
            a,
            b,
            op: PhantomData,
        }
    }
}

impl<F, A, B> IntoMatrix<<Self as MatrixShape>::Scalar> for CwiseBinaryOp<F, A, B>
where
    F: BinaryOp<<A as MatrixShape>::Scalar>,
    A: MatrixShape,
    B: MatrixShape<Scalar = <A as MatrixShape>::Scalar>,
{
}

impl<F, A, B> MatrixShape for CwiseBinaryOp<F, A, B>
where
    F: BinaryOp<<A as MatrixShape>::Scalar>,
    A: MatrixShape,
    B: MatrixShape<Scalar = <A as MatrixShape>::Scalar>,
{
    type Scalar = <A as MatrixShape>::Scalar;

    fn rows(&self) -> Index {
        if self.a.rows() != self.b.rows() {
            panic!("Dimension mismatch");
        }
        self.a.rows()
    }

    fn cols(&self) -> Index {
        if self.a.cols() != self.b.cols() {
            panic!("Dimension mismatch");
        }
        self.a.cols()
    }
}

impl<F, A, B> IntoIterator for CwiseBinaryOp<F, A, B>
where
    F: BinaryOp<<A as MatrixShape>::Scalar>,
    A: MatrixShape,
    B: MatrixShape<Scalar = <A as MatrixShape>::Scalar>,
{
    type Item = <A as MatrixShape>::Scalar;
    type IntoIter = std::iter::Map<
        std::iter::Zip<<A as IntoIterator>::IntoIter, <B as IntoIterator>::IntoIter>,
        fn((Self::Item, Self::Item)) -> Self::Item,
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.a.into_iter().zip(self.b).map(|(a, b)| F::call(a, b))
    }
}

impl<F, A, B> std::ops::Neg for CwiseBinaryOp<F, A, B>
where
    F: BinaryOp<<A as MatrixShape>::Scalar>,
    A: MatrixShape,
    B: MatrixShape<Scalar = <A as MatrixShape>::Scalar>,
    <Self as MatrixShape>::Scalar: std::ops::Neg<Output = <Self as MatrixShape>::Scalar>,
{
    type Output = CwiseUnaryOp<Neg<<Self as MatrixShape>::Scalar>, Self>;
    fn neg(self) -> <Self as std::ops::Neg>::Output {
        CwiseUnaryOp {
            a: self,
            op: PhantomData,
        }
    }
}
