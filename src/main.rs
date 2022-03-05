use dfmblas::{IntoMatrix, Matrix};

fn main() {
    let rows = 5;
    let cols = 3;
    let m1: Matrix<f64> = Matrix::random(rows, cols);
    let m2: Matrix<f64> = Matrix::random(rows, cols);

    let m = -(m1.clone() - (m1 + m2));
    let x = m.into_matrix();

    println!("{:?}", x);
}
