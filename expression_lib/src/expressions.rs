use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::fmt::Write;
use nalgebra::{DMatrix, DVector};
//
// #[polars_expr(output_type=Float64)]
// fn ols(inputs: &[Series]) -> PolarsResult<Series> {
//     print!("{:#?}", inputs);
//     let a = inputs[0].f64()?;
//     let b = inputs[1].f64()?;
//     // print!("a:{:#?}", a);
//     // print!("b:{:#?}", b);
//     Ok(Series::new("Results", &[b.sum(),a.sum()]))
// }

fn series_to_nalgebra_vector(series: &Series) -> Result<DVector<f64>, PolarsError> {
    let n = series.len();

    // let mut data = series.f64()?.into_iter().collect::<Vec<f64>>();
    let data = series
        .f64()?
        .into_iter()
        .map(|opt| opt.unwrap_or(0.0))
        .collect::<Vec<f64>>();


    // Create a nalgebra matrix from the flattened data
    let vector = DVector::from_vec(data);

    Ok(vector)
}
fn series_to_nalgebra_matrix(series_vec: &[Series]) -> Result<DMatrix<f64>, PolarsError> {
    // Determine the number of rows and columns
    let nrows = series_vec[0].len();
    let ncols = series_vec.len();

    // Create a vector to store the flattened data
    let mut data = Vec::with_capacity(nrows * ncols);

    // Iterate over each series and collect the data
    for series in series_vec {
        let series_data = series
                    .f64()?
                    .into_iter()
                    .map(|opt| opt.unwrap_or(0.0))
                    .collect::<Vec<f64>>();
        data.extend(series_data);
    }

    // Create a nalgebra matrix from the flattened data
    let matrix = DMatrix::from_vec(nrows, ncols, data);

    Ok(matrix)
}

#[polars_expr(output_type=Float64)]
fn ols(inputs: &[Series]) -> PolarsResult<Series> {
    print!("{:#?}", inputs);
    // let y = series_to_nalgebra_vector(&inputs[0]);
    // let X = series_to_nalgebra_matrix(&inputs[1..]);
    // // let x = inputs[1..].to_ndarray()?;
    // print!("X:{:#?}", X);
    // // print!("b:{:#?}", b);
    // let X_transpose = X?.clone().transpose();
    // let X_transpose_X = &X_transpose * &X?;
    // let X_transpose_y = X_transpose * y?;
    // // let beta = X_transpose_X.try_inverse().expect("Cannot compute inverse").unwrap() * X_transpose_y;
    //
    // // Output the coefficients
    // println!("Coefficients: {:?}", X_transpose_y);
    // Ok(Series::new("Results", &[0.]))

    let y = series_to_nalgebra_vector(&inputs[0])?;
    let x = series_to_nalgebra_matrix(&inputs[1..])?;

    // Compute the coefficients using least squares method
    let x_transpose = x.clone().transpose();
    let x_transpose_x = &x_transpose * &x;
    let x_transpose_y = x_transpose * y;

    let beta = x_transpose_x.try_inverse()
        .ok_or_else(|| PolarsError::ComputeError("Matrix is singular and cannot be inverted".into()))?
        * x_transpose_y;

    // Convert the coefficients to a Polars Series
    let coefficients = Series::new("coefficients", beta.iter().cloned().collect::<Vec<f64>>());
    println!("Coefficients: {:?}", coefficients);
    // Ok(Series::new("Results", &[0.]))
    Ok(coefficients)

}
/// The `DefaultKwargs` isn't very ergonomic as it doesn't validate any schema.
/// Provide your own kwargs struct with the proper schema and accept that type
/// in your plugin expression.
#[derive(Deserialize)]
pub struct MyKwargs {
    float_arg: f64,
    integer_arg: i64,
    string_arg: String,
    boolean_arg: bool,
}

/// If you want to accept `kwargs`. You define a `kwargs` argument
/// on the second position in you plugin. You can provide any custom struct that is deserializable
/// with the pickle protocol (on the rust side).
#[polars_expr(output_type=Utf8)]
fn append_kwargs(input: &[Series], kwargs: MyKwargs) -> PolarsResult<Series> {
    let input = &input[0];
    let input = input.cast(&DataType::Utf8)?;
    let ca = input.utf8().unwrap();

    Ok(ca
        .apply_to_buffer(|val, buf| {
            write!(
                buf,
                "{}-{}-{}-{}-{}",
                val, kwargs.float_arg, kwargs.integer_arg, kwargs.string_arg, kwargs.boolean_arg
            )
            .unwrap()
        })
        .into_series())
}
