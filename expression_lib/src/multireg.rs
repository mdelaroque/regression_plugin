use polars::datatypes::PlHashSet;
use polars::export::arrow::array::PrimitiveArray;
use polars::export::num::Float;
use polars::prelude::*;
use pyo3_polars::export::polars_core::utils::arrow::types::NativeType;
use pyo3_polars::export::polars_core::with_match_physical_integer_type;
use std::hash::Hash;

#[allow(clippy::all)]
pub(super) fn ols(a: &Float64Chunked, b: &Float64Chunked) -> Option<f64> {
    a.sum()
}
