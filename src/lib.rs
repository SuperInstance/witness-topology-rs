//! # witness-topology-rs
//!
//! Witness complex for topology approximation from point clouds.
//!
//! Provides landmark selection (random, maxmin), witness complex construction,
//! weak witness variant, persistence homology on witness complexes, and
//! multi-scale analysis with varying witness threshold.

pub mod landmark;
pub mod persistence;
pub mod scale;
pub mod weak_witness;
pub mod witness;
