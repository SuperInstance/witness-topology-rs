//! Multi-scale analysis with varying witness threshold.

use crate::landmark::{distance_matrix, maxmin_landmarks, random_landmarks, Point};
use crate::persistence::{betti_numbers, compute_persistence, PersistencePair};
use crate::witness::build_witness_complex;
use crate::weak_witness::build_weak_witness_complex;

/// Result of multi-scale analysis at a particular scale.
#[derive(Debug, Clone)]
pub struct ScaleResult {
    /// The k value used.
    pub k: usize,
    /// Number of vertices.
    pub num_vertices: usize,
    /// Number of edges.
    pub num_edges: usize,
    /// Number of triangles.
    pub num_triangles: usize,
    /// Euler characteristic.
    pub euler: i64,
    /// Betti numbers at this scale.
    pub betti: Vec<usize>,
}

/// Perform multi-scale analysis by varying the witness threshold (k parameter).
pub fn multiscale_analysis(
    points: &[Point],
    num_landmarks: usize,
    k_values: &[usize],
    seed: u64,
) -> Vec<ScaleResult> {
    let landmarks = random_landmarks(points, num_landmarks, seed);
    let dist = distance_matrix(points);

    k_values
        .iter()
        .map(|&k| {
            let complex = build_witness_complex(points, &landmarks, k);
            let pairs = compute_persistence(&complex, &dist);
            let betti = betti_numbers(&pairs, f64::INFINITY);

            ScaleResult {
                k,
                num_vertices: complex.num_simplices(0),
                num_edges: complex.num_simplices(1),
                num_triangles: complex.num_simplices(2),
                euler: complex.euler_characteristic(),
                betti,
            }
        })
        .collect()
}

/// Compare witness complex vs weak witness complex across scales.
pub fn compare_variants(
    points: &[Point],
    num_landmarks: usize,
    k_values: &[usize],
    seed: u64,
) -> (Vec<ScaleResult>, Vec<ScaleResult>) {
    let landmarks_random = random_landmarks(points, num_landmarks, seed);
    let landmarks_maxmin = maxmin_landmarks(points, num_landmarks);
    let dist = distance_matrix(points);

    let witness_results: Vec<ScaleResult> = k_values
        .iter()
        .map(|&k| {
            let complex = build_witness_complex(points, &landmarks_random, k);
            let pairs = compute_persistence(&complex, &dist);
            let betti = betti_numbers(&pairs, f64::INFINITY);
            ScaleResult {
                k,
                num_vertices: complex.num_simplices(0),
                num_edges: complex.num_simplices(1),
                num_triangles: complex.num_simplices(2),
                euler: complex.euler_characteristic(),
                betti,
            }
        })
        .collect();

    let weak_results: Vec<ScaleResult> = k_values
        .iter()
        .map(|&k| {
            let complex = build_weak_witness_complex(points, &landmarks_maxmin, 2);
            let pairs = compute_persistence(&complex, &dist);
            let betti = betti_numbers(&pairs, f64::INFINITY);
            ScaleResult {
                k,
                num_vertices: complex.num_simplices(0),
                num_edges: complex.num_simplices(1),
                num_triangles: complex.num_simplices(2),
                euler: complex.euler_characteristic(),
                betti,
            }
        })
        .collect();

    (witness_results, weak_results)
}

/// Find the stability range: the range of k values where Betti numbers don't change.
pub fn stability_range(results: &[ScaleResult]) -> Option<(usize, usize)> {
    if results.is_empty() {
        return None;
    }

    let reference_betti = &results[results.len() / 2].betti;

    let first_stable = results
        .iter()
        .position(|r| r.betti == *reference_betti)?;
    let last_stable = results
        .iter()
        .rposition(|r| r.betti == *reference_betti)?;

    Some((results[first_stable].k, results[last_stable].k))
}

/// Compute the barcode (all persistence pairs) for a given scale.
pub fn barcode_at_scale(
    points: &[Point],
    num_landmarks: usize,
    k: usize,
    seed: u64,
) -> Vec<PersistencePair> {
    let landmarks = random_landmarks(points, num_landmarks, seed);
    let dist = distance_matrix(points);
    let complex = build_witness_complex(points, &landmarks, k);
    compute_persistence(&complex, &dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cloud() -> Vec<Point> {
        // Create a simple circular-ish point cloud
        let mut points = Vec::new();
        for i in 0..8 {
            let angle = (i as f64) * std::f64::consts::PI * 2.0 / 8.0;
            points.push(vec![angle.cos(), angle.sin()]);
        }
        // Add some interior points
        points.push(vec![0.0, 0.0]);
        points.push(vec![0.5, 0.0]);
        points
    }

    #[test]
    fn test_multiscale_analysis() {
        let cloud = sample_cloud();
        let k_values = vec![2, 3, 4];
        let results = multiscale_analysis(&cloud, 5, &k_values, 42);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.num_vertices > 0);
        }
    }

    #[test]
    fn test_multiscale_growing_edges() {
        let cloud = sample_cloud();
        let k_values = vec![2, 5];
        let results = multiscale_analysis(&cloud, 5, &k_values, 42);
        // More neighbors → more edges (generally)
        assert!(results[1].num_edges >= results[0].num_edges);
    }

    #[test]
    fn test_compare_variants() {
        let cloud = sample_cloud();
        let k_values = vec![3];
        let (witness, weak) = compare_variants(&cloud, 4, &k_values, 42);
        assert_eq!(witness.len(), 1);
        assert_eq!(weak.len(), 1);
    }

    #[test]
    fn test_stability_range() {
        let cloud = sample_cloud();
        let k_values = vec![2, 3, 4, 5, 6];
        let results = multiscale_analysis(&cloud, 5, &k_values, 42);
        let range = stability_range(&results);
        // Should find some stable range
        assert!(range.is_some());
    }

    #[test]
    fn test_stability_range_empty() {
        let range = stability_range(&[]);
        assert!(range.is_none());
    }

    #[test]
    fn test_barcode_at_scale() {
        let cloud = sample_cloud();
        let pairs = barcode_at_scale(&cloud, 5, 3, 42);
        // Should have at least H0 classes
        assert!(!pairs.is_empty());
    }
}
