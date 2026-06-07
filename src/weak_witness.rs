//! Weak witness complex variant.
//!
//! A weak witness for a simplex σ is a point p such that p is closer to
//! all vertices of σ than to any landmark not in σ.

use crate::landmark::{euclidean_distance, Point};
use crate::witness::SimplicialComplex;

/// Build a weak witness complex.
///
/// For each candidate simplex (up to max_dim dimensions), check if there exists
/// a witness point that is closer to all vertices of the simplex than to any
/// non-vertex landmark.
pub fn build_weak_witness_complex(
    points: &[Point],
    landmarks: &[usize],
    max_dim: usize,
) -> SimplicialComplex {
    let mut complex = SimplicialComplex::new();

    // Add all landmark vertices
    for &l in landmarks {
        complex.add_simplex(vec![l]);
    }

    // Build edges
    for i in 0..landmarks.len() {
        for j in (i + 1)..landmarks.len() {
            let li = landmarks[i];
            let lj = landmarks[j];
            if has_weak_witness(points, landmarks, &[li, lj]) {
                complex.add_simplex(vec![li, lj]);
            }
        }
    }

    // Build triangles if max_dim >= 2
    if max_dim >= 2 {
        for i in 0..landmarks.len() {
            for j in (i + 1)..landmarks.len() {
                for k in (j + 1)..landmarks.len() {
                    let li = landmarks[i];
                    let lj = landmarks[j];
                    let lk = landmarks[k];
                    if complex.contains(&[li, lj])
                        && complex.contains(&[li, lk])
                        && complex.contains(&[lj, lk])
                        && has_weak_witness(points, landmarks, &[li, lj, lk])
                    {
                        complex.add_simplex(vec![li, lj, lk]);
                    }
                }
            }
        }
    }

    complex
}

/// Check if a simplex has a weak witness.
fn has_weak_witness(points: &[Point], landmarks: &[usize], simplex: &[usize]) -> bool {
    points.iter().any(|p| {
        let max_vertex_dist = simplex
            .iter()
            .map(|&v| euclidean_distance(p, &points[v]))
            .fold(f64::NEG_INFINITY, f64::max);

        let min_non_vertex_dist = landmarks
            .iter()
            .filter(|l| !simplex.contains(l))
            .map(|&l| euclidean_distance(p, &points[l]))
            .fold(f64::INFINITY, f64::min);

        max_vertex_dist < min_non_vertex_dist
    })
}

/// Build a weak witness complex with a relaxation parameter.
/// Instead of requiring strict inequality, allow a margin.
pub fn build_relaxed_weak_witness_complex(
    points: &[Point],
    landmarks: &[usize],
    max_dim: usize,
    margin: f64,
) -> SimplicialComplex {
    let mut complex = SimplicialComplex::new();

    for &l in landmarks {
        complex.add_simplex(vec![l]);
    }

    for i in 0..landmarks.len() {
        for j in (i + 1)..landmarks.len() {
            let li = landmarks[i];
            let lj = landmarks[j];
            if has_relaxed_weak_witness(points, landmarks, &[li, lj], margin) {
                complex.add_simplex(vec![li, lj]);
            }
        }
    }

    if max_dim >= 2 {
        for i in 0..landmarks.len() {
            for j in (i + 1)..landmarks.len() {
                for k in (j + 1)..landmarks.len() {
                    let li = landmarks[i];
                    let lj = landmarks[j];
                    let lk = landmarks[k];
                    if complex.contains(&[li, lj])
                        && complex.contains(&[li, lk])
                        && complex.contains(&[lj, lk])
                        && has_relaxed_weak_witness(points, landmarks, &[li, lj, lk], margin)
                    {
                        complex.add_simplex(vec![li, lj, lk]);
                    }
                }
            }
        }
    }

    complex
}

fn has_relaxed_weak_witness(
    points: &[Point],
    landmarks: &[usize],
    simplex: &[usize],
    margin: f64,
) -> bool {
    points.iter().any(|p| {
        let max_vertex_dist = simplex
            .iter()
            .map(|&v| euclidean_distance(p, &points[v]))
            .fold(f64::NEG_INFINITY, f64::max);

        let min_non_vertex_dist = landmarks
            .iter()
            .filter(|l| !simplex.contains(l))
            .map(|&l| euclidean_distance(p, &points[l]))
            .fold(f64::INFINITY, f64::min);

        max_vertex_dist < min_non_vertex_dist + margin
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cloud() -> Vec<Point> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ]
    }

    #[test]
    fn test_weak_witness_basic() {
        let cloud = sample_cloud();
        let landmarks = vec![0, 1, 2, 3];
        let complex = build_weak_witness_complex(&cloud, &landmarks, 2);
        assert_eq!(complex.num_simplices(0), 4);
    }

    #[test]
    fn test_weak_witness_three_points() {
        let cloud = vec![vec![0.0], vec![1.0], vec![2.0], vec![0.5], vec![1.5]];
        let landmarks = vec![0, 1, 2];
        let complex = build_weak_witness_complex(&cloud, &landmarks, 2);
        assert_eq!(complex.num_simplices(0), 3);
    }

    #[test]
    fn test_relaxed_weak_witness() {
        let cloud = sample_cloud();
        let landmarks = vec![0, 1, 2, 3];
        let strict = build_weak_witness_complex(&cloud, &landmarks, 2);
        let relaxed = build_relaxed_weak_witness_complex(&cloud, &landmarks, 2, 1.0);
        // Relaxed should have at least as many simplices
        assert!(relaxed.num_simplices(1) >= strict.num_simplices(1));
    }

    #[test]
    fn test_weak_witness_two_points() {
        let cloud = vec![vec![0.0], vec![1.0], vec![0.5]];
        let landmarks = vec![0, 1];
        let complex = build_weak_witness_complex(&cloud, &landmarks, 1);
        assert_eq!(complex.num_simplices(0), 2);
    }

    #[test]
    fn test_weak_witness_single_landmark() {
        let cloud = sample_cloud();
        let landmarks = vec![0];
        let complex = build_weak_witness_complex(&cloud, &landmarks, 2);
        assert_eq!(complex.num_simplices(0), 1);
        assert_eq!(complex.num_simplices(1), 0);
    }
}
