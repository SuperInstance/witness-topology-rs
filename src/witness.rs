//! Witness complex construction from point clouds and landmarks.

use crate::landmark::{euclidean_distance, Point};

/// A simplex (vertices are landmark indices).
pub type Simplex = Vec<usize>;

/// A simplicial complex.
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    /// All simplices, organized by dimension (0=simplices, 1=edges, 2=triangles, etc.)
    pub simplices: Vec<Vec<Simplex>>,
}

impl SimplicialComplex {
    /// Create an empty complex.
    pub fn new() -> Self {
        Self {
            simplices: vec![vec![]],
        }
    }

    /// Add a simplex to the complex (with its faces).
    pub fn add_simplex(&mut self, simplex: Simplex) {
        let dim = simplex.len() - 1;
        while self.simplices.len() <= dim {
            self.simplices.push(vec![]);
        }

        // Check if already present
        if !self.simplices[dim].contains(&simplex) {
            // Add all faces recursively
            if simplex.len() > 1 {
                for i in 0..simplex.len() {
                    let mut face = simplex.clone();
                    face.remove(i);
                    self.add_simplex(face);
                }
            }
            self.simplices[dim].push(simplex);
        }
    }

    /// Get vertices (0-simplices).
    pub fn vertices(&self) -> &[Simplex] {
        &self.simplices[0]
    }

    /// Get edges (1-simplices).
    pub fn edges(&self) -> &[Simplex] {
        if self.simplices.len() > 1 {
            &self.simplices[1]
        } else {
            &[]
        }
    }

    /// Get triangles (2-simplices).
    pub fn triangles(&self) -> &[Simplex] {
        if self.simplices.len() > 2 {
            &self.simplices[2]
        } else {
            &[]
        }
    }

    /// Number of simplices of a given dimension.
    pub fn num_simplices(&self, dim: usize) -> usize {
        if dim < self.simplices.len() {
            self.simplices[dim].len()
        } else {
            0
        }
    }

    /// Check if a simplex is in the complex.
    pub fn contains(&self, simplex: &[usize]) -> bool {
        let dim = simplex.len() - 1;
        if dim >= self.simplices.len() {
            return false;
        }
        let mut s = simplex.to_vec();
        s.sort();
        self.simplices[dim].iter().any(|existing| {
            let mut e = existing.clone();
            e.sort();
            e == s
        })
    }

    /// Euler characteristic.
    pub fn euler_characteristic(&self) -> i64 {
        self.simplices
            .iter()
            .enumerate()
            .map(|(dim, simplices)| {
                if dim % 2 == 0 {
                    simplices.len() as i64
                } else {
                    -(simplices.len() as i64)
                }
            })
            .sum()
    }
}

impl Default for SimplicialComplex {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a witness complex from a point cloud and landmark indices.
///
/// For each witness point, find its k nearest landmarks. If two landmarks
/// are both among the k nearest neighbors of some witness, add an edge.
/// Higher-dimensional simplices are added when all pairs are witnessed.
pub fn build_witness_complex(
    points: &[Point],
    landmarks: &[usize],
    k: usize,
) -> SimplicialComplex {
    let mut complex = SimplicialComplex::new();

    // Add all landmark vertices
    for &l in landmarks {
        complex.add_simplex(vec![l]);
    }

    // For each point, find k nearest landmarks
    let k = k.min(landmarks.len());
    let mut witnessed_edges: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    let mut witnessed_triangles: std::collections::HashSet<(usize, usize, usize)> =
        std::collections::HashSet::new();

    for point in points {
        let mut landmark_dists: Vec<(usize, f64)> = landmarks
            .iter()
            .map(|&l| (l, euclidean_distance(point, &points[l])))
            .collect();
        landmark_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let nearest: Vec<usize> = landmark_dists[..k].iter().map(|(l, _)| *l).collect();

        // Add edges between all pairs of k-nearest landmarks
        for i in 0..nearest.len() {
            for j in (i + 1)..nearest.len() {
                let a = nearest[i].min(nearest[j]);
                let b = nearest[i].max(nearest[j]);
                witnessed_edges.insert((a, b));
            }
        }

        // Add triangles
        if k >= 3 {
            for i in 0..nearest.len() {
                for j in (i + 1)..nearest.len() {
                    for l in (j + 1)..nearest.len() {
                        let mut tri = [nearest[i], nearest[j], nearest[l]];
                        tri.sort();
                        witnessed_triangles.insert((tri[0], tri[1], tri[2]));
                    }
                }
            }
        }
    }

    for (a, b) in &witnessed_edges {
        complex.add_simplex(vec![*a, *b]);
    }
    for (a, b, c) in &witnessed_triangles {
        complex.add_simplex(vec![*a, *b, *c]);
    }

    complex
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
    fn test_empty_complex() {
        let c = SimplicialComplex::new();
        assert_eq!(c.num_simplices(0), 0);
        assert_eq!(c.euler_characteristic(), 0);
    }

    #[test]
    fn test_add_vertex() {
        let mut c = SimplicialComplex::new();
        c.add_simplex(vec![0]);
        assert_eq!(c.num_simplices(0), 1);
        assert!(c.contains(&[0]));
    }

    #[test]
    fn test_add_edge_adds_vertices() {
        let mut c = SimplicialComplex::new();
        c.add_simplex(vec![0, 1]);
        assert_eq!(c.num_simplices(0), 2);
        assert_eq!(c.num_simplices(1), 1);
    }

    #[test]
    fn test_add_triangle_adds_faces() {
        let mut c = SimplicialComplex::new();
        c.add_simplex(vec![0, 1, 2]);
        assert_eq!(c.num_simplices(0), 3);
        assert_eq!(c.num_simplices(1), 3);
        assert_eq!(c.num_simplices(2), 1);
        assert_eq!(c.euler_characteristic(), 3 - 3 + 1);
    }

    #[test]
    fn test_witness_complex_basic() {
        let cloud = sample_cloud();
        let landmarks = vec![0, 1, 2, 3];
        let complex = build_witness_complex(&cloud, &landmarks, 2);
        assert_eq!(complex.num_simplices(0), 4);
        assert!(complex.num_simplices(1) > 0);
    }

    #[test]
    fn test_witness_complex_k1_no_edges() {
        let cloud = sample_cloud();
        let landmarks = vec![0, 1, 2];
        let complex = build_witness_complex(&cloud, &landmarks, 1);
        assert_eq!(complex.num_simplices(0), 3);
        assert_eq!(complex.num_simplices(1), 0);
    }

    #[test]
    fn test_witness_complex_collinear() {
        let cloud = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let landmarks = vec![0, 1, 2, 3];
        let complex = build_witness_complex(&cloud, &landmarks, 3);
        assert!(complex.num_simplices(0) > 0);
    }

    #[test]
    fn test_contains() {
        let mut c = SimplicialComplex::new();
        c.add_simplex(vec![2, 1]);
        assert!(c.contains(&[1, 2]));
        assert!(c.contains(&[2, 1]));
        assert!(!c.contains(&[1, 3]));
    }

    #[test]
    fn test_euler_single_triangle() {
        let mut c = SimplicialComplex::new();
        c.add_simplex(vec![0, 1, 2]);
        assert_eq!(c.euler_characteristic(), 1);
    }
}
