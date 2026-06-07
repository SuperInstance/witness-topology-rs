//! Persistence homology on witness complexes.
//!
//! Simplified persistent homology computation for small complexes.

use crate::witness::{Simplex, SimplicialComplex};

/// A persistence pair (birth, death) in homological dimension.
#[derive(Debug, Clone)]
pub struct PersistencePair {
    /// Homological dimension.
    pub dim: usize,
    /// Birth time (filtration value).
    pub birth: f64,
    /// Death time (filtration value), None if the class lives forever.
    pub death: Option<f64>,
}

impl PersistencePair {
    /// Persistence of this feature (lifetime).
    pub fn persistence(&self) -> f64 {
        match self.death {
            Some(d) => d - self.birth,
            None => f64::INFINITY,
        }
    }
}

/// Compute persistence homology using a simple filtration by diameter.
/// The filtration value of a simplex is the maximum distance between any two vertices.
/// Uses the boundary matrix reduction algorithm.
pub fn compute_persistence(
    complex: &SimplicialComplex,
    dist: &[Vec<f64>],
) -> Vec<PersistencePair> {
    let mut all_simplices: Vec<Simplex> = Vec::new();

    for dim_simplices in &complex.simplices {
        for simplex in dim_simplices {
            all_simplices.push(simplex.clone());
        }
    }

    // Sort by filtration value (max pairwise distance), then by dimension
    all_simplices.sort_by(|a, b| {
        let fa = filtration_value(a, dist);
        let fb = filtration_value(b, dist);
        fa.partial_cmp(&fb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.len().cmp(&b.len()))
    });

    let n = all_simplices.len();
    if n == 0 {
        return vec![];
    }

    // Build index mapping
    let mut simplex_index: std::collections::HashMap<Vec<usize>, usize> =
        std::collections::HashMap::new();
    for (i, s) in all_simplices.iter().enumerate() {
        let mut sorted = s.clone();
        sorted.sort();
        simplex_index.insert(sorted, i);
    }

    // Build boundary matrix
    let mut boundary: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, simplex) in all_simplices.iter().enumerate() {
        if simplex.len() <= 1 {
            continue;
        }
        for j in 0..simplex.len() {
            let mut face = simplex.clone();
            face.remove(j);
            face.sort();
            if let Some(&face_idx) = simplex_index.get(&face) {
                boundary[i].push(face_idx);
            }
        }
        boundary[i].sort();
        boundary[i].dedup();
    }

    // Standard reduction algorithm
    let mut low: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        if !boundary[i].is_empty() {
            low[i] = Some(*boundary[i].last().unwrap());
        }
    }

    for i in 0..n {
        while let Some(l) = low[i] {
            // Find earlier column with same low
            let mut found = false;
            for j in 0..i {
                if low[j] == Some(l) {
                    // Add column j to column i (mod 2)
                    let mut merged: Vec<usize> = boundary[i]
                        .iter()
                        .chain(boundary[j].iter())
                        .fold(std::collections::HashMap::new(), |mut acc, &x| {
                            *acc.entry(x).or_insert(0usize) += 1;
                            acc
                        })
                        .into_iter()
                        .filter(|(_, c)| c % 2 == 1)
                        .map(|(x, _)| x)
                        .collect();
                    merged.sort();
                    boundary[i] = merged;
                    low[i] = boundary[i].last().copied();
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
    }

    // Extract persistence pairs
    let mut pairs = Vec::new();
    let mut paired: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for i in 0..n {
        if let Some(l) = low[i] {
            paired.insert(l);
            let birth_dim = all_simplices[l].len() - 1;
            let birth_val = filtration_value(&all_simplices[l], dist);
            let death_val = filtration_value(&all_simplices[i], dist);
            if death_val > birth_val {
                pairs.push(PersistencePair {
                    dim: birth_dim,
                    birth: birth_val,
                    death: Some(death_val),
                });
            }
        }
    }

    // Unpaired simplices are essential classes
    for i in 0..n {
        if !paired.contains(&i) && low[i].is_none() {
            let dim = all_simplices[i].len() - 1;
            let birth_val = filtration_value(&all_simplices[i], dist);
            pairs.push(PersistencePair {
                dim,
                birth: birth_val,
                death: None,
            });
        }
    }

    pairs
}

/// Compute filtration value (maximum pairwise distance in simplex).
fn filtration_value(simplex: &[usize], dist: &[Vec<f64>]) -> f64 {
    if simplex.len() <= 1 {
        return 0.0;
    }
    let mut max_d = 0.0;
    for i in 0..simplex.len() {
        for j in (i + 1)..simplex.len() {
            let d = dist[simplex[i]][simplex[j]];
            if d > max_d {
                max_d = d;
            }
        }
    }
    max_d
}

/// Compute Betti numbers from persistence pairs at a given filtration value.
pub fn betti_numbers(pairs: &[PersistencePair], threshold: f64) -> Vec<usize> {
    let max_dim = pairs.iter().map(|p| p.dim).max().unwrap_or(0);
    let mut betti = vec![0usize; max_dim + 1];

    for pair in pairs {
        let alive = pair.birth <= threshold
            && match pair.death {
                Some(d) => d > threshold,
                None => true,
            };
        if alive {
            betti[pair.dim] += 1;
        }
    }

    betti
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witness::SimplicialComplex;

    #[test]
    fn test_persistence_single_point() {
        let mut c = SimplicialComplex::new();
        c.add_simplex(vec![0]);
        let dist = vec![vec![0.0]];
        let pairs = compute_persistence(&c, &dist);
        assert!(pairs.len() >= 1);
        assert_eq!(pairs[0].dim, 0);
        assert!(pairs[0].death.is_none());
    }

    #[test]
    fn test_persistence_edge() {
        let mut c = SimplicialComplex::new();
        c.add_simplex(vec![0, 1]);
        let dist = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let pairs = compute_persistence(&c, &dist);
        // One connected component persists
        let h0: Vec<_> = pairs.iter().filter(|p| p.dim == 0).collect();
        assert!(h0.len() >= 1);
    }

    #[test]
    fn test_persistence_triangle() {
        let mut c = SimplicialComplex::new();
        c.add_simplex(vec![0, 1, 2]);
        let dist = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0],
        ];
        let pairs = compute_persistence(&c, &dist);
        let h0: Vec<_> = pairs.iter().filter(|p| p.dim == 0).collect();
        assert!(h0.len() >= 1);
    }

    #[test]
    fn test_betti_numbers() {
        let pairs = vec![
            PersistencePair {
                dim: 0,
                birth: 0.0,
                death: None,
            },
            PersistencePair {
                dim: 0,
                birth: 0.0,
                death: Some(1.0),
            },
        ];
        let betti = betti_numbers(&pairs, 0.5);
        assert_eq!(betti[0], 2); // both alive at 0.5
    }

    #[test]
    fn test_persistence_pair_persistence() {
        let p = PersistencePair {
            dim: 0,
            birth: 1.0,
            death: Some(3.0),
        };
        assert!((p.persistence() - 2.0).abs() < 1e-10);

        let p_inf = PersistencePair {
            dim: 0,
            birth: 0.0,
            death: None,
        };
        assert!(p_inf.persistence().is_infinite());
    }

    #[test]
    fn test_empty_complex_persistence() {
        let c = SimplicialComplex::new();
        let dist: Vec<Vec<f64>> = vec![];
        let pairs = compute_persistence(&c, &dist);
        assert!(pairs.is_empty());
    }
}
