//! Landmark selection strategies for witness complexes.

/// A point in the cloud, represented as a vector of coordinates.
pub type Point = Vec<f64>;

/// Select landmark points randomly from the point cloud.
/// Returns indices of selected landmarks.
pub fn random_landmarks(points: &[Point], num_landmarks: usize, seed: u64) -> Vec<usize> {
    let n = points.len();
    if num_landmarks >= n {
        return (0..n).collect();
    }

    // Simple LCG PRNG
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng
    };

    // Fisher-Yates partial shuffle
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..num_landmarks {
        let rand_val = next();
        let j = i + (rand_val as usize % (n - i));
        indices.swap(i, j);
    }
    indices[..num_landmarks].to_vec()
}

/// Select landmark points using the maxmin strategy.
/// Iteratively selects the point that maximizes the minimum distance to all
/// previously selected landmarks. This gives good spatial coverage.
pub fn maxmin_landmarks(points: &[Point], num_landmarks: usize) -> Vec<usize> {
    let n = points.len();
    if num_landmarks >= n {
        return (0..n).collect();
    }
    if n == 0 || num_landmarks == 0 {
        return vec![];
    }

    let mut landmarks = Vec::with_capacity(num_landmarks);
    let mut min_distances = vec![f64::INFINITY; n];

    // Start with the first point
    landmarks.push(0);
    for i in 0..n {
        min_distances[i] = euclidean_distance(&points[0], &points[i]);
    }

    while landmarks.len() < num_landmarks {
        // Find point with maximum min-distance to existing landmarks
        let (best_idx, _) = min_distances
            .iter()
            .enumerate()
            .filter(|(i, _)| !landmarks.contains(i))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        landmarks.push(best_idx);

        // Update minimum distances
        for i in 0..n {
            let d = euclidean_distance(&points[best_idx], &points[i]);
            if d < min_distances[i] {
                min_distances[i] = d;
            }
        }
    }

    landmarks
}

/// Compute Euclidean distance between two points.
pub fn euclidean_distance(a: &Point, b: &Point) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute the distance matrix between all pairs of points.
pub fn distance_matrix(points: &[Point]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut dm = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(&points[i], &points[j]);
            dm[i][j] = d;
            dm[j][i] = d;
        }
    }
    dm
}

/// Compute distances from each point to the landmark set.
/// Returns a vector where each element is the distance to the nearest landmark.
pub fn distances_to_landmarks(points: &[Point], landmarks: &[usize]) -> Vec<f64> {
    points
        .iter()
        .map(|p| {
            landmarks
                .iter()
                .map(|&l| euclidean_distance(p, &points[l]))
                .fold(f64::INFINITY, |a, b| a.min(b))
        })
        .collect()
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
            vec![2.0, 0.0],
        ]
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance_same() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(euclidean_distance(&a, &a).abs() < 1e-10);
    }

    #[test]
    fn test_random_landmarks_count() {
        let cloud = sample_cloud();
        let landmarks = random_landmarks(&cloud, 3, 42);
        assert_eq!(landmarks.len(), 3);
        assert_eq!(landmarks.iter().collect::<std::collections::HashSet<_>>().len(), 3);
    }

    #[test]
    fn test_random_landmarks_all() {
        let cloud = sample_cloud();
        let landmarks = random_landmarks(&cloud, 10, 42);
        assert_eq!(landmarks.len(), 6); // only 6 points
    }

    #[test]
    fn test_maxmin_landmarks_count() {
        let cloud = sample_cloud();
        let landmarks = maxmin_landmarks(&cloud, 3);
        assert_eq!(landmarks.len(), 3);
    }

    #[test]
    fn test_maxmin_landmarks_coverage() {
        let cloud = sample_cloud();
        let landmarks = maxmin_landmarks(&cloud, 3);
        // Maxmin should pick points that are far apart
        // First landmark is always index 0
        assert_eq!(landmarks[0], 0);
    }

    #[test]
    fn test_maxmin_landmarks_empty() {
        let landmarks = maxmin_landmarks(&[], 3);
        assert!(landmarks.is_empty());
    }

    #[test]
    fn test_distance_matrix() {
        let cloud = vec![vec![0.0], vec![1.0], vec![3.0]];
        let dm = distance_matrix(&cloud);
        assert!((dm[0][1] - 1.0).abs() < 1e-10);
        assert!((dm[0][2] - 3.0).abs() < 1e-10);
        assert!((dm[1][2] - 2.0).abs() < 1e-10);
        assert!((dm[0][0]).abs() < 1e-10);
    }

    #[test]
    fn test_distances_to_landmarks() {
        let cloud = sample_cloud();
        let landmarks = vec![0, 3]; // (0,0) and (1,1)
        let dists = distances_to_landmarks(&cloud, &landmarks);
        // Point (0.5, 0.5) is equidistant: sqrt(0.5)
        let expected = (0.5_f64.powi(2) + 0.5_f64.powi(2)).sqrt();
        assert!((dists[4] - expected).abs() < 1e-10);
    }
}
