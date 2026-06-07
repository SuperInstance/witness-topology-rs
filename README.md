# witness-topology-rs

**Witness complex for topology approximation from point clouds.**

This crate implements witness complexes — a computationally efficient way to approximate the topology of high-dimensional point clouds. Select landmarks using random or maxmin strategies, build witness and weak witness complexes, compute persistent homology via boundary matrix reduction, and analyze topological structure across multiple scales. With 35 tests covering landmark selection, complex construction, persistence computation, and multi-scale analysis, it provides fast topological reasoning for any agent that needs to understand the shape of data.

## Why This Matters

Topology reveals structure that statistics misses. Two point clouds can have identical means and variances but completely different shapes — one a sphere, one a torus. Persistent homology captures these shape features (connected components, loops, voids) as a multi-scale topological fingerprint. The witness complex makes this tractable for large datasets by using a small set of landmark points, with non-landmark points acting as "witnesses" that vote on which simplices should exist. For an AGI system, this is the tool for understanding the qualitative shape of belief spaces, action spaces, and data manifolds — without getting lost in dimensionality.

## Quick Start

```toml
# Cargo.toml
[dependencies]
witness-topology-rs = "0.1.0"
```

```rust
use witness_topology_rs::landmark::{random_landmarks, maxmin_landmarks, distance_matrix};
use witness_topology_rs::witness::build_witness_complex;
use witness_topology_rs::persistence::{compute_persistence, betti_numbers};
use witness_topology_rs::scale::multiscale_analysis;
use witness_topology_rs::weak_witness::build_weak_witness_complex;

// Point cloud: points on a circle
let points: Vec<Vec<f64>> = (0..50)
    .map(|i| {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / 50.0;
        vec![theta.cos(), theta.sin()]
    })
    .collect();

// Select 10 landmarks (maxmin for good spatial coverage)
let landmarks = maxmin_landmarks(&points, 10);

// Build witness complex with k=3 nearest witnesses
let complex = build_witness_complex(&points, &landmarks, 3);
println!("Vertices: {}", complex.num_simplices(0));
println!("Edges: {}", complex.num_simplices(1));
println!("Triangles: {}", complex.num_simplices(2));
println!("Euler characteristic: {}", complex.euler_characteristic());

// Compute persistent homology
let dist = distance_matrix(&points);
let pairs = compute_persistence(&complex, &dist);
let betti = betti_numbers(&pairs, f64::INFINITY);
println!("Betti numbers: {:?}", betti); // [1, 1, 0, ...] — one component, one loop!

// Multi-scale analysis
let results = multiscale_analysis(&points, 10, &[2, 3, 5, 7], 42);
for r in &results {
    println!("k={}: {} vertices, {} edges, betti={:?}", r.k, r.num_vertices, r.num_edges, r.betti);
}
```

## Architecture

| Module | Purpose |
|---|---|
| `landmark` | Landmark selection (random, maxmin), distance matrix computation |
| `witness` | Witness complex construction from landmarks and witnesses |
| `weak_witness` | Weak witness variant — stricter simplex admission |
| `persistence` | Persistent homology via boundary matrix reduction, Betti numbers |
| `scale` | Multi-scale analysis sweeping the k parameter, variant comparison |

## API Tour

### Landmark Selection (`landmark`)

- **`random_landmarks(points, num_landmarks, seed) → Vec<usize>`** — Random subset
- **`maxmin_landmarks(points, num_landmarks) → Vec<usize>`** — Greedy farthest-point for spatial coverage
- **`euclidean_distance(a, b) → f64`** — Pairwise distance
- **`distance_matrix(points) → Vec<Vec<f64>>`** — All-pairs distances

### Witness Complex (`witness`)

- **`Simplex`** — Type alias for `Vec<usize>` (sorted vertex indices)
- **`SimplicialComplex { simplices }`** — Hierarchical simplex storage by dimension
  - `::new()` — Empty complex
  - `.add_simplex(simplex)` — Insert a simplex
  - `.num_simplices(dim)` — Count simplices at given dimension
  - `.euler_characteristic()` — Alternating sum of simplex counts
- **`build_witness_complex(points, landmarks, k) → SimplicialComplex`** — Standard witness complex

### Weak Witness (`weak_witness`)

- **`build_weak_witness_complex(points, landmarks, max_dim) → SimplicialComplex`**
  - More selective: only admits simplices with a *dedicated* witness point

### Persistence (`persistence`)

- **`PersistencePair { dim, birth, death }`** — A topological feature's lifetime
  - `.persistence() → f64` — Duration (death - birth, or ∞)
- **`compute_persistence(complex, dist) → Vec<PersistencePair>`** — Full persistence diagram
- **`betti_numbers(pairs, threshold) → Vec<usize>`** — Betti numbers at a filtration value

### Multi-Scale Analysis (`scale`)

- **`ScaleResult { k, num_vertices, num_edges, num_triangles, euler, betti }`** — Snapshot at one scale
- **`multiscale_analysis(points, num_landmarks, k_values, seed) → Vec<ScaleResult>`** — Sweep k
- **`compare_variants(points, num_landmarks, k_values, seed) → (Vec<ScaleResult>, Vec<ScaleResult>)`** — Witness vs. weak witness

## Performance

- Maxmin landmarks: O(k × n²) for k landmarks from n points
- Witness complex: O(n × k²) for n points and k landmarks
- Persistence: O(m³) for m simplices (boundary matrix reduction) — practical for complexes with ≤ ~1000 simplices
- Multi-scale: O(sweep × persistence_per_scale)
- Pure Rust, no external dependencies

## Ecosystem

Part of the **SuperInstance** family:

- [`sheaf-coherence-rs`](https://github.com/SuperInstance/sheaf-coherence-rs) — Sheaf-theoretic consistency
- [`persistent-sheaf-rs`](https://github.com/SuperInstance/persistent-sheaf-rs) — Persistent sheaf homology
- [`optimal-transport-rs`](https://github.com/SuperInstance/optimal-transport-rs) — Distribution comparison
- [`topo-sonata-rs`](https://github.com/SuperInstance/topo-sonata-rs) — Topological music analysis
- [`spectral-prosody-rs`](https://github.com/SuperInstance/spectral-prosody-rs) — Spectral feature extraction

## Ideas for Improvement

- **Approximate nearest neighbors** — Use `annoy` or `hnswlib` for landmark-witness lookup in high dimensions
- **Sparse persistence** — Cohomological persistence for speedup
- **Parallel boundary reduction** — Column operations are independent
- **Witness complex sparsification** — Density-aware witness weighting
- **GPU distance matrix** — Massive point clouds on CUDA/wGPU
- **Integration with `ripser`** — Backend swap for production-scale persistence

## License

MIT
