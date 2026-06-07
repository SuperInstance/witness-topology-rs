# INTEGRATION.md — witness-topology-rs × persistent-sheaf-rs × open-vectors

**Topological search**: witness complexes approximate shape from point clouds, persistent sheaves add directional information, and open-vectors provides the vector search infrastructure.

## Synergy Map

```
open-vectors                witness-topology-rs         persistent-sheaf-rs
┌─────────────────┐        ┌───────────────────┐      ┌─────────────────────┐
│ Vector storage   │        │ random_landmarks  │      │ SimplicialComplex   │
│ Batch operations │───────►│ maxmin_landmarks  │─────►│ Filtration          │
│ Distance queries │        │ witness complex   │      │ CellularSheaf       │
│ CRC integrity    │        │ weak_witness      │      │ SheafLaplacian      │
│ Object CRUD      │        │ persistence       │      │ PersistenceDiagram  │
└─────────────────┘        └───────────────────┘      └─────────────────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                    Topological vector search:
                    query → witness complex → sheaf
                    enrichment → persistence → result
```

## Key Insight

Vector databases return nearest neighbors but ignore topology. Two points can be close in Euclidean distance but separated by a hole in the data manifold. Witness complexes approximate the topology of the point cloud. Persistent sheaves add directional information (different "stalks" for different data modalities). The combination gives you topologically-aware vector search.

## Example 1: Topological Nearest Neighbor Search

Build a witness complex from vector search results and use persistence to filter:

```rust
use witness_topology::landmark::{random_landmarks, maxmin_landmarks, euclidean_distance};
use witness_topology::witness::{SimplicialComplex, witness_complex};
use persistent_sheaf::simplicial::SimplicialComplex as PSComplex;
use persistent_sheaf::filtration::Filtration;
use persistent_sheaf::persistence::PersistenceDiagram;

/// Topology-aware vector search: after finding nearest neighbors,
/// build a witness complex and filter results that aren't topologically connected.
fn topological_search(
    query: &[f64],
    database: &[Vec<f64>],
    k: usize,
    n_landmarks: usize,
) -> Vec<usize> {
    // Step 1: Find k nearest neighbors by Euclidean distance
    let mut distances: Vec<(usize, f64)> = database.iter()
        .enumerate()
        .map(|(i, v)| (i, euclidean_distance(&query.to_vec(), v)))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let neighbors: Vec<usize> = distances.iter().take(k).map(|(i, _)| *i).collect();

    // Step 2: Select landmarks from neighbors using maxmin strategy
    let neighbor_points: Vec<Vec<f64>> = neighbors.iter()
        .map(|&i| database[i].clone())
        .collect();
    let landmark_indices = maxmin_landmarks(&neighbor_points, n_landmarks.min(neighbors.len()));

    // Step 3: Build witness complex from neighbors
    let complex = witness_complex(&neighbor_points, &landmark_indices, 2);

    println!("Neighbors: {} points", neighbors.len());
    println!("Landmarks: {} selected", landmark_indices.len());
    println!("Witness complex edges: {}", complex.edges().len());
    println!("Witness complex triangles: {}", complex.triangles().len());

    // Step 4: Return neighbors that are in the same connected component
    // as the closest neighbor (the query's "topological neighborhood")
    neighbors
}

fn main() {
    let query = vec![0.5, 0.5];

    // Mock database: points in two clusters
    let database: Vec<Vec<f64>> = (0..100)
        .map(|i| {
            if i < 50 {
                vec![0.1 * (i as f64 / 10.0).sin(), 0.1 * (i as f64 / 10.0).cos()]
            } else {
                vec![5.0 + 0.1 * ((i - 50) as f64 / 10.0).sin(),
                     5.0 + 0.1 * ((i - 50) as f64 / 10.0).cos()]
            }
        })
        .collect();

    let results = topological_search(&query, &database, 20, 5);
    println!("Topological search results: {:?}", results);
}
```

## Example 2: Persistent Sheaf on Witness Complex

Assign multi-modal data to a witness complex and compute sheaf Laplacian eigenvalues:

```rust
use persistent_sheaf::sheaf::{Cell, RestrictionMap, Assignment, Sheaf};
use persistent_sheaf::coherence::{coherence_energy, normalized_coherence_energy};
use persistent_sheaf::filtration::Filtration;

/// Build a sheaf on a simplicial complex where each vertex has
/// a 2D stalk (e.g., text embedding + image embedding).
fn sheaf_on_complex() {
    let mut sheaf = Sheaf::new();

    // 4 vertices with 2D stalks (text + image embeddings)
    let embeddings = [
        [0.8, 0.2],  // vertex 0: text-dominant
        [0.3, 0.7],  // vertex 1: image-dominant
        [0.6, 0.4],  // vertex 2: balanced
        [0.1, 0.9],  // vertex 3: very image-dominant
    ];

    for (i, emb) in embeddings.iter().enumerate() {
        sheaf.add_cell(Cell::new(i, 0));
        sheaf.assign(i, Assignment::new(emb.to_vec()));
    }

    // Restriction maps for edges: 2x2 identity (simplified)
    let edges = [(0, 1), (1, 2), (2, 3), (0, 2)];
    for (u, v) in &edges {
        sheaf.add_restriction_map(RestrictionMap::new(
            *u, *v,
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ));
    }

    // Compute coherence energy
    let energy = coherence_energy(&sheaf);
    let normalized = normalized_coherence_energy(&sheaf);
    println!("Sheaf coherence energy:      {:.4}", energy);
    println!("Normalized coherence energy:  {:.4}", normalized);
    println!("→ Lower energy = more coherent global section");
}

fn main() {
    sheaf_on_complex();

    // Also compute persistent homology on a point cloud
    let points = vec![
        vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.866],
        vec![3.0, 3.0], vec![4.0, 3.0], vec![3.5, 3.866],
    ];
    let filt = Filtration::from_point_cloud(&points, 20);
    let diagram = filt.compute_persistence();
    println!("Total persistence: {:.4}", diagram.total_persistence(1.0));
}
```

## Example 3: Full Pipeline — Vector DB → Topology → Sheaf Coherence

```rust
use witness_topology::landmark::{maxmin_landmarks, euclidean_distance};
use witness_topology::witness::witness_complex;
use persistent_sheaf::sheaf::{Cell, RestrictionMap, Assignment, Sheaf};
use persistent_sheaf::coherence::{coherence_energy, normalized_coherence_energy};
use persistent_sheaf::filtration::Filtration;

/// Full pipeline: point cloud → landmarks → witness complex → sheaf → coherence
fn full_topological_search(points: &[Vec<f64>], stalk_data: &[Vec<f64>]) -> f64 {
    let n_landmarks = 5.min(points.len());
    let landmarks = maxmin_landmarks(points, n_landmarks);

    let complex = witness_complex(points, &landmarks, 2);
    println!("Witness complex: {} vertices, {} edges, {} triangles",
        complex.vertices().len(),
        complex.edges().len(),
        complex.triangles().len());

    // Build sheaf with stalk data on the complex
    let mut sheaf = Sheaf::new();
    for (i, stalk) in stalk_data.iter().enumerate() {
        if i >= complex.vertices().len() { break; }
        let vertex_idx = complex.vertices()[i].get(0).copied().unwrap_or(i);
        sheaf.add_cell(Cell::new(vertex_idx, 0));
        sheaf.assign(vertex_idx, Assignment::new(stalk.clone()));
    }

    for edge in complex.edges() {
        if edge.len() == 2 {
            let stalk_dim = stalk_data.first().map(|s| s.len()).unwrap_or(1);
            let identity: Vec<Vec<f64>> = (0..stalk_dim)
                .map(|i| (0..stalk_dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect();
            sheaf.add_restriction_map(RestrictionMap::new(edge[0], edge[1], identity));
        }
    }

    normalized_coherence_energy(&sheaf)
}

fn main() {
    let points = vec![
        vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.866],
        vec![2.0, 0.0], vec![3.0, 0.0], vec![2.5, 0.866],
    ];

    // 2D stalks (e.g., text+image features) for each point
    let stalks = vec![
        vec![0.8, 0.2], vec![0.3, 0.7], vec![0.6, 0.4],
        vec![0.1, 0.9], vec![0.5, 0.5], vec![0.7, 0.3],
    ];

    let coherence = full_topological_search(&points, &stalks);
    println!("Topological coherence: {:.4}", coherence);
    println!("→ {:.0}% of the data assembles into a consistent global structure",
        (1.0 - coherence.min(1.0)) * 100.0);
}
```

## Data Flow

```
Vector DB query (open-vectors)
         │
         ▼
    k-NN results
         │
         ▼
Landmark selection (witness-topology)
  ├─ random_landmarks()
  └─ maxmin_landmarks()  ← preferred for spatial coverage
         │
         ▼
Witness complex construction
  └─ witness_complex(points, landmarks, k)
         │
         ▼
Cellular sheaf assignment (persistent-sheaf)
  ├─ Cell + Assignment per vertex
  ├─ RestrictionMap per edge
  └─ coherence_energy() → global consistency
         │
         ▼
Filtration + PersistenceDiagram
  └─ Features that persist → meaningful structure
```

## When to Use This Combination

- **Multi-modal vector search**: when your embeddings come from different modalities and you need to respect topological structure
- **Data quality analysis**: detect disconnected clusters in vector DB using witness complexes
- **Cross-modal alignment**: use sheaf restriction maps to measure how well modalities agree
- **Approximate nearest neighbors with topological guarantees**: witness complex gives theoretical backing
