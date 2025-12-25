#!/usr/bin/env python3
"""
Dimensional Analysis Tool

Demonstrates the dimensional progression of circle topologies
from 1D (linear) through 4D (hypersphere) embeddings.
"""

from circle_topology import CircleTopology


def print_dimensional_table(max_n=9):
    """Print a detailed table showing dimensional progression."""
    print("=" * 80)
    print("DIMENSIONAL PROGRESSION: Circle Topologies Through Space")
    print("=" * 80)
    print()
    print("Mathematical Insight:")
    print("  As we increase dimensions, additional symmetries reduce the count")
    print("  of topologically distinct circle arrangements.")
    print()
    print("Progression:")
    print("  1D → 2D: Forget ordering (make commutative)")
    print("  2D → 3D: Forget root choice (flip transformations)")
    print("  3D → 4D: Quotient by centrosymmetry & full rotations")
    print()
    print("-" * 80)
    print(f"{'N':<3} | {'1D Linear':>10} | {'2D Planar':>10} | {'3D Sphere':>10} | {'4D Hyper':>10} | Reduction")
    print(f"{'':3} | {'(Catalan)':>10} | {'(Rooted)':>10} | {'(Unrooted)':>10} | {'(Clusters)':>10} | Factor")
    print("-" * 80)
    
    topology = CircleTopology()
    
    for n in range(max_n + 1):
        cat = topology.catalan_number(n)
        planar = topology.non_intersecting_circles(n)
        sphere = topology.sphere_surface_clusters(n)
        hyper = topology.hypersphere_4d_clusters(n)
        
        if n > 0:
            reduction = f"{cat/hyper:.1f}x"
        else:
            reduction = "-"
        
        print(f"{n:<3} | {cat:>10} | {planar:>10} | {sphere:>10} | {hyper:>10} | {reduction:>8}")
    
    print("-" * 80)
    print()
    
    # Show some interesting patterns
    print("Interesting Observations:")
    print()
    
    # Ratios
    print("Reduction Ratios for N=9:")
    n = 9
    cat = topology.catalan_number(n)
    planar = topology.non_intersecting_circles(n)
    sphere = topology.sphere_surface_clusters(n)
    hyper = topology.hypersphere_4d_clusters(n)
    
    print(f"  1D to 2D: {cat} → {planar} ({cat/planar:.2f}x reduction)")
    print(f"  2D to 3D: {planar} → {sphere} ({planar/sphere:.2f}x reduction)")
    print(f"  3D to 4D: {sphere} → {hyper} ({sphere/hyper:.2f}x reduction)")
    print(f"  1D to 4D: {cat} → {hyper} ({cat/hyper:.2f}x total reduction)")
    print()
    
    # OEIS references
    print("OEIS Sequence References:")
    print("  1D: A000108 - Catalan numbers")
    print("  2D: A000081 - Rooted trees")
    print("  3D: A000055 - Unrooted (free) trees")
    print("  4D: Theoretical - Centrosymmetric quotient")
    print()


def compare_dimensions(n_values=None):
    """Compare specific N values across dimensions."""
    if n_values is None:
        n_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    print("=" * 80)
    print("DIMENSIONAL COMPARISON FOR SPECIFIC VALUES")
    print("=" * 80)
    print()
    
    topology = CircleTopology()
    
    for n in n_values:
        print(f"N = {n}:")
        cat = topology.catalan_number(n)
        planar = topology.non_intersecting_circles(n)
        sphere = topology.sphere_surface_clusters(n)
        hyper = topology.hypersphere_4d_clusters(n)
        
        print(f"  1D (Linear):       {cat:6} arrangements")
        print(f"  2D (Planar):       {planar:6} topologies")
        print(f"  3D (Sphere):       {sphere:6} clusters")
        print(f"  4D (Hypersphere):  {hyper:6} clusters")
        
        if n > 2:
            print(f"  Path: {cat} → {planar} → {sphere} → {hyper}")
        print()


def analyze_reduction_pattern():
    """Analyze how the reduction factor changes with N."""
    print("=" * 80)
    print("REDUCTION PATTERN ANALYSIS")
    print("=" * 80)
    print()
    
    topology = CircleTopology()
    
    print(f"{'N':<3} | {'3D/4D Ratio':>12} | {'Pattern'}")
    print("-" * 40)
    
    for n in range(2, 10):
        sphere = topology.sphere_surface_clusters(n)
        hyper = topology.hypersphere_4d_clusters(n)
        
        ratio = sphere / hyper if hyper > 0 else 0
        
        # Analyze pattern
        if ratio < 1.5:
            pattern = "Minimal reduction"
        elif ratio < 2.5:
            pattern = "~2x reduction"
        elif ratio < 3.5:
            pattern = "~3x reduction"
        else:
            pattern = "Strong reduction"
        
        print(f"{n:<3} | {ratio:>12.2f} | {pattern}")
    
    print()
    print("Observation: The reduction factor appears to approach ~2.4x for larger N")
    print()


def main():
    """Main entry point."""
    print_dimensional_table(max_n=9)
    print()
    analyze_reduction_pattern()


if __name__ == "__main__":
    main()
