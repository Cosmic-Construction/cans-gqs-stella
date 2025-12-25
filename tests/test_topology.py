"""
Tests for the topology module.

Tests circle topology, flip transforms, and the planar-spherical bridge
that connects topology to polytope neural embeddings.
"""

import pytest
import numpy as np

from cans_gqs.topology import (
    CircleTopology,
    CircleExpression,
    find_flip_clusters,
    PlanarSphericalBridge,
    TopologyEmbeddingIntegrator,
    EulerCharacteristicAnalyzer,
)


class TestCircleTopology:
    """Tests for the CircleTopology class."""

    def test_catalan_numbers(self):
        """Test Catalan number computation."""
        # Known Catalan numbers: 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862
        expected = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862]
        for n, exp in enumerate(expected):
            assert CircleTopology.catalan_number(n) == exp

    def test_rooted_trees(self):
        """Test rooted tree counting (OEIS A000081)."""
        # A000081: 0, 1, 1, 2, 4, 9, 20, 48, 115, 286
        expected = [0, 1, 1, 2, 4, 9, 20, 48, 115, 286]
        for n, exp in enumerate(expected):
            assert CircleTopology.rooted_trees(n) == exp

    def test_unrooted_trees(self):
        """Test unrooted (free) tree counting (OEIS A000055)."""
        # A000055: 1, 1, 1, 1, 2, 3, 6, 11, 23, 47
        expected = [1, 1, 1, 1, 2, 3, 6, 11, 23, 47]
        for n, exp in enumerate(expected):
            assert CircleTopology.unrooted_trees(n) == exp

    def test_non_intersecting_circles(self):
        """Test non-intersecting circle counting."""
        # Should match rooted_trees(n+1)
        for n in range(10):
            assert CircleTopology.non_intersecting_circles(n) == \
                   CircleTopology.rooted_trees(n + 1)

    def test_sphere_surface_clusters(self):
        """Test sphere surface cluster counting."""
        # Should match unrooted_trees(n+1)
        for n in range(10):
            assert CircleTopology.sphere_surface_clusters(n) == \
                   CircleTopology.unrooted_trees(n + 1)

    def test_rooted_greater_than_unrooted(self):
        """Rooted trees should be >= unrooted trees."""
        for n in range(2, 15):
            assert CircleTopology.rooted_trees(n) >= CircleTopology.unrooted_trees(n)

    def test_generate_sequence(self):
        """Test sequence generation."""
        seq = CircleTopology.generate_sequence(5, 'none')
        assert len(seq) == 6
        assert seq[0] == CircleTopology.non_intersecting_circles(0)


class TestCircleExpression:
    """Tests for the CircleExpression class."""

    def test_valid_expression(self):
        """Test creation of valid expressions."""
        expr = CircleExpression('(())')
        assert expr.count_circles() == 2

    def test_invalid_expression(self):
        """Test that invalid expressions raise errors."""
        with pytest.raises(ValueError):
            CircleExpression('(()')

        with pytest.raises(ValueError):
            CircleExpression('())')

    def test_factor_count(self):
        """Test counting of top-level factors."""
        assert CircleExpression('()()()()').factor_count() == 4
        assert CircleExpression('(())()()').factor_count() == 3
        assert CircleExpression('(((())))').factor_count() == 1
        assert CircleExpression('(()())()').factor_count() == 2

    def test_flip_transform(self):
        """Test flip transformation generates alternatives."""
        expr = CircleExpression('()()')
        flips = expr.flip_transform()
        assert '()()' in flips
        assert len(flips) >= 1


class TestEulerCharacteristic:
    """Tests for the EulerCharacteristicAnalyzer class."""

    def test_platonic_solids_euler(self):
        """All Platonic solids have Euler characteristic 2."""
        for polytope in ['tetrahedron', 'cube', 'octahedron',
                         'dodecahedron', 'icosahedron']:
            chi = EulerCharacteristicAnalyzer.euler_characteristic_3d(polytope)
            assert chi == 2, f"{polytope} should have χ=2, got {chi}"

    def test_4d_polytopes_euler(self):
        """All regular 4D polytopes have Euler characteristic 0."""
        for polytope in ['5-cell', 'tesseract', '16-cell', '24-cell',
                         '120-cell', '600-cell']:
            chi = EulerCharacteristicAnalyzer.euler_characteristic_4d(polytope)
            assert chi == 0, f"{polytope} should have χ=0, got {chi}"

    def test_angular_defect_positive(self):
        """Angular defects should be positive for Platonic solids."""
        for polytope in ['tetrahedron', 'cube', 'octahedron',
                         'dodecahedron', 'icosahedron']:
            defect = EulerCharacteristicAnalyzer.angular_defect(polytope)
            assert defect > 0

    def test_dodecahedron_has_12_faces(self):
        """Verify dodecahedron has 12 pentagonal faces."""
        data = EulerCharacteristicAnalyzer.PLATONIC_SOLIDS['dodecahedron']
        assert data['F'] == 12
        assert data['face_type'] == 5  # Pentagon


class TestPlanarSphericalBridge:
    """Tests for the PlanarSphericalBridge class."""

    def test_dimensional_reduction_ratio(self):
        """Test dimensional reduction ratio computation."""
        bridge = PlanarSphericalBridge()
        ratios = bridge.dimensional_reduction_ratio(5)

        # All ratios should be >= 1 (reduction)
        assert ratios['1D_to_2D'] >= 1
        assert ratios['2D_to_3D'] >= 1
        assert ratios['1D_to_4D'] >= ratios['1D_to_2D']

    def test_hexagon_pentagon_defect(self):
        """Test hexagon to pentagon analysis."""
        bridge = PlanarSphericalBridge()
        result = bridge.hexagon_to_pentagon_defect()

        # Hexagon interior angle should be 120 degrees
        assert np.isclose(np.degrees(result['hexagon_interior_angle_rad']), 120)

        # Pentagon interior angle should be 108 degrees
        assert np.isclose(np.degrees(result['pentagon_interior_angle_rad']), 108)

        # Must have exactly 12 pentagons
        assert result['required_pentagons'] == 12

    def test_tree_root_equivalence(self):
        """Test rooted to free tree equivalence analysis."""
        bridge = PlanarSphericalBridge()
        result = bridge.tree_root_to_flip_equivalence(5)

        assert result['rooted_trees'] >= result['free_trees']
        assert result['symmetry_factor'] >= 1


class TestTopologyEmbeddingIntegrator:
    """Tests for the TopologyEmbeddingIntegrator class."""

    def test_embedding_capacity_3d(self):
        """Test embedding capacity analysis for 3D polytopes."""
        integrator = TopologyEmbeddingIntegrator()

        for polytope in ['tetrahedron', 'cube', 'dodecahedron']:
            capacity = integrator.embedding_capacity_analysis(polytope)
            assert 'error' not in capacity
            assert capacity['dimension'] == 3
            assert capacity['vertices'] > 0
            assert capacity['euler_characteristic'] == 2

    def test_embedding_capacity_4d(self):
        """Test embedding capacity analysis for 4D polytopes."""
        integrator = TopologyEmbeddingIntegrator()

        for polytope in ['tesseract', '24-cell', '120-cell']:
            capacity = integrator.embedding_capacity_analysis(polytope)
            assert 'error' not in capacity
            assert capacity['dimension'] == 4
            assert capacity['euler_characteristic'] == 0

    def test_tensor_shape_implications(self):
        """Test tensor shape analysis."""
        integrator = TopologyEmbeddingIntegrator()
        tensors = integrator.tensor_shape_implications(100, 'dodecahedron')

        assert tensors['embedding_tensor']['shape'] == (100, 3)
        assert tensors['distance_tensor']['unique_classes'] > 0
        assert tensors['vertex_assignment_tensor']['avg_per_vertex'] == 5.0

    def test_rooted_to_free_analogy(self):
        """Test the rooted→free embedding analogy."""
        integrator = TopologyEmbeddingIntegrator()
        analogy = integrator.rooted_to_free_embedding_analogy(5)

        assert 'tree_analogy' in analogy
        assert 'tiling_analogy' in analogy
        assert 'embedding_implication' in analogy
        assert analogy['embedding_implication']['compression_factor'] >= 1


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_full_dimensional_progression(self):
        """Test the full 1D→2D→3D→4D progression."""
        topology = CircleTopology()

        for n in range(1, 8):
            catalan = topology.catalan_number(n)
            rooted = topology.rooted_trees(n)
            unrooted = topology.unrooted_trees(n)

            # Each step should reduce or maintain count
            assert catalan >= rooted, f"n={n}: catalan >= rooted failed"
            assert rooted >= unrooted, f"n={n}: rooted >= unrooted failed"

    def test_dodecahedron_twelve_pentagons_connection(self):
        """Verify the 12 pentagons = sphere closure connection."""
        bridge = PlanarSphericalBridge()
        euler = EulerCharacteristicAnalyzer()

        # Hexagon-pentagon analysis
        hex_pent = bridge.hexagon_to_pentagon_defect()
        assert hex_pent['required_pentagons'] == 12

        # Dodecahedron has 12 faces
        dodeca = euler.PLATONIC_SOLIDS['dodecahedron']
        assert dodeca['F'] == 12

        # Both point to the same topological constraint!

    def test_gauss_bonnet_sum(self):
        """Test Gauss-Bonnet theorem for Platonic solids."""
        euler = EulerCharacteristicAnalyzer()

        for polytope_name, data in euler.PLATONIC_SOLIDS.items():
            # Total angular defect should be 4π
            defect_per_vertex = euler.angular_defect(polytope_name)
            total_defect = defect_per_vertex * data['V']
            assert np.isclose(total_defect, 4 * np.pi), \
                f"{polytope_name}: total defect should be 4π"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
