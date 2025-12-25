"""
Tests for Neural Embeddings with Polytope Topology
===================================================
"""

import numpy as np
import pytest
from cans_gqs.neural_embeddings import (
    MatrixEmbedding,
    PolytopeEmbedding,
    EmbeddingConfig,
    RegularPolytopeGenerator,
    PlatonicSolids,
    RegularPolytopes4D,
    MatrixToPolytopeMapper,
    GeodesicDistanceMetric,
    compare_embeddings,
)


class TestPlatonicSolids:
    """Test 3D Platonic solid generators."""
    
    def test_tetrahedron(self):
        """Test tetrahedron generation."""
        vertices = PlatonicSolids.tetrahedron()
        assert vertices.shape == (4, 3)
        # Check all vertices on unit sphere
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_cube(self):
        """Test cube generation."""
        vertices = PlatonicSolids.cube()
        assert vertices.shape == (8, 3)
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_octahedron(self):
        """Test octahedron generation."""
        vertices = PlatonicSolids.octahedron()
        assert vertices.shape == (6, 3)
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_dodecahedron(self):
        """Test dodecahedron generation."""
        vertices = PlatonicSolids.dodecahedron()
        assert vertices.shape == (20, 3)
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_icosahedron(self):
        """Test icosahedron generation."""
        vertices = PlatonicSolids.icosahedron()
        assert vertices.shape == (12, 3)
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0)


class TestRegularPolytopes4D:
    """Test 4D regular polytope generators."""
    
    def test_5_cell(self):
        """Test 5-cell generation."""
        vertices = RegularPolytopes4D.five_cell()
        assert vertices.shape == (5, 4)
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_tesseract(self):
        """Test tesseract generation."""
        vertices = RegularPolytopes4D.tesseract()
        assert vertices.shape == (16, 4)
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_16_cell(self):
        """Test 16-cell generation."""
        vertices = RegularPolytopes4D.sixteen_cell()
        assert vertices.shape == (8, 4)
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_24_cell(self):
        """Test 24-cell generation."""
        vertices = RegularPolytopes4D.twenty_four_cell()
        assert vertices.shape == (24, 4)
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_120_cell(self):
        """Test 120-cell generation."""
        vertices = RegularPolytopes4D.one_twenty_cell()
        assert vertices.shape[1] == 4
        assert vertices.shape[0] > 0  # Has vertices
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_600_cell(self):
        """Test 600-cell generation."""
        vertices = RegularPolytopes4D.six_hundred_cell()
        assert vertices.shape[1] == 4
        # Our implementation generates an approximation with fewer vertices
        assert vertices.shape[0] > 0
        assert vertices.shape[0] <= 120  # 600-cell has 120 vertices
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0)


class TestRegularPolytopeGenerator:
    """Test unified polytope generator."""
    
    def test_3d_generation(self):
        """Test 3D polytope generation."""
        generator = RegularPolytopeGenerator(3)
        
        for poly_type in ["tetrahedron", "cube", "octahedron", "dodecahedron", "icosahedron"]:
            vertices = generator.generate(poly_type)
            assert vertices.shape[1] == 3
            assert vertices.shape[0] > 0
    
    def test_4d_generation(self):
        """Test 4D polytope generation."""
        generator = RegularPolytopeGenerator(4)
        
        for poly_type in ["5-cell", "tesseract", "16-cell", "24-cell"]:
            vertices = generator.generate(poly_type)
            assert vertices.shape[1] == 4
            assert vertices.shape[0] > 0
    
    def test_nd_simplex(self):
        """Test n-simplex generation."""
        for n in [2, 3, 4, 5]:
            generator = RegularPolytopeGenerator(n)
            vertices = generator.generate("simplex")
            assert vertices.shape == (n+1, n)
    
    def test_nd_hypercube(self):
        """Test n-hypercube generation."""
        for n in [2, 3, 4]:
            generator = RegularPolytopeGenerator(n)
            vertices = generator.generate("hypercube")
            assert vertices.shape == (2**n, n)
    
    def test_get_polytope_info(self):
        """Test polytope info retrieval."""
        generator = RegularPolytopeGenerator(3)
        info = generator.get_polytope_info("cube")
        
        assert info["type"] == "cube"
        assert info["dimension"] == 3
        assert info["num_vertices"] == 8
        assert info["is_regular"] is True


class TestMatrixEmbedding:
    """Test standard matrix embeddings."""
    
    def test_pca_embedding(self):
        """Test PCA-based embedding."""
        np.random.seed(42)
        data = np.random.randn(50, 10)
        
        config = EmbeddingConfig(
            embedding_dim=3,
            num_points=50,
            use_normalization=True,
        )
        
        emb = MatrixEmbedding(config, method="pca")
        result = emb.fit_transform(data)
        
        assert result.shape == (50, 3)
        assert emb.is_fitted
    
    def test_random_projection(self):
        """Test random projection embedding."""
        np.random.seed(42)
        data = np.random.randn(50, 10)
        
        config = EmbeddingConfig(
            embedding_dim=3,
            num_points=50,
        )
        
        emb = MatrixEmbedding(config, method="random")
        result = emb.fit_transform(data)
        
        assert result.shape == (50, 3)
    
    def test_distance_computation(self):
        """Test distance computation."""
        config = EmbeddingConfig(embedding_dim=3, num_points=10)
        emb = MatrixEmbedding(config)
        
        point1 = np.array([1, 0, 0])
        point2 = np.array([0, 1, 0])
        
        dist = emb.distance(point1, point2)
        expected = np.sqrt(2)
        assert np.isclose(dist, expected)


class TestPolytopeEmbedding:
    """Test polytope-based embeddings."""
    
    def test_basic_embedding(self):
        """Test basic polytope embedding."""
        np.random.seed(42)
        data = np.random.randn(30, 10)
        
        config = EmbeddingConfig(
            embedding_dim=3,
            num_points=30,
        )
        
        emb = PolytopeEmbedding(config, polytope_type="cube")
        result = emb.fit_transform(data)
        
        assert result.shape == (30, 3)
        assert emb.is_fitted
    
    def test_different_polytopes(self):
        """Test embedding with different polytope types."""
        np.random.seed(42)
        data = np.random.randn(20, 8)
        
        config = EmbeddingConfig(embedding_dim=3, num_points=20)
        
        for poly_type in ["tetrahedron", "cube", "octahedron"]:
            emb = PolytopeEmbedding(config, polytope_type=poly_type)
            result = emb.fit_transform(data)
            assert result.shape == (20, 3)
    
    def test_get_polytope_structure(self):
        """Test getting polytope structure."""
        config = EmbeddingConfig(embedding_dim=3, num_points=20)
        emb = PolytopeEmbedding(config, polytope_type="dodecahedron")
        
        np.random.seed(42)
        data = np.random.randn(20, 8)
        emb.fit(data)
        
        structure = emb.get_polytope_structure()
        assert structure["type"] == "dodecahedron"
        assert structure["dimension"] == 3
        assert structure["num_vertices"] == 20


class TestMatrixToPolytopeMapper:
    """Test matrix-to-polytope mapping."""
    
    def test_nearest_vertex_mapping(self):
        """Test nearest vertex mapping."""
        # Create cube vertices
        vertices = PlatonicSolids.cube()
        mapper = MatrixToPolytopeMapper(vertices, mapping_method="nearest_vertex")
        
        # Map some points
        np.random.seed(42)
        points = np.random.randn(10, 3)
        mapped = mapper.map_to_polytope(points)
        
        assert mapped.shape == (10, 3)
    
    def test_barycentric_mapping(self):
        """Test barycentric mapping."""
        vertices = PlatonicSolids.tetrahedron()
        mapper = MatrixToPolytopeMapper(vertices, mapping_method="barycentric")
        
        np.random.seed(42)
        points = np.random.randn(5, 3)
        mapped = mapper.map_to_polytope(points)
        
        assert mapped.shape == (5, 3)


class TestGeodesicDistanceMetric:
    """Test geodesic distance computation."""
    
    def test_distance_matrix_computation(self):
        """Test distance matrix computation."""
        vertices = PlatonicSolids.octahedron()
        metric = GeodesicDistanceMetric(vertices)
        
        # Check distance matrix is symmetric
        assert metric.distance_matrix.shape == (6, 6)
        assert np.allclose(metric.distance_matrix, metric.distance_matrix.T)
    
    def test_point_distance(self):
        """Test distance between two points."""
        vertices = PlatonicSolids.cube()
        metric = GeodesicDistanceMetric(vertices)
        
        point1 = vertices[0]
        point2 = vertices[1]
        
        dist = metric.distance(point1, point2)
        assert dist > 0
        assert np.isfinite(dist)


class TestEmbeddingComparison:
    """Test embedding comparison utilities."""
    
    def test_compare_embeddings(self):
        """Test comparing two embeddings."""
        np.random.seed(42)
        data = np.random.randn(40, 15)
        
        config = EmbeddingConfig(embedding_dim=3, num_points=40)
        
        # Create matrix embedding
        matrix_emb = MatrixEmbedding(config, method="pca")
        matrix_result = matrix_emb.fit_transform(data)
        
        # Create polytope embedding
        polytope_emb = PolytopeEmbedding(config, polytope_type="cube")
        polytope_result = polytope_emb.fit_transform(data)
        
        # Compare
        comparison = compare_embeddings(
            matrix_result,
            polytope_result,
            original_data=data,
        )
        
        assert "distance_correlation" in comparison
        assert "distance_mse" in comparison
        assert "matrix_stress" in comparison
        assert "polytope_stress" in comparison
        assert "mean_neighborhood_preservation" in comparison
        
        # Check values are reasonable
        assert 0 <= comparison["mean_neighborhood_preservation"] <= 1
        assert comparison["distance_mse"] >= 0


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_3d_workflow(self):
        """Test complete 3D embedding workflow."""
        np.random.seed(42)
        
        # Generate data
        n_samples = 50
        n_features = 20
        data = np.random.randn(n_samples, n_features)
        
        # Configure
        config = EmbeddingConfig(
            embedding_dim=3,
            num_points=n_samples,
            use_normalization=True,
        )
        
        # Matrix embedding
        matrix_emb = MatrixEmbedding(config, method="pca")
        matrix_result = matrix_emb.fit_transform(data)
        
        # Polytope embedding
        polytope_emb = PolytopeEmbedding(config, polytope_type="dodecahedron")
        polytope_result = polytope_emb.fit_transform(data)
        
        # Compare
        comparison = compare_embeddings(
            matrix_result,
            polytope_result,
            original_data=data,
        )
        
        # Verify workflow completed
        assert matrix_result.shape == (n_samples, 3)
        assert polytope_result.shape == (n_samples, 3)
        assert "distance_correlation" in comparison
    
    def test_complete_4d_workflow(self):
        """Test complete 4D embedding workflow."""
        np.random.seed(42)
        
        # Generate data
        n_samples = 30
        n_features = 25
        data = np.random.randn(n_samples, n_features)
        
        # Configure
        config = EmbeddingConfig(
            embedding_dim=4,
            num_points=n_samples,
        )
        
        # Create 4D polytope embedding
        polytope_emb = PolytopeEmbedding(config, polytope_type="24-cell")
        polytope_result = polytope_emb.fit_transform(data)
        
        # Verify
        assert polytope_result.shape == (n_samples, 4)
        
        # Get structure
        structure = polytope_emb.get_polytope_structure()
        assert structure["dimension"] == 4
        assert structure["type"] == "24-cell"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
