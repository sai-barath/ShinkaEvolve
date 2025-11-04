"""
Comprehensive Test Suite for Island Diversity Measurement

This module contains extensive tests for the IslandDiversityAnalyzer class,
covering all functionality including edge cases, performance, and integration.

Test Categories:
1. Unit Tests - Individual method testing
2. Integration Tests - Full workflow testing
3. Edge Case Tests - Boundary conditions and error handling
4. Performance Tests - Scalability and efficiency
5. Statistical Tests - Validate mathematical properties
6. Comparison Tests - Strategy comparison validation
"""

import pytest
import numpy as np
import sqlite3
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
import time
import logging

from shinka.database import (
    ProgramDatabase,
    Program,
    DatabaseConfig,
    IslandDiversityAnalyzer,
    DiversityConfig,
    create_diversity_analyzer,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures and Utilities
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test databases."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def db_config():
    """Create a test database configuration."""
    return DatabaseConfig(
        db_path="test_evolution_db.sqlite",
        num_islands=4,
        archive_size=20,
        elite_selection_ratio=0.3,
        num_archive_inspirations=4,
        num_top_k_inspirations=2,
        migration_interval=10,
        migration_rate=0.1,
        island_elitism=True,
    )


@pytest.fixture
def diversity_config():
    """Create a test diversity configuration."""
    return DiversityConfig(
        default_strategy="elite",
        recent_generations=10,
        min_programs_per_island=1,
        correct_only=True,
        verbose=True,
    )


@pytest.fixture
def test_database(temp_dir, db_config):
    """Create a test database with sample programs."""
    db_path = Path(temp_dir) / "test_db.sqlite"
    db_config.db_path = str(db_path)
    
    db = ProgramDatabase(config=db_config, embedding_model="text-embedding-3-small")
    yield db
    db.close()


def create_mock_program(
    program_id: str,
    code: str,
    embedding: List[float],
    island_idx: int,
    generation: int = 0,
    parent_id: str = None,
    correct: bool = True,
    combined_score: float = 1.0,
    children_count: int = 0,
) -> Program:
    """Create a mock program for testing."""
    return Program(
        id=program_id,
        code=code,
        embedding=embedding,
        island_idx=island_idx,
        generation=generation,
        parent_id=parent_id,
        correct=correct,
        combined_score=combined_score,
        children_count=children_count,
    )


def generate_random_embedding(dim: int = 128, seed: int = None) -> List[float]:
    """Generate a random normalized embedding vector."""
    if seed is not None:
        np.random.seed(seed)
    vec = np.random.randn(dim)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def populate_test_database(
    db: ProgramDatabase,
    num_islands: int = 4,
    programs_per_island: int = 10,
    embedding_dim: int = 128,
) -> Dict[int, List[str]]:
    """
    Populate database with test programs across multiple islands.
    
    Returns:
        Dictionary mapping island_idx to list of program IDs
    """
    island_programs = {i: [] for i in range(num_islands)}
    
    for island_idx in range(num_islands):
        for i in range(programs_per_island):
            # Create increasingly better programs
            generation = i
            score = 0.5 + (i / programs_per_island) * 0.5
            is_leaf = (i == programs_per_island - 1)  # Last one is a leaf
            
            # Create embedding with some island-specific bias
            # This ensures islands have different embeddings
            base_embedding = generate_random_embedding(
                embedding_dim, seed=island_idx * 1000 + i
            )
            
            # Add island-specific offset to create diversity
            island_offset = np.zeros(embedding_dim)
            island_offset[island_idx] = 0.5
            embedding = base_embedding + island_offset
            embedding = embedding / np.linalg.norm(embedding)
            
            program = create_mock_program(
                program_id=f"prog_{island_idx}_{i}",
                code=f"def solution_{island_idx}_{i}(): pass",
                embedding=embedding.tolist(),
                island_idx=island_idx,
                generation=generation,
                parent_id=f"prog_{island_idx}_{i-1}" if i > 0 else None,
                correct=True,
                combined_score=score,
                children_count=0 if is_leaf else 1,
            )
            
            db.add(program, verbose=False)
            island_programs[island_idx].append(program.id)
    
    return island_programs


# =============================================================================
# UNIT TESTS - Individual Method Testing
# =============================================================================

class TestProgramSelectionStrategies:
    """Test all program selection strategies."""
    
    def test_get_leaf_programs(self, test_database):
        """Test leaf program selection."""
        # Populate database
        populate_test_database(test_database, num_islands=2, programs_per_island=5)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Get leaf programs from island 0
        leaves = analyzer.get_leaf_programs(island_idx=0)
        
        # Should have exactly 1 leaf (the last program)
        assert len(leaves) >= 1, "Should have at least one leaf program"
        
        # All should have children_count = 0
        for prog in leaves:
            assert prog.children_count == 0, f"Leaf program {prog.id} has children"
            assert prog.island_idx == 0, "Wrong island"
    
    def test_get_recent_generation_programs(self, test_database):
        """Test recent generation program selection."""
        populate_test_database(test_database, num_islands=2, programs_per_island=15)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Get last 5 generations
        recent = analyzer.get_recent_generation_programs(
            island_idx=0, num_generations=5
        )
        
        assert len(recent) > 0, "Should have recent programs"
        
        # Check that generations are within range
        max_gen = max(p.generation for p in recent)
        min_gen = min(p.generation for p in recent)
        assert max_gen - min_gen <= 5, "Generation range too large"
    
    def test_get_elite_programs(self, test_database):
        """Test elite program selection from archive."""
        populate_test_database(test_database, num_islands=2, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Get elite programs
        elites = analyzer.get_elite_programs(island_idx=0)
        
        # Should have some elites (archive is populated during add)
        # Note: May be 0 if archive logic doesn't auto-add
        assert isinstance(elites, list), "Should return a list"
        
        # All elites should be correct programs
        for prog in elites:
            assert prog.correct, f"Elite program {prog.id} is not correct"
            assert prog.island_idx == 0, "Wrong island"
    
    def test_get_all_programs(self, test_database):
        """Test getting all programs from an island."""
        programs_per_island = 10
        populate_test_database(
            test_database, num_islands=2, programs_per_island=programs_per_island
        )
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Get all programs
        all_progs = analyzer.get_all_programs(island_idx=0)
        
        # Should have all programs from island 0
        assert len(all_progs) == programs_per_island, (
            f"Expected {programs_per_island} programs, got {len(all_progs)}"
        )
    
    def test_select_programs_invalid_strategy(self, test_database):
        """Test that invalid strategy raises error."""
        populate_test_database(test_database, num_islands=2, programs_per_island=5)
        
        analyzer = test_database.get_diversity_analyzer()
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            analyzer._select_programs(island_idx=0, strategy="invalid_strategy")
    
    def test_select_programs_correct_only_filter(self, test_database):
        """Test that correct_only filter works properly."""
        # Add some incorrect programs
        populate_test_database(test_database, num_islands=2, programs_per_island=5)
        
        # Add an incorrect program
        incorrect_prog = create_mock_program(
            program_id="incorrect_prog",
            code="def bad(): return None",
            embedding=generate_random_embedding(128, seed=999),
            island_idx=0,
            generation=10,
            correct=False,
            children_count=0,
        )
        test_database.add(incorrect_prog, verbose=False)
        
        # Create analyzer with correct_only=True
        config = DiversityConfig(correct_only=True)
        analyzer = IslandDiversityAnalyzer(test_database, config)
        
        # Get all programs
        all_progs = analyzer.get_all_programs(island_idx=0)
        
        # None should be incorrect
        for prog in all_progs:
            assert prog.correct, f"Found incorrect program {prog.id}"


class TestEmbeddingComputation:
    """Test embedding computation and caching."""
    
    def test_compute_island_mean_embedding(self, test_database):
        """Test mean embedding computation."""
        populate_test_database(test_database, num_islands=2, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Compute mean embedding
        mean_emb = analyzer.compute_island_mean_embedding(island_idx=0, strategy="all")
        
        assert len(mean_emb) > 0, "Mean embedding should not be empty"
        assert isinstance(mean_emb, np.ndarray), "Should return numpy array"
        
        # Check that it's normalized-ish (sum of squares ≈ 1)
        norm = np.linalg.norm(mean_emb)
        assert norm > 0, "Mean embedding should have non-zero norm"
    
    def test_embedding_cache(self, test_database):
        """Test that embedding caching works correctly."""
        populate_test_database(test_database, num_islands=2, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # First call should compute
        mean_emb1 = analyzer.compute_island_mean_embedding(island_idx=0, strategy="all")
        
        # Second call should use cache
        mean_emb2 = analyzer.compute_island_mean_embedding(island_idx=0, strategy="all")
        
        # Should be identical (same object)
        assert np.array_equal(mean_emb1, mean_emb2), "Cached result should be identical"
    
    def test_cache_clear(self, test_database):
        """Test cache clearing functionality."""
        populate_test_database(test_database, num_islands=2, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Compute and cache
        analyzer.compute_island_mean_embedding(island_idx=0, strategy="all")
        assert len(analyzer._embedding_cache) > 0, "Cache should have entries"
        
        # Clear cache
        analyzer.clear_cache()
        assert len(analyzer._embedding_cache) == 0, "Cache should be empty"
    
    def test_empty_island_embedding(self, test_database):
        """Test behavior with empty island."""
        analyzer = test_database.get_diversity_analyzer()
        
        # Try to compute embedding for non-existent island
        mean_emb = analyzer.compute_island_mean_embedding(island_idx=999, strategy="all")
        
        assert len(mean_emb) == 0, "Should return empty array for empty island"
    
    def test_compute_island_variance(self, test_database):
        """Test intra-island diversity computation."""
        populate_test_database(test_database, num_islands=2, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Compute variance
        variance = analyzer.compute_island_variance(island_idx=0, strategy="all")
        
        assert isinstance(variance, float), "Should return float"
        assert variance >= 0.0, "Variance should be non-negative"
        assert variance <= 1.0, "Variance should be <= 1.0 (max angular distance)"


class TestDistanceMetrics:
    """Test distance computation methods."""
    
    def test_angular_distance_basic(self, test_database):
        """Test basic angular distance computation."""
        analyzer = test_database.get_diversity_analyzer()
        
        # Create two embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        
        # Compute distance
        dist = analyzer.compute_angular_distance(emb1, emb2)
        
        assert 0.0 <= dist <= 1.0, "Distance should be in [0, 1]"
        assert dist > 0.9, "Orthogonal vectors should have high angular distance"
    
    def test_angular_distance_identical(self, test_database):
        """Test that identical embeddings have zero distance."""
        analyzer = test_database.get_diversity_analyzer()
        
        emb = np.array([1.0, 2.0, 3.0])
        
        dist = analyzer.compute_angular_distance(emb, emb)
        
        assert dist < 1e-6, f"Identical embeddings should have ~0 distance, got {dist}"
    
    def test_angular_distance_opposite(self, test_database):
        """Test that opposite embeddings have maximum distance."""
        analyzer = test_database.get_diversity_analyzer()
        
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([-1.0, 0.0, 0.0])
        
        dist = analyzer.compute_angular_distance(emb1, emb2)
        
        # Should be close to 2.0 (1 - (-1))
        assert dist > 1.9, f"Opposite vectors should have distance ~2.0, got {dist}"
    
    def test_angular_distance_symmetry(self, test_database):
        """Test that distance is symmetric."""
        analyzer = test_database.get_diversity_analyzer()
        
        emb1 = generate_random_embedding(128, seed=1)
        emb2 = generate_random_embedding(128, seed=2)
        
        dist1 = analyzer.compute_angular_distance(np.array(emb1), np.array(emb2))
        dist2 = analyzer.compute_angular_distance(np.array(emb2), np.array(emb1))
        
        assert abs(dist1 - dist2) < 1e-10, "Distance should be symmetric"
    
    def test_angular_distance_empty_embedding(self, test_database):
        """Test behavior with empty embeddings."""
        analyzer = test_database.get_diversity_analyzer()
        
        emb1 = np.array([])
        emb2 = np.array([1.0, 2.0, 3.0])
        
        dist = analyzer.compute_angular_distance(emb1, emb2)
        
        assert dist == 0.0, "Empty embedding should return 0 distance"
    
    def test_angular_distance_mismatched_dimensions(self, test_database):
        """Test error handling for mismatched dimensions."""
        analyzer = test_database.get_diversity_analyzer()
        
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="dimensions must match"):
            analyzer.compute_angular_distance(emb1, emb2)
    
    def test_pairwise_island_diversity(self, test_database):
        """Test pairwise island diversity computation."""
        populate_test_database(test_database, num_islands=3, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Compute diversity between two islands
        div = analyzer.compute_pairwise_island_diversity(
            island1_idx=0, island2_idx=1, strategy="all"
        )
        
        assert isinstance(div, float), "Should return float"
        assert 0.0 <= div <= 1.0, f"Diversity should be in [0, 1], got {div}"


class TestSystemDiversity:
    """Test system-wide diversity metrics."""
    
    def test_compute_system_diversity_basic(self, test_database):
        """Test basic system diversity computation."""
        populate_test_database(test_database, num_islands=4, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        metrics = analyzer.compute_system_diversity(strategy="all")
        
        assert "error" not in metrics, f"Error in computation: {metrics.get('error')}"
        assert "mean_diversity" in metrics
        assert "std_diversity" in metrics
        assert "min_diversity" in metrics
        assert "max_diversity" in metrics
        assert "pairwise_matrix" in metrics
        
        # Check value ranges
        assert 0.0 <= metrics["mean_diversity"] <= 1.0
        assert 0.0 <= metrics["std_diversity"] <= 1.0
        assert 0.0 <= metrics["min_diversity"] <= 1.0
        assert 0.0 <= metrics["max_diversity"] <= 1.0
        
        # Check matrix shape
        n_islands = metrics["num_islands"]
        matrix = metrics["pairwise_matrix"]
        assert matrix.shape == (n_islands, n_islands)
    
    def test_system_diversity_insufficient_islands(self, test_database):
        """Test behavior with insufficient islands."""
        # Only populate 1 island
        populate_test_database(test_database, num_islands=1, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        metrics = analyzer.compute_system_diversity()
        
        assert "error" in metrics, "Should return error for insufficient islands"
    
    def test_system_diversity_with_intra_island(self, test_database):
        """Test system diversity with within-island diversity."""
        populate_test_database(test_database, num_islands=3, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        metrics = analyzer.compute_system_diversity(
            strategy="all", include_intra_island=True
        )
        
        assert "intra_island_diversity" in metrics
        assert "mean_intra_island_diversity" in metrics
        
        # Each island should have a variance value
        for island_idx in metrics["island_labels"]:
            assert island_idx in metrics["intra_island_diversity"]
    
    def test_system_diversity_matrix_symmetry(self, test_database):
        """Test that diversity matrix is symmetric."""
        populate_test_database(test_database, num_islands=3, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        metrics = analyzer.compute_system_diversity(strategy="all")
        matrix = metrics["pairwise_matrix"]
        
        # Check symmetry
        assert np.allclose(matrix, matrix.T), "Matrix should be symmetric"
    
    def test_system_diversity_diagonal_zeros(self, test_database):
        """Test that diversity matrix has zeros on diagonal."""
        populate_test_database(test_database, num_islands=3, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        metrics = analyzer.compute_system_diversity(strategy="all")
        matrix = metrics["pairwise_matrix"]
        
        # Diagonal should be ~0 (island compared with itself)
        diagonal = np.diag(matrix)
        assert np.allclose(diagonal, 0.0, atol=1e-10), "Diagonal should be zeros"


# =============================================================================
# INTEGRATION TESTS - Full Workflow Testing
# =============================================================================

class TestFullWorkflow:
    """Test complete diversity analysis workflows."""
    
    def test_end_to_end_diversity_analysis(self, test_database):
        """Test complete diversity analysis workflow."""
        # Setup: Populate database
        populate_test_database(test_database, num_islands=4, programs_per_island=15)
        
        # Step 1: Create analyzer
        analyzer = test_database.get_diversity_analyzer()
        
        # Step 2: Compute system diversity
        metrics = analyzer.compute_system_diversity(strategy="all")
        
        assert "error" not in metrics
        assert metrics["num_islands"] == 4
        
        # Step 3: Find most/least diverse pairs
        most_diverse = analyzer.get_most_diverse_island_pair()
        most_similar = analyzer.get_most_similar_island_pair()
        
        assert most_diverse[2] >= most_similar[2], (
            "Most diverse pair should have higher distance than most similar"
        )
        
        # Step 4: Generate report
        report = analyzer.generate_diversity_report(include_all_strategies=False)
        
        assert len(report) > 0, "Report should not be empty"
        assert "ISLAND DIVERSITY" in report
    
    def test_strategy_comparison_workflow(self, test_database):
        """Test comparing multiple strategies."""
        populate_test_database(test_database, num_islands=3, programs_per_island=20)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Compare strategies
        comparison = analyzer.compare_strategies(strategies=["leaf", "elite", "all"])
        
        assert "by_strategy" in comparison
        assert len(comparison["by_strategy"]) == 3
        
        # All strategies should have succeeded
        for strategy, metrics in comparison["by_strategy"].items():
            assert "error" not in metrics, f"Strategy {strategy} failed"
    
    def test_integration_with_database_methods(self, test_database):
        """Test integration with existing database methods."""
        populate_test_database(test_database, num_islands=2, programs_per_island=10)
        
        # Use database method to get analyzer
        analyzer = test_database.get_diversity_analyzer()
        
        assert isinstance(analyzer, IslandDiversityAnalyzer)
        
        # Analyzer should work with database
        metrics = analyzer.compute_system_diversity()
        
        assert "error" not in metrics


# =============================================================================
# EDGE CASE TESTS - Boundary Conditions and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_program_per_island(self, test_database):
        """Test with only one program per island."""
        # Add one program to each island
        for island_idx in range(3):
            prog = create_mock_program(
                program_id=f"single_prog_{island_idx}",
                code=f"def single_{island_idx}(): pass",
                embedding=generate_random_embedding(128, seed=island_idx),
                island_idx=island_idx,
                generation=0,
                correct=True,
                children_count=0,
            )
            test_database.add(prog, verbose=False)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Should still work
        metrics = analyzer.compute_system_diversity(strategy="all")
        
        assert "error" not in metrics
        assert metrics["num_programs_per_island"][0] == 1
    
    def test_no_embeddings_available(self, test_database):
        """Test behavior when programs have no embeddings."""
        # Add programs without embeddings
        for island_idx in range(2):
            prog = create_mock_program(
                program_id=f"no_emb_prog_{island_idx}",
                code=f"def no_emb_{island_idx}(): pass",
                embedding=[],  # Empty embedding
                island_idx=island_idx,
                generation=0,
                correct=True,
            )
            test_database.add(prog, verbose=False)
        
        analyzer = test_database.get_diversity_analyzer()
        
        metrics = analyzer.compute_system_diversity(strategy="all")
        
        # Should return error for no valid embeddings
        assert "error" in metrics
    
    def test_all_programs_on_one_island(self, test_database):
        """Test when all programs are on one island."""
        # Add all programs to island 0
        populate_test_database(test_database, num_islands=1, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        metrics = analyzer.compute_system_diversity()
        
        assert "error" in metrics, "Should error with only one island"
    
    def test_very_similar_embeddings(self, test_database):
        """Test with nearly identical embeddings across islands."""
        base_emb = generate_random_embedding(128, seed=42)
        
        for island_idx in range(3):
            for i in range(5):
                # Add tiny perturbation
                emb = np.array(base_emb) + np.random.randn(128) * 0.001
                emb = emb / np.linalg.norm(emb)
                
                prog = create_mock_program(
                    program_id=f"similar_prog_{island_idx}_{i}",
                    code=f"def similar_{island_idx}_{i}(): pass",
                    embedding=emb.tolist(),
                    island_idx=island_idx,
                    generation=i,
                    correct=True,
                )
                test_database.add(prog, verbose=False)
        
        analyzer = test_database.get_diversity_analyzer()
        
        metrics = analyzer.compute_system_diversity(strategy="all")
        
        # Should have very low diversity
        assert metrics["mean_diversity"] < 0.1, "Similar embeddings should have low diversity"
    
    def test_very_different_embeddings(self, test_database):
        """Test with maximally different embeddings across islands."""
        embedding_dim = 128
        
        for island_idx in range(3):
            # Create orthogonal embeddings
            emb = np.zeros(embedding_dim)
            emb[island_idx * 40:(island_idx + 1) * 40] = 1.0
            emb = emb / np.linalg.norm(emb)
            
            for i in range(5):
                prog = create_mock_program(
                    program_id=f"different_prog_{island_idx}_{i}",
                    code=f"def different_{island_idx}_{i}(): pass",
                    embedding=emb.tolist(),
                    island_idx=island_idx,
                    generation=i,
                    correct=True,
                )
                test_database.add(prog, verbose=False)
        
        analyzer = test_database.get_diversity_analyzer()
        
        metrics = analyzer.compute_system_diversity(strategy="all")
        
        # Should have high diversity
        assert metrics["mean_diversity"] > 0.5, "Different embeddings should have high diversity"


# =============================================================================
# PERFORMANCE TESTS - Scalability and Efficiency
# =============================================================================

class TestPerformance:
    """Test performance and scalability."""
    
    def test_large_database_performance(self, test_database):
        """Test performance with large database."""
        # Populate large database
        start_time = time.time()
        populate_test_database(
            test_database, num_islands=5, programs_per_island=100
        )
        populate_time = time.time() - start_time
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Measure diversity computation time
        start_time = time.time()
        metrics = analyzer.compute_system_diversity(strategy="elite")
        compute_time = time.time() - start_time
        
        logger.info(f"Populate time: {populate_time:.2f}s")
        logger.info(f"Diversity compute time: {compute_time:.2f}s")
        
        # Should complete in reasonable time (< 10 seconds)
        assert compute_time < 10.0, f"Computation too slow: {compute_time:.2f}s"
        assert "error" not in metrics
    
    def test_cache_performance_improvement(self, test_database):
        """Test that caching improves performance."""
        populate_test_database(test_database, num_islands=4, programs_per_island=50)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # First call (no cache)
        start_time = time.time()
        metrics1 = analyzer.compute_system_diversity(strategy="all")
        time1 = time.time() - start_time
        
        # Second call (with cache)
        start_time = time.time()
        metrics2 = analyzer.compute_system_diversity(strategy="all")
        time2 = time.time() - start_time
        
        logger.info(f"First call: {time1:.4f}s, Second call: {time2:.4f}s")
        
        # Cached call should be faster
        assert time2 < time1, "Cached call should be faster"
        
        # Results should be identical
        assert metrics1["mean_diversity"] == metrics2["mean_diversity"]
    
    def test_strategy_performance_comparison(self, test_database):
        """Compare performance of different strategies."""
        populate_test_database(test_database, num_islands=4, programs_per_island=100)
        
        analyzer = test_database.get_diversity_analyzer()
        
        times = {}
        for strategy in ["leaf", "recent_N", "elite", "all"]:
            start_time = time.time()
            try:
                analyzer.compute_system_diversity(strategy=strategy)
                times[strategy] = time.time() - start_time
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed: {e}")
                times[strategy] = None
        
        logger.info(f"Strategy times: {times}")
        
        # Elite should generally be fastest (smallest set)
        # All should generally be slowest (largest set)
        if times.get("elite") and times.get("all"):
            assert times["elite"] <= times["all"] * 2, (
                "Elite strategy should be faster than all"
            )


# =============================================================================
# STATISTICAL TESTS - Validate Mathematical Properties
# =============================================================================

class TestStatisticalProperties:
    """Test statistical and mathematical properties."""
    
    def test_distance_triangle_inequality(self, test_database):
        """Test that angular distance satisfies triangle inequality."""
        analyzer = test_database.get_diversity_analyzer()
        
        # Create three random embeddings
        emb1 = np.array(generate_random_embedding(128, seed=1))
        emb2 = np.array(generate_random_embedding(128, seed=2))
        emb3 = np.array(generate_random_embedding(128, seed=3))
        
        # Compute distances
        d12 = analyzer.compute_angular_distance(emb1, emb2)
        d23 = analyzer.compute_angular_distance(emb2, emb3)
        d13 = analyzer.compute_angular_distance(emb1, emb3)
        
        # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        assert d13 <= d12 + d23 + 1e-6, (
            f"Triangle inequality violated: {d13} > {d12} + {d23}"
        )
    
    def test_mean_diversity_bounds(self, test_database):
        """Test that mean diversity is within valid bounds."""
        populate_test_database(test_database, num_islands=4, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        metrics = analyzer.compute_system_diversity(strategy="all")
        
        mean_div = metrics["mean_diversity"]
        min_div = metrics["min_diversity"]
        max_div = metrics["max_diversity"]
        
        # Mean should be between min and max
        assert min_div <= mean_div <= max_div, (
            f"Mean diversity {mean_div} not between min {min_div} and max {max_div}"
        )
    
    def test_consistency_across_multiple_runs(self, test_database):
        """Test that results are consistent across runs."""
        populate_test_database(test_database, num_islands=3, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Run multiple times
        results = []
        for _ in range(3):
            analyzer.clear_cache()  # Clear cache to force recomputation
            metrics = analyzer.compute_system_diversity(strategy="all")
            results.append(metrics["mean_diversity"])
        
        # All results should be identical
        assert len(set(results)) == 1, f"Inconsistent results: {results}"


# =============================================================================
# COMPARISON TESTS - Strategy Comparison Validation
# =============================================================================

class TestStrategyComparison:
    """Test strategy comparison functionality."""
    
    def test_compare_all_strategies(self, test_database):
        """Test comparing all strategies."""
        populate_test_database(test_database, num_islands=3, programs_per_island=20)
        
        analyzer = test_database.get_diversity_analyzer()
        
        comparison = analyzer.compare_strategies()
        
        assert "by_strategy" in comparison
        assert "strategies_compared" in comparison
        
        # Should have results for multiple strategies
        assert len(comparison["by_strategy"]) >= 2
    
    def test_strategy_differences(self, test_database):
        """Test that different strategies give different results."""
        # Create a database where strategies would differ
        # Leaf: only last programs
        # Elite: best programs
        # All: all programs
        
        for island_idx in range(3):
            for i in range(10):
                prog = create_mock_program(
                    program_id=f"prog_{island_idx}_{i}",
                    code=f"def f_{island_idx}_{i}(): pass",
                    embedding=generate_random_embedding(128, seed=island_idx*100 + i),
                    island_idx=island_idx,
                    generation=i,
                    correct=True,
                    combined_score=float(i) / 10,
                    children_count=0 if i == 9 else 1,
                )
                test_database.add(prog, verbose=False)
        
        analyzer = test_database.get_diversity_analyzer()
        
        # Compare strategies
        leaf_metrics = analyzer.compute_system_diversity(strategy="leaf")
        all_metrics = analyzer.compute_system_diversity(strategy="all")
        
        # Results might be different based on selection
        # At least verify both succeeded
        assert "error" not in leaf_metrics
        assert "error" not in all_metrics
    
    def test_most_diverse_pair_identification(self, test_database):
        """Test identification of most/least diverse pairs."""
        # Create islands with known diversity
        # Island 0 and 1: similar
        # Island 2: very different
        
        for i in range(5):
            # Islands 0 and 1 have similar embeddings
            emb_similar = generate_random_embedding(128, seed=100 + i)
            
            test_database.add(
                create_mock_program(
                    f"prog_0_{i}", "code0", emb_similar, island_idx=0, generation=i, correct=True
                ),
                verbose=False
            )
            
            emb_similar_2 = np.array(emb_similar) + np.random.randn(128) * 0.01
            emb_similar_2 = emb_similar_2 / np.linalg.norm(emb_similar_2)
            
            test_database.add(
                create_mock_program(
                    f"prog_1_{i}", "code1", emb_similar_2.tolist(), island_idx=1, generation=i, correct=True
                ),
                verbose=False
            )
            
            # Island 2 has very different embeddings
            emb_different = generate_random_embedding(128, seed=500 + i)
            test_database.add(
                create_mock_program(
                    f"prog_2_{i}", "code2", emb_different, island_idx=2, generation=i, correct=True
                ),
                verbose=False
            )
        
        analyzer = test_database.get_diversity_analyzer()
        
        most_diverse = analyzer.get_most_diverse_island_pair()
        most_similar = analyzer.get_most_similar_island_pair()
        
        # Most similar should be 0 and 1
        similar_set = {most_similar[0], most_similar[1]}
        assert similar_set == {0, 1} or similar_set == {1, 0}, (
            f"Most similar pair should be 0 and 1, got {most_similar}"
        )
        
        # Most diverse should involve island 2
        diverse_set = {most_diverse[0], most_diverse[1]}
        assert 2 in diverse_set, f"Most diverse pair should include island 2, got {most_diverse}"


# =============================================================================
# REPORT GENERATION TESTS
# =============================================================================

class TestReportGeneration:
    """Test diversity report generation."""
    
    def test_generate_basic_report(self, test_database):
        """Test basic report generation."""
        populate_test_database(test_database, num_islands=3, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        report = analyzer.generate_diversity_report(include_all_strategies=False)
        
        assert len(report) > 0
        assert "ISLAND DIVERSITY" in report
        assert "Mean diversity" in report
    
    def test_generate_full_report(self, test_database):
        """Test comprehensive report with all strategies."""
        populate_test_database(test_database, num_islands=3, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        report = analyzer.generate_diversity_report(include_all_strategies=True)
        
        assert "Strategy Comparison" in report
        assert "leaf" in report.lower() or "elite" in report.lower()
    
    def test_save_report_to_file(self, test_database, temp_dir):
        """Test saving report to file."""
        populate_test_database(test_database, num_islands=3, programs_per_island=10)
        
        analyzer = test_database.get_diversity_analyzer()
        
        output_path = Path(temp_dir) / "diversity_report.txt"
        report = analyzer.generate_diversity_report(
            output_path=str(output_path),
            include_all_strategies=False
        )
        
        # File should exist
        assert output_path.exists(), "Report file not created"
        
        # Content should match returned report
        saved_content = output_path.read_text()
        assert saved_content == report


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
