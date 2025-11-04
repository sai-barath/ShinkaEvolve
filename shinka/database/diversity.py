"""
Island Diversity Measurement Module

This module provides tools for measuring program diversity within and across
evolutionary islands in ShinkaEvolve. It supports multiple program selection
strategies and computes both pairwise and system-wide diversity metrics.

Key Features:
- Multiple program selection strategies (leaf nodes, recent generations, elites)
- Angular distance computation between island embeddings
- System-wide diversity metrics
- Visualization and analysis tools
"""

import logging
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DiversityConfig:
    """Configuration for diversity measurement."""
    
    # Selection strategy: "leaf" | "recent_N" | "elite" | "all"
    default_strategy: str = "elite"
    
    # Number of recent generations to consider for "recent_N" strategy
    recent_generations: int = 10
    
    # Minimum number of programs required per island for diversity computation
    min_programs_per_island: int = 1
    
    # Whether to include only correct programs
    correct_only: bool = True
    
    # Logging and monitoring
    log_interval: int = 5  # Log diversity every N generations
    verbose: bool = False


class IslandDiversityAnalyzer:
    """
    Measures program diversity within and across evolutionary islands.
    
    This class provides methods to:
    1. Select representative programs from islands using various strategies
    2. Compute mean embeddings for each island
    3. Measure pairwise diversity between islands using angular distance
    4. Compute system-wide diversity metrics
    5. Track diversity evolution over time
    
    Example:
        >>> analyzer = IslandDiversityAnalyzer(database, config)
        >>> system_metrics = analyzer.compute_system_diversity(strategy="elite")
        >>> print(f"Mean diversity: {system_metrics['mean_diversity']:.3f}")
    """
    
    def __init__(self, database, config: Optional[DiversityConfig] = None):
        """
        Initialize the diversity analyzer.
        
        Args:
            database: ProgramDatabase instance
            config: DiversityConfig instance (optional)
        """
        self.db = database
        self.config = config or DiversityConfig()
        self.cursor = database.cursor
        self.conn = database.conn
        
        # Cache for computed embeddings to avoid redundant computation
        self._embedding_cache: Dict[Tuple[int, str], np.ndarray] = {}
        self._cache_generation: Optional[int] = None
    
    def clear_cache(self):
        """Clear the embedding cache. Call when database is updated."""
        self._embedding_cache.clear()
        self._cache_generation = None
        logger.debug("Diversity analyzer cache cleared")
    
    # =========================================================================
    # Program Selection Strategies
    # =========================================================================
    
    def get_leaf_programs(self, island_idx: int) -> List[Any]:
        """
        Get programs with no children (leaf nodes in evolution tree).
        
        These represent the current frontier of evolution on this island.
        Leaf nodes are programs that haven't been used as parents yet,
        indicating they are the most recent successful variants.
        
        Args:
            island_idx: Island identifier
            
        Returns:
            List of Program objects that are leaf nodes
            
        Example:
            >>> leaf_programs = analyzer.get_leaf_programs(island_idx=0)
            >>> print(f"Island 0 has {len(leaf_programs)} leaf programs")
        """
        if not self.cursor:
            raise ConnectionError("Database cursor not available")
        
        query = """
            SELECT * FROM programs 
            WHERE island_idx = ? 
            AND children_count = 0
        """
        
        if self.config.correct_only:
            query += " AND correct = 1"
        
        self.cursor.execute(query, (island_idx,))
        rows = self.cursor.fetchall()
        
        programs = [self.db._program_from_row(row) for row in rows]
        programs = [p for p in programs if p is not None and p.embedding]
        
        logger.debug(
            f"Selected {len(programs)} leaf programs from island {island_idx}"
        )
        return programs
    
    def get_recent_generation_programs(
        self, island_idx: int, num_generations: Optional[int] = None
    ) -> List[Any]:
        """
        Get programs from the latest N generations.
        
        This strategy is more efficient than using all programs and captures
        recent evolutionary trends. It's useful when you want a temporal
        snapshot of the island's recent progress.
        
        Args:
            island_idx: Island identifier
            num_generations: Number of recent generations (defaults to config value)
            
        Returns:
            List of Program objects from recent generations
            
        Example:
            >>> recent_progs = analyzer.get_recent_generation_programs(
            ...     island_idx=0, num_generations=5
            ... )
        """
        if not self.cursor:
            raise ConnectionError("Database cursor not available")
        
        n_gens = num_generations or self.config.recent_generations
        
        # Get the maximum generation in the database
        self.cursor.execute(
            "SELECT MAX(generation) as max_gen FROM programs WHERE island_idx = ?",
            (island_idx,)
        )
        result = self.cursor.fetchone()
        max_gen = result["max_gen"] if result and result["max_gen"] else 0
        
        min_gen = max(0, max_gen - n_gens + 1)
        
        query = """
            SELECT * FROM programs 
            WHERE island_idx = ? 
            AND generation >= ?
        """
        
        if self.config.correct_only:
            query += " AND correct = 1"
        
        self.cursor.execute(query, (island_idx, min_gen))
        rows = self.cursor.fetchall()
        
        programs = [self.db._program_from_row(row) for row in rows]
        programs = [p for p in programs if p is not None and p.embedding]
        
        logger.debug(
            f"Selected {len(programs)} programs from generations "
            f"{min_gen}-{max_gen} on island {island_idx}"
        )
        return programs
    
    def get_elite_programs(self, island_idx: int) -> List[Any]:
        """
        Get elite programs from archive for this island.
        
        This is the most efficient strategy as it focuses on the best
        performing programs. Elite programs are in the archive, representing
        the highest quality solutions found so far.
        
        Args:
            island_idx: Island identifier
            
        Returns:
            List of elite Program objects from the archive
            
        Example:
            >>> elites = analyzer.get_elite_programs(island_idx=0)
            >>> avg_score = np.mean([p.combined_score for p in elites])
        """
        if not self.cursor:
            raise ConnectionError("Database cursor not available")
        
        query = """
            SELECT p.* FROM programs p 
            JOIN archive a ON p.id = a.program_id
            WHERE p.island_idx = ?
        """
        
        if self.config.correct_only:
            query += " AND p.correct = 1"
        
        self.cursor.execute(query, (island_idx,))
        rows = self.cursor.fetchall()
        
        programs = [self.db._program_from_row(row) for row in rows]
        programs = [p for p in programs if p is not None and p.embedding]
        
        logger.debug(
            f"Selected {len(programs)} elite programs from island {island_idx}"
        )
        return programs
    
    def get_all_programs(self, island_idx: int) -> List[Any]:
        """
        Get all programs from an island.
        
        This strategy uses all programs, which can be computationally expensive
        for large databases but provides the most comprehensive diversity measure.
        
        Args:
            island_idx: Island identifier
            
        Returns:
            List of all Program objects on the island
        """
        if not self.cursor:
            raise ConnectionError("Database cursor not available")
        
        query = "SELECT * FROM programs WHERE island_idx = ?"
        
        if self.config.correct_only:
            query += " AND correct = 1"
        
        self.cursor.execute(query, (island_idx,))
        rows = self.cursor.fetchall()
        
        programs = [self.db._program_from_row(row) for row in rows]
        programs = [p for p in programs if p is not None and p.embedding]
        
        logger.debug(
            f"Selected {len(programs)} total programs from island {island_idx}"
        )
        return programs
    
    def _select_programs(self, island_idx: int, strategy: str) -> List[Any]:
        """
        Select programs based on the specified strategy.
        
        Args:
            island_idx: Island identifier
            strategy: Selection strategy ("leaf" | "recent_N" | "elite" | "all")
            
        Returns:
            List of selected Program objects
            
        Raises:
            ValueError: If strategy is not recognized
        """
        strategy_lower = strategy.lower()
        
        if strategy_lower == "leaf":
            return self.get_leaf_programs(island_idx)
        elif strategy_lower == "recent_n":
            return self.get_recent_generation_programs(island_idx)
        elif strategy_lower == "elite":
            return self.get_elite_programs(island_idx)
        elif strategy_lower == "all":
            return self.get_all_programs(island_idx)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Must be one of: leaf, recent_N, elite, all"
            )
    
    # =========================================================================
    # Island-Level Metrics
    # =========================================================================
    
    def compute_island_mean_embedding(
        self, island_idx: int, strategy: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute mean embedding vector for an island's programs.
        
        The mean embedding represents the semantic centroid of the island's
        program population. It's used to characterize the overall "location"
        of the island in embedding space.
        
        Args:
            island_idx: Island identifier
            strategy: Selection strategy (defaults to config value)
            
        Returns:
            Mean embedding vector as numpy array. Empty array if no programs.
            
        Example:
            >>> mean_emb = analyzer.compute_island_mean_embedding(0, "elite")
            >>> print(f"Embedding dimension: {len(mean_emb)}")
        """
        strategy = strategy or self.config.default_strategy
        
        # Check cache
        cache_key = (island_idx, strategy)
        if cache_key in self._embedding_cache:
            logger.debug(f"Using cached embedding for island {island_idx}")
            return self._embedding_cache[cache_key]
        
        programs = self._select_programs(island_idx, strategy)
        
        if not programs:
            logger.warning(
                f"No programs found for island {island_idx} with strategy '{strategy}'"
            )
            return np.array([])
        
        # Extract embeddings
        embeddings = []
        for prog in programs:
            if prog.embedding and len(prog.embedding) > 0:
                embeddings.append(np.array(prog.embedding))
        
        if not embeddings:
            logger.warning(
                f"No valid embeddings found for island {island_idx} "
                f"with strategy '{strategy}'"
            )
            return np.array([])
        
        # Compute mean
        mean_embedding = np.mean(embeddings, axis=0)
        
        # Cache result
        self._embedding_cache[cache_key] = mean_embedding
        
        logger.debug(
            f"Computed mean embedding for island {island_idx} using {len(embeddings)} "
            f"programs (strategy: {strategy})"
        )
        
        return mean_embedding
    
    def compute_island_variance(
        self, island_idx: int, strategy: Optional[str] = None
    ) -> float:
        """
        Compute intra-island diversity (variance in embeddings).
        
        Measures how diverse programs are within a single island.
        Higher variance indicates more diverse solutions on the island.
        
        Args:
            island_idx: Island identifier
            strategy: Selection strategy (defaults to config value)
            
        Returns:
            Mean pairwise angular distance within the island
        """
        strategy = strategy or self.config.default_strategy
        programs = self._select_programs(island_idx, strategy)
        
        if len(programs) < 2:
            return 0.0
        
        embeddings = []
        for prog in programs:
            if prog.embedding and len(prog.embedding) > 0:
                embeddings.append(np.array(prog.embedding))
        
        if len(embeddings) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = self.compute_angular_distance(embeddings[i], embeddings[j])
                distances.append(dist)
        
        return float(np.mean(distances))
    
    # =========================================================================
    # Distance Metrics
    # =========================================================================
    
    def compute_angular_distance(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute angular distance between two embeddings.
        
        Angular distance is defined as: 1 - cosine_similarity
        This provides an intuitive measure where:
        - 0.0 = identical embeddings (no diversity)
        - 1.0 = maximally different embeddings (high diversity)
        
        The computation reuses the existing cosine similarity logic from
        ProgramDatabase to ensure consistency.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Angular distance in range [0.0, 1.0]
            
        Example:
            >>> dist = analyzer.compute_angular_distance(emb1, emb2)
            >>> if dist < 0.1:
            ...     print("Islands are very similar")
            >>> elif dist > 0.5:
            ...     print("Islands are highly diverse")
        """
        if len(embedding1) == 0 or len(embedding2) == 0:
            logger.warning("Cannot compute distance with empty embedding")
            return 0.0
        
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimensions must match: {len(embedding1)} vs {len(embedding2)}"
            )
        
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            logger.warning("Cannot compute distance with zero-norm embedding")
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Clamp to [-1, 1] to handle numerical errors
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        # Angular distance = 1 - cosine_similarity
        angular_dist = 1.0 - cosine_sim
        
        return float(angular_dist)
    
    def compute_pairwise_island_diversity(
        self, island1_idx: int, island2_idx: int, strategy: Optional[str] = None
    ) -> float:
        """
        Measure diversity between two islands using angular distance.
        
        This computes the angular distance between the mean embeddings of
        two islands, providing a measure of how different the two island
        populations are from each other.
        
        Args:
            island1_idx: First island identifier
            island2_idx: Second island identifier
            strategy: Selection strategy (defaults to config value)
            
        Returns:
            Angular distance between islands in range [0.0, 1.0]
            - Close to 0.0: Low diversity (similar islands)
            - Close to 1.0: High diversity (different islands)
            
        Example:
            >>> diversity = analyzer.compute_pairwise_island_diversity(0, 1)
            >>> print(f"Islands 0 and 1 diversity: {diversity:.3f}")
        """
        strategy = strategy or self.config.default_strategy
        
        emb1 = self.compute_island_mean_embedding(island1_idx, strategy)
        emb2 = self.compute_island_mean_embedding(island2_idx, strategy)
        
        if len(emb1) == 0 or len(emb2) == 0:
            logger.warning(
                f"Cannot compute diversity between islands {island1_idx} and "
                f"{island2_idx}: missing embeddings"
            )
            return 0.0
        
        distance = self.compute_angular_distance(emb1, emb2)
        
        logger.debug(
            f"Diversity between islands {island1_idx} and {island2_idx}: "
            f"{distance:.4f} (strategy: {strategy})"
        )
        
        return distance
    
    # =========================================================================
    # System-Wide Metrics
    # =========================================================================
    
    def compute_system_diversity(
        self, strategy: Optional[str] = None, include_intra_island: bool = False
    ) -> Dict[str, Any]:
        """
        Compute overall diversity across all islands in the system.
        
        This provides a comprehensive view of diversity by computing all
        pairwise island distances and summarizing them with statistics.
        
        Args:
            strategy: Selection strategy (defaults to config value)
            include_intra_island: Whether to compute within-island diversity
            
        Returns:
            Dictionary containing:
                - mean_diversity: Average of all pairwise distances
                - std_diversity: Standard deviation of pairwise distances
                - min_diversity: Most similar island pair
                - max_diversity: Most different island pair
                - pairwise_matrix: Full distance matrix (n_islands x n_islands)
                - island_embeddings: Mean embeddings for each island
                - island_labels: List of island indices
                - strategy_used: Strategy that was used
                - num_programs_per_island: Program counts per island
                - intra_island_diversity: Within-island diversity (optional)
                
        Example:
            >>> metrics = analyzer.compute_system_diversity(strategy="elite")
            >>> print(f"System mean diversity: {metrics['mean_diversity']:.3f}")
            >>> print(f"Most diverse pair has distance: {metrics['max_diversity']:.3f}")
        """
        strategy = strategy or self.config.default_strategy
        
        # Get initialized islands
        initialized_islands = self.db.island_manager.get_initialized_islands()
        
        if len(initialized_islands) < 2:
            return {
                "error": "Need at least 2 initialized islands for diversity computation",
                "num_islands": len(initialized_islands),
                "initialized_islands": initialized_islands
            }
        
        logger.info(
            f"Computing system diversity for {len(initialized_islands)} islands "
            f"using strategy '{strategy}'"
        )
        
        # Compute mean embeddings for all islands
        island_embeddings = {}
        num_programs_per_island = {}
        
        for island_idx in initialized_islands:
            programs = self._select_programs(island_idx, strategy)
            num_programs_per_island[island_idx] = len(programs)
            
            if len(programs) < self.config.min_programs_per_island:
                logger.warning(
                    f"Island {island_idx} has only {len(programs)} programs "
                    f"(minimum: {self.config.min_programs_per_island})"
                )
            
            island_embeddings[island_idx] = self.compute_island_mean_embedding(
                island_idx, strategy
            )
        
        # Filter out islands with no valid embeddings
        valid_islands = [
            idx for idx in initialized_islands
            if len(island_embeddings[idx]) > 0
        ]
        
        if len(valid_islands) < 2:
            return {
                "error": "Need at least 2 islands with valid embeddings",
                "num_islands": len(initialized_islands),
                "valid_islands": len(valid_islands),
                "initialized_islands": initialized_islands
            }
        
        # Compute pairwise distance matrix
        n = len(valid_islands)
        distance_matrix = np.zeros((n, n))
        pairwise_distances = []
        
        for i in range(n):
            for j in range(i + 1, n):
                island_i = valid_islands[i]
                island_j = valid_islands[j]
                
                distance = self.compute_angular_distance(
                    island_embeddings[island_i],
                    island_embeddings[island_j]
                )
                
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                pairwise_distances.append(distance)
        
        # Compute statistics
        result = {
            "mean_diversity": float(np.mean(pairwise_distances)),
            "std_diversity": float(np.std(pairwise_distances)),
            "min_diversity": float(np.min(pairwise_distances)),
            "max_diversity": float(np.max(pairwise_distances)),
            "median_diversity": float(np.median(pairwise_distances)),
            "pairwise_matrix": distance_matrix,
            "island_embeddings": island_embeddings,
            "island_labels": valid_islands,
            "strategy_used": strategy,
            "num_programs_per_island": num_programs_per_island,
            "num_islands": len(valid_islands),
            "num_island_pairs": len(pairwise_distances)
        }
        
        # Optionally compute within-island diversity
        if include_intra_island:
            intra_island_diversity = {}
            for island_idx in valid_islands:
                intra_island_diversity[island_idx] = self.compute_island_variance(
                    island_idx, strategy
                )
            result["intra_island_diversity"] = intra_island_diversity
            result["mean_intra_island_diversity"] = float(
                np.mean(list(intra_island_diversity.values()))
            )
        
        logger.info(
            f"System diversity computed: mean={result['mean_diversity']:.4f}, "
            f"std={result['std_diversity']:.4f}, "
            f"range=[{result['min_diversity']:.4f}, {result['max_diversity']:.4f}]"
        )
        
        return result
    
    def compute_diversity_over_time(
        self,
        generation_checkpoints: Optional[List[int]] = None,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track how system diversity evolves across generations.
        
        This is useful for monitoring whether islands are converging (decreasing
        diversity) or diverging (increasing diversity) over time.
        
        Args:
            generation_checkpoints: List of generations to analyze (None = all)
            strategy: Selection strategy (defaults to config value)
            
        Returns:
            Dictionary with generation-indexed diversity metrics
            
        Example:
            >>> time_series = analyzer.compute_diversity_over_time(
            ...     generation_checkpoints=[0, 5, 10, 15, 20]
            ... )
            >>> for gen, metrics in time_series['by_generation'].items():
            ...     print(f"Gen {gen}: diversity = {metrics['mean_diversity']:.3f}")
        """
        strategy = strategy or self.config.default_strategy
        
        # Get current max generation
        self.cursor.execute("SELECT MAX(generation) as max_gen FROM programs")
        result = self.cursor.fetchone()
        max_gen = result["max_gen"] if result and result["max_gen"] else 0
        
        if generation_checkpoints is None:
            # Use all generations with a reasonable sampling interval
            interval = max(1, max_gen // 20)
            generation_checkpoints = list(range(0, max_gen + 1, interval))
        
        logger.info(
            f"Computing diversity over time for {len(generation_checkpoints)} checkpoints"
        )
        
        diversity_by_generation = {}
        
        for gen in generation_checkpoints:
            # Temporarily modify strategy to include only programs up to this generation
            # This requires a custom query
            logger.debug(f"Computing diversity for generation {gen}")
            
            # Save original programs and restore after computation
            # For now, we'll just note this limitation
            # TODO: Implement time-windowed diversity computation
            pass
        
        # For now, return current diversity with a note
        current_diversity = self.compute_system_diversity(strategy)
        
        return {
            "note": "Time-series diversity tracking to be implemented",
            "current_generation": max_gen,
            "current_diversity": current_diversity,
            "strategy": strategy
        }
    
    # =========================================================================
    # Analysis and Reporting
    # =========================================================================
    
    def compare_strategies(
        self, strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare diversity metrics across different selection strategies.
        
        This helps understand how the choice of strategy affects the
        measured diversity values.
        
        Args:
            strategies: List of strategies to compare (None = all strategies)
            
        Returns:
            Dictionary with per-strategy diversity metrics
            
        Example:
            >>> comparison = analyzer.compare_strategies()
            >>> for strategy, metrics in comparison['by_strategy'].items():
            ...     print(f"{strategy}: {metrics['mean_diversity']:.3f}")
        """
        if strategies is None:
            strategies = ["leaf", "recent_N", "elite"]
        
        logger.info(f"Comparing {len(strategies)} selection strategies")
        
        results = {}
        for strategy in strategies:
            try:
                results[strategy] = self.compute_system_diversity(strategy)
            except Exception as e:
                logger.error(f"Failed to compute diversity for strategy '{strategy}': {e}")
                results[strategy] = {"error": str(e)}
        
        # Summary comparison
        comparison = {
            "by_strategy": results,
            "strategies_compared": strategies
        }
        
        # Add comparative statistics if all succeeded
        if all("error" not in r for r in results.values()):
            comparison["mean_diversity_comparison"] = {
                s: results[s]["mean_diversity"] for s in strategies
            }
            comparison["max_diversity_comparison"] = {
                s: results[s]["max_diversity"] for s in strategies
            }
        
        return comparison
    
    def get_most_diverse_island_pair(
        self, strategy: Optional[str] = None
    ) -> Tuple[int, int, float]:
        """
        Find the pair of islands with maximum diversity.
        
        Args:
            strategy: Selection strategy (defaults to config value)
            
        Returns:
            Tuple of (island1_idx, island2_idx, diversity_score)
        """
        metrics = self.compute_system_diversity(strategy)
        
        if "error" in metrics:
            return (-1, -1, 0.0)
        
        matrix = metrics["pairwise_matrix"]
        islands = metrics["island_labels"]
        
        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        
        return (
            islands[max_idx[0]],
            islands[max_idx[1]],
            float(matrix[max_idx])
        )
    
    def get_most_similar_island_pair(
        self, strategy: Optional[str] = None
    ) -> Tuple[int, int, float]:
        """
        Find the pair of islands with minimum diversity (most similar).
        
        Args:
            strategy: Selection strategy (defaults to config value)
            
        Returns:
            Tuple of (island1_idx, island2_idx, diversity_score)
        """
        metrics = self.compute_system_diversity(strategy)
        
        if "error" in metrics:
            return (-1, -1, 0.0)
        
        matrix = metrics["pairwise_matrix"]
        islands = metrics["island_labels"]
        
        # Set diagonal to infinity to avoid self-comparison
        matrix_copy = matrix.copy()
        np.fill_diagonal(matrix_copy, np.inf)
        
        min_idx = np.unravel_index(np.argmin(matrix_copy), matrix_copy.shape)
        
        return (
            islands[min_idx[0]],
            islands[min_idx[1]],
            float(matrix[min_idx])
        )
    
    def generate_diversity_report(
        self,
        output_path: Optional[str] = None,
        include_all_strategies: bool = True,
        include_visualizations: bool = False
    ) -> str:
        """
        Generate a comprehensive diversity analysis report.
        
        Args:
            output_path: Path to save report (None = return as string)
            include_all_strategies: Whether to compare all strategies
            include_visualizations: Whether to generate plots (requires matplotlib)
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ISLAND DIVERSITY ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Database info
        num_islands = len(self.db.island_manager.get_initialized_islands())
        self.cursor.execute("SELECT COUNT(*) as count FROM programs")
        total_programs = self.cursor.fetchone()["count"]
        
        report_lines.append(f"Database Information:")
        report_lines.append(f"  Total programs: {total_programs}")
        report_lines.append(f"  Number of islands: {num_islands}")
        report_lines.append("")
        
        # Strategy comparison
        if include_all_strategies:
            report_lines.append("Strategy Comparison:")
            report_lines.append("-" * 80)
            comparison = self.compare_strategies()
            
            for strategy, metrics in comparison["by_strategy"].items():
                if "error" not in metrics:
                    report_lines.append(f"\n  Strategy: {strategy}")
                    report_lines.append(f"    Mean diversity: {metrics['mean_diversity']:.4f}")
                    report_lines.append(f"    Std diversity:  {metrics['std_diversity']:.4f}")
                    report_lines.append(f"    Min diversity:  {metrics['min_diversity']:.4f}")
                    report_lines.append(f"    Max diversity:  {metrics['max_diversity']:.4f}")
                    
                    # Program counts
                    prog_counts = metrics['num_programs_per_island']
                    report_lines.append(f"    Programs per island: {prog_counts}")
            report_lines.append("")
        else:
            # Just default strategy
            metrics = self.compute_system_diversity()
            if "error" not in metrics:
                report_lines.append(f"Diversity Metrics (strategy: {metrics['strategy_used']}):")
                report_lines.append(f"  Mean diversity: {metrics['mean_diversity']:.4f}")
                report_lines.append(f"  Std diversity:  {metrics['std_diversity']:.4f}")
                report_lines.append(f"  Min diversity:  {metrics['min_diversity']:.4f}")
                report_lines.append(f"  Max diversity:  {metrics['max_diversity']:.4f}")
                report_lines.append("")
        
        # Island pair analysis
        most_diverse = self.get_most_diverse_island_pair()
        most_similar = self.get_most_similar_island_pair()
        
        report_lines.append("Island Pair Analysis:")
        report_lines.append(f"  Most diverse pair: Islands {most_diverse[0]} and {most_diverse[1]} "
                           f"(distance: {most_diverse[2]:.4f})")
        report_lines.append(f"  Most similar pair: Islands {most_similar[0]} and {most_similar[1]} "
                           f"(distance: {most_similar[2]:.4f})")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report_text)
            logger.info(f"Diversity report saved to {output_path}")
        
        return report_text


def create_diversity_analyzer(database, config: Optional[DiversityConfig] = None):
    """
    Factory function to create a diversity analyzer instance.
    
    Args:
        database: ProgramDatabase instance
        config: Optional DiversityConfig
        
    Returns:
        IslandDiversityAnalyzer instance
    """
    return IslandDiversityAnalyzer(database, config)
