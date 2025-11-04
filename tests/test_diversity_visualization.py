"""
Visualization Tests for Island Diversity

This module tests visualization functions for diversity metrics.
Requires matplotlib and seaborn for plot generation.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from shinka.database import (
    ProgramDatabase,
    Program,
    DatabaseConfig,
    IslandDiversityAnalyzer,
    DiversityConfig,
)


# Skip all tests if plotting libraries not available
pytestmark = pytest.mark.skipif(
    not PLOTTING_AVAILABLE,
    reason="Matplotlib/seaborn not available"
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def test_database(temp_dir):
    """Create a test database with sample programs."""
    db_config = DatabaseConfig(
        db_path=str(Path(temp_dir) / "test_db.sqlite"),
        num_islands=4,
        archive_size=20,
    )
    
    db = ProgramDatabase(config=db_config, embedding_model="text-embedding-3-small")
    
    # Populate with test data
    from tests.test_diversity import populate_test_database
    populate_test_database(db, num_islands=4, programs_per_island=15)
    
    yield db
    db.close()


class TestDiversityVisualization:
    """Test diversity visualization functions."""
    
    def test_plot_diversity_heatmap(self, test_database, temp_dir):
        """Test diversity heatmap generation."""
        analyzer = test_database.get_diversity_analyzer()
        
        # Compute diversity
        metrics = analyzer.compute_system_diversity()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        matrix = metrics["pairwise_matrix"]
        islands = metrics["island_labels"]
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            xticklabels=[f"I{i}" for i in islands],
            yticklabels=[f"I{i}" for i in islands],
            ax=ax,
            cbar_kws={"label": "Angular Distance"}
        )
        
        ax.set_title("Island Diversity Matrix")
        
        # Save plot
        output_path = Path(temp_dir) / "diversity_heatmap.png"
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Verify file exists
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_plot_diversity_comparison(self, test_database, temp_dir):
        """Test strategy comparison bar plot."""
        analyzer = test_database.get_diversity_analyzer()
        
        # Compare strategies
        comparison = analyzer.compare_strategies(strategies=["leaf", "elite", "all"])
        
        # Extract mean diversity values
        strategies = []
        mean_divs = []
        
        for strategy, metrics in comparison["by_strategy"].items():
            if "error" not in metrics:
                strategies.append(strategy)
                mean_divs.append(metrics["mean_diversity"])
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars = ax.bar(strategies, mean_divs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel("Mean Diversity")
        ax.set_xlabel("Selection Strategy")
        ax.set_title("Diversity by Selection Strategy")
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, val in zip(bars, mean_divs):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom'
            )
        
        # Save plot
        output_path = Path(temp_dir) / "strategy_comparison.png"
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
