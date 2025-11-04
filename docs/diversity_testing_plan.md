# Island Diversity Measurement - Detailed Testing Plan

## Overview

This document outlines a comprehensive testing strategy for the Island Diversity Measurement system in ShinkaEvolve. The testing covers unit tests, integration tests, edge cases, performance benchmarks, and statistical validation.

---

## Test Organization

### Test File Structure
```
tests/
├── test_diversity.py                    # Main test suite (1000+ lines)
├── test_diversity_visualization.py      # Visualization tests
└── test_diversity_integration.py        # Integration with existing system (to be created)
```

### Test Categories

1. **Unit Tests** - Individual method testing
2. **Integration Tests** - Full workflow testing
3. **Edge Case Tests** - Boundary conditions and error handling
4. **Performance Tests** - Scalability and efficiency
5. **Statistical Tests** - Mathematical property validation
6. **Comparison Tests** - Strategy comparison validation
7. **Visualization Tests** - Plot generation and rendering

---

## Detailed Test Cases

## 1. UNIT TESTS - Individual Method Testing

### 1.1 Program Selection Strategies

#### Test 1.1.1: Leaf Program Selection
**Objective**: Verify that `get_leaf_programs()` correctly identifies programs with no children.

**Test Steps**:
1. Create database with 5 programs per island
2. Set last program as leaf (children_count=0)
3. Call `get_leaf_programs(island_idx=0)`
4. Verify exactly 1 leaf returned
5. Verify leaf has children_count=0
6. Verify leaf is on correct island

**Expected Results**:
- Returns list with 1 element
- Element has children_count=0
- Element has island_idx=0

**Edge Cases**:
- No leaf programs exist (all have children)
- All programs are leaves
- Empty island

**Success Criteria**: ✅ All assertions pass

---

#### Test 1.1.2: Recent Generation Selection
**Objective**: Verify `get_recent_generation_programs()` selects programs from latest N generations.

**Test Steps**:
1. Create database with 15 programs across generations 0-14
2. Call `get_recent_generation_programs(island_idx=0, num_generations=5)`
3. Verify all returned programs have generation >= 10
4. Verify all are on correct island

**Expected Results**:
- Returns programs from generations 10-14
- All programs on island 0
- Length matches expected count

**Edge Cases**:
- Requesting more generations than exist
- num_generations=0
- num_generations > total generations

**Success Criteria**: ✅ Generation range correct within ±1

---

#### Test 1.1.3: Elite Program Selection
**Objective**: Verify `get_elite_programs()` returns only archive programs.

**Test Steps**:
1. Populate database with programs
2. Verify some programs are in archive
3. Call `get_elite_programs(island_idx=0)`
4. Verify all returned programs are in archive
5. Verify all are correct (correct=True)

**Expected Results**:
- Returns only programs from archive table
- All programs have correct=True
- All on specified island

**Edge Cases**:
- Empty archive
- Archive with incorrect programs
- No programs on island

**Success Criteria**: ✅ All returned programs in archive

---

#### Test 1.1.4: All Programs Selection
**Objective**: Verify `get_all_programs()` returns complete island population.

**Test Steps**:
1. Create database with known number of programs per island (e.g., 10)
2. Call `get_all_programs(island_idx=0)`
3. Verify count matches expected
4. Verify all on correct island

**Expected Results**:
- Returns exactly 10 programs
- All have island_idx=0
- Includes programs from all generations

**Edge Cases**:
- Empty island
- Island with 1 program
- Island with 1000+ programs

**Success Criteria**: ✅ Count matches database query

---

#### Test 1.1.5: Invalid Strategy Error Handling
**Objective**: Verify proper error handling for invalid strategies.

**Test Steps**:
1. Call `_select_programs(island_idx=0, strategy="invalid_strategy")`
2. Expect ValueError to be raised
3. Verify error message contains "Unknown strategy"

**Expected Results**:
- Raises ValueError
- Error message is descriptive

**Success Criteria**: ✅ ValueError raised with correct message

---

#### Test 1.1.6: Correct-Only Filter
**Objective**: Verify `correct_only` config filters out incorrect programs.

**Test Steps**:
1. Add mix of correct and incorrect programs
2. Create analyzer with correct_only=True
3. Select programs using any strategy
4. Verify no incorrect programs returned

**Expected Results**:
- All returned programs have correct=True
- Incorrect programs excluded

**Success Criteria**: ✅ No incorrect programs in results

---

### 1.2 Embedding Computation

#### Test 1.2.1: Mean Embedding Computation
**Objective**: Verify mean embedding calculation is correct.

**Test Steps**:
1. Create 3 programs with known embeddings
   - emb1 = [1, 0, 0]
   - emb2 = [0, 1, 0]
   - emb3 = [0, 0, 1]
2. Compute mean: should be [1/3, 1/3, 1/3] (normalized)
3. Verify result dimensions
4. Verify result is normalized

**Expected Results**:
- Mean embedding computed correctly
- Result is numpy array
- Non-zero norm

**Mathematical Verification**:
```python
mean = (emb1 + emb2 + emb3) / 3
assert np.allclose(mean, [1/3, 1/3, 1/3])
```

**Success Criteria**: ✅ Mean within numerical precision (1e-10)

---

#### Test 1.2.2: Embedding Cache Functionality
**Objective**: Verify caching improves performance and maintains consistency.

**Test Steps**:
1. Compute mean embedding (uncached)
2. Measure time T1
3. Compute same mean embedding (cached)
4. Measure time T2
5. Verify T2 < T1
6. Verify results identical

**Expected Results**:
- Second call faster (T2 < T1)
- Results numerically identical
- Cache size increases after first call

**Success Criteria**: ✅ T2 < T1 AND results identical

---

#### Test 1.2.3: Cache Clearing
**Objective**: Verify cache can be cleared properly.

**Test Steps**:
1. Compute embeddings to populate cache
2. Verify cache not empty: `len(cache) > 0`
3. Call `clear_cache()`
4. Verify cache empty: `len(cache) == 0`
5. Recompute and verify it works

**Expected Results**:
- Cache empties completely
- Subsequent operations still work
- Next computation repopulates cache

**Success Criteria**: ✅ Cache size = 0 after clear

---

#### Test 1.2.4: Empty Island Handling
**Objective**: Verify graceful handling of empty islands.

**Test Steps**:
1. Call `compute_island_mean_embedding(island_idx=999)`
2. Verify returns empty array
3. Verify no exceptions raised
4. Verify warning logged

**Expected Results**:
- Returns np.array([])
- No exceptions
- Warning in logs

**Success Criteria**: ✅ Returns empty array without error

---

#### Test 1.2.5: Intra-Island Variance
**Objective**: Verify within-island diversity calculation.

**Test Steps**:
1. Create island with diverse programs
2. Compute `compute_island_variance()`
3. Verify result is float
4. Verify result in [0, 1]
5. Compare with manual calculation

**Expected Results**:
- Returns float value
- 0.0 ≤ variance ≤ 1.0
- Higher variance for diverse programs

**Mathematical Verification**:
```python
# For programs with embeddings e1, e2, e3:
manual_variance = mean([angular_dist(ei, ej) for i,j in pairs])
assert abs(computed_variance - manual_variance) < 1e-6
```

**Success Criteria**: ✅ Variance in valid range

---

### 1.3 Distance Metrics

#### Test 1.3.1: Basic Angular Distance
**Objective**: Verify angular distance formula is correct.

**Test Steps**:
1. Create orthogonal embeddings:
   - emb1 = [1, 0, 0]
   - emb2 = [0, 1, 0]
2. Compute distance
3. Verify distance ≈ 1.0 (for orthogonal vectors)

**Mathematical Foundation**:
```
angular_distance = 1 - cosine_similarity
cosine_sim(orthogonal) = 0
Therefore: angular_dist = 1 - 0 = 1.0
```

**Expected Results**:
- Distance ≈ 1.0
- In range [0, 1]

**Success Criteria**: ✅ Distance within 1e-6 of 1.0

---

#### Test 1.3.2: Identical Embeddings Distance
**Objective**: Verify identical embeddings have zero distance.

**Test Steps**:
1. Create embedding emb = [1, 2, 3]
2. Compute distance(emb, emb)
3. Verify distance ≈ 0.0

**Mathematical Foundation**:
```
cosine_sim(emb, emb) = 1.0
angular_dist = 1 - 1.0 = 0.0
```

**Expected Results**:
- Distance < 1e-6

**Success Criteria**: ✅ Distance ≈ 0.0

---

#### Test 1.3.3: Opposite Embeddings Distance
**Objective**: Verify opposite embeddings have maximum distance.

**Test Steps**:
1. Create opposite embeddings:
   - emb1 = [1, 0, 0]
   - emb2 = [-1, 0, 0]
2. Compute distance
3. Verify distance ≈ 2.0

**Mathematical Foundation**:
```
cosine_sim([1,0,0], [-1,0,0]) = -1.0
angular_dist = 1 - (-1) = 2.0
```

**Expected Results**:
- Distance ≈ 2.0

**Success Criteria**: ✅ Distance within 1e-6 of 2.0

---

#### Test 1.3.4: Distance Symmetry
**Objective**: Verify d(A,B) = d(B,A).

**Test Steps**:
1. Create random embeddings emb1, emb2
2. Compute d1 = distance(emb1, emb2)
3. Compute d2 = distance(emb2, emb1)
4. Verify |d1 - d2| < 1e-10

**Mathematical Property**:
- Distance metric must be symmetric

**Expected Results**:
- d1 == d2

**Success Criteria**: ✅ Absolute difference < 1e-10

---

#### Test 1.3.5: Empty Embedding Handling
**Objective**: Verify graceful handling of empty embeddings.

**Test Steps**:
1. Compute distance([], [1,2,3])
2. Verify returns 0.0
3. Verify no exceptions

**Expected Results**:
- Returns 0.0
- Warning logged
- No exceptions

**Success Criteria**: ✅ Returns 0.0 gracefully

---

#### Test 1.3.6: Dimension Mismatch Error
**Objective**: Verify error on mismatched dimensions.

**Test Steps**:
1. Create embeddings of different sizes:
   - emb1 = [1, 2, 3]
   - emb2 = [1, 2]
2. Call distance(emb1, emb2)
3. Expect ValueError

**Expected Results**:
- Raises ValueError
- Error mentions "dimensions"

**Success Criteria**: ✅ ValueError raised

---

#### Test 1.3.7: Pairwise Island Diversity
**Objective**: Verify pairwise diversity calculation.

**Test Steps**:
1. Create 3 islands with programs
2. Compute diversity between islands 0 and 1
3. Verify result in [0, 1]
4. Verify uses mean embeddings

**Expected Results**:
- Returns float
- 0.0 ≤ diversity ≤ 1.0
- Consistent with manual calculation

**Success Criteria**: ✅ Valid diversity value

---

### 1.4 System Diversity

#### Test 1.4.1: Basic System Diversity
**Objective**: Verify system-wide diversity computation.

**Test Steps**:
1. Populate 4 islands with programs
2. Call `compute_system_diversity()`
3. Verify returns dictionary with required keys
4. Verify all metrics in valid ranges

**Required Keys**:
- mean_diversity
- std_diversity
- min_diversity
- max_diversity
- pairwise_matrix
- island_embeddings
- island_labels

**Expected Results**:
- All keys present
- All values in valid ranges
- Matrix shape = (n_islands, n_islands)

**Success Criteria**: ✅ All keys present, values valid

---

#### Test 1.4.2: Insufficient Islands Error
**Objective**: Verify error with < 2 islands.

**Test Steps**:
1. Populate only 1 island
2. Call `compute_system_diversity()`
3. Verify returns error dict

**Expected Results**:
- Returns dict with "error" key
- Error message descriptive

**Success Criteria**: ✅ Error returned

---

#### Test 1.4.3: System Diversity with Intra-Island
**Objective**: Verify within-island diversity computation.

**Test Steps**:
1. Populate 3 islands
2. Call with `include_intra_island=True`
3. Verify additional keys present

**Additional Keys Expected**:
- intra_island_diversity
- mean_intra_island_diversity

**Expected Results**:
- Intra-island diversity for each island
- Mean computed correctly

**Success Criteria**: ✅ Additional metrics present

---

#### Test 1.4.4: Distance Matrix Symmetry
**Objective**: Verify pairwise matrix is symmetric.

**Test Steps**:
1. Compute system diversity
2. Extract matrix M
3. Verify M == M.T (transpose)

**Mathematical Property**:
- Distance matrix must be symmetric

**Expected Results**:
- M[i,j] == M[j,i] for all i,j

**Success Criteria**: ✅ Matrix symmetric within 1e-10

---

#### Test 1.4.5: Distance Matrix Diagonal
**Objective**: Verify diagonal elements are zero.

**Test Steps**:
1. Compute system diversity
2. Extract diagonal: diag(M)
3. Verify all ≈ 0.0

**Mathematical Property**:
- d(island, island) = 0

**Expected Results**:
- All diagonal elements < 1e-6

**Success Criteria**: ✅ Diagonal ≈ 0

---

## 2. INTEGRATION TESTS - Full Workflow Testing

### Test 2.1: End-to-End Diversity Analysis
**Objective**: Test complete diversity analysis workflow.

**Test Steps**:
1. Setup: Populate database (4 islands, 15 programs each)
2. Create analyzer
3. Compute system diversity
4. Find most/least diverse pairs
5. Generate report
6. Verify all steps succeeded

**Expected Results**:
- All operations complete without errors
- Report contains expected sections
- Metrics are consistent

**Success Criteria**: ✅ Complete workflow succeeds

---

### Test 2.2: Strategy Comparison Workflow
**Objective**: Test comparing multiple strategies.

**Test Steps**:
1. Populate database
2. Compare strategies: ["leaf", "elite", "all"]
3. Verify all strategies return results
4. Compare metrics across strategies

**Expected Results**:
- All 3 strategies succeed
- Results are comparable
- Metrics follow expected patterns (elite < all in count)

**Success Criteria**: ✅ All strategies succeed

---

### Test 2.3: Integration with Database Methods
**Objective**: Verify integration with ProgramDatabase.

**Test Steps**:
1. Create database
2. Call `db.get_diversity_analyzer()`
3. Use analyzer with database
4. Verify seamless integration

**Expected Results**:
- Factory method works
- Analyzer can access database
- No integration issues

**Success Criteria**: ✅ Seamless integration

---

## 3. EDGE CASE TESTS - Boundary Conditions

### Test 3.1: Single Program Per Island
**Objective**: Test with minimal programs.

**Test Steps**:
1. Add 1 program to each of 3 islands
2. Compute diversity
3. Verify handles gracefully

**Expected Results**:
- Computation succeeds
- Valid diversity values
- No errors

**Success Criteria**: ✅ Handles edge case

---

### Test 3.2: No Embeddings Available
**Objective**: Test when programs lack embeddings.

**Test Steps**:
1. Add programs with empty embeddings
2. Attempt diversity computation
3. Verify appropriate error

**Expected Results**:
- Returns error dict
- Error message clear

**Success Criteria**: ✅ Appropriate error handling

---

### Test 3.3: All Programs on One Island
**Objective**: Test single-island scenario.

**Test Steps**:
1. Add all programs to island 0
2. Attempt diversity computation
3. Verify error for insufficient islands

**Expected Results**:
- Returns error
- Message indicates need for multiple islands

**Success Criteria**: ✅ Error for single island

---

### Test 3.4: Very Similar Embeddings
**Objective**: Test near-identical embeddings.

**Test Steps**:
1. Create embeddings with tiny variations (ε = 0.001)
2. Compute diversity
3. Verify low diversity value

**Expected Results**:
- mean_diversity < 0.1
- No numerical issues

**Success Criteria**: ✅ Low diversity detected

---

### Test 3.5: Maximally Different Embeddings
**Objective**: Test orthogonal embeddings.

**Test Steps**:
1. Create orthogonal embeddings per island
2. Compute diversity
3. Verify high diversity value

**Expected Results**:
- mean_diversity > 0.5
- Reflects high diversity

**Success Criteria**: ✅ High diversity detected

---

## 4. PERFORMANCE TESTS - Scalability

### Test 4.1: Large Database Performance
**Objective**: Verify performance with large datasets.

**Test Configuration**:
- 5 islands
- 100 programs per island
- Total: 500 programs

**Test Steps**:
1. Populate large database
2. Measure diversity computation time
3. Verify completes in < 10 seconds

**Expected Results**:
- Computation time < 10s
- No memory issues
- Accurate results

**Success Criteria**: ✅ Time < 10s

---

### Test 4.2: Cache Performance Improvement
**Objective**: Verify caching improves performance.

**Test Steps**:
1. First call (uncached): measure time T1
2. Second call (cached): measure time T2
3. Verify T2 < T1
4. Verify results identical

**Expected Results**:
- T2 significantly < T1 (at least 2x faster)
- Results identical

**Success Criteria**: ✅ Speedup > 2x

---

### Test 4.3: Strategy Performance Comparison
**Objective**: Compare strategy performance.

**Test Steps**:
1. Measure time for each strategy
2. Compare times
3. Verify elite ≤ all

**Expected Pattern**:
- Elite fastest (smallest set)
- All slowest (largest set)
- Leaf intermediate

**Success Criteria**: ✅ Elite faster than all

---

## 5. STATISTICAL TESTS - Mathematical Properties

### Test 5.1: Triangle Inequality
**Objective**: Verify d(A,C) ≤ d(A,B) + d(B,C).

**Test Steps**:
1. Create 3 random embeddings
2. Compute all pairwise distances
3. Verify triangle inequality holds

**Mathematical Property**:
```
For all embeddings A, B, C:
d(A,C) ≤ d(A,B) + d(B,C)
```

**Success Criteria**: ✅ Inequality satisfied

---

### Test 5.2: Mean Diversity Bounds
**Objective**: Verify min ≤ mean ≤ max.

**Test Steps**:
1. Compute system diversity
2. Extract min, mean, max
3. Verify ordering

**Mathematical Property**:
```
min_diversity ≤ mean_diversity ≤ max_diversity
```

**Success Criteria**: ✅ Ordering correct

---

### Test 5.3: Consistency Across Runs
**Objective**: Verify deterministic results.

**Test Steps**:
1. Run diversity computation 3 times
2. Clear cache between runs
3. Verify identical results

**Expected Results**:
- All 3 runs produce identical values
- No randomness in computation

**Success Criteria**: ✅ Results identical

---

## 6. COMPARISON TESTS - Strategy Validation

### Test 6.1: Compare All Strategies
**Objective**: Test all selection strategies.

**Test Steps**:
1. Run comparison for all strategies
2. Verify all succeed
3. Compare results

**Strategies**:
- leaf
- recent_N
- elite
- all

**Success Criteria**: ✅ All strategies succeed

---

### Test 6.2: Strategy Result Differences
**Objective**: Verify strategies give different results.

**Test Steps**:
1. Compare leaf vs all
2. Verify different program counts
3. Verify potentially different diversity

**Expected Results**:
- Different program selections
- Possibly different diversity values

**Success Criteria**: ✅ Strategies differ appropriately

---

### Test 6.3: Most Diverse Pair Identification
**Objective**: Verify correct pair identification.

**Test Steps**:
1. Create islands with known diversity
   - Islands 0,1: similar
   - Island 2: different
2. Find most diverse pair
3. Verify includes island 2

**Expected Results**:
- Most similar: (0, 1)
- Most diverse: includes 2

**Success Criteria**: ✅ Correct pairs identified

---

## 7. REPORT GENERATION TESTS

### Test 7.1: Basic Report Generation
**Objective**: Test report creation.

**Test Steps**:
1. Generate report
2. Verify contains key sections
3. Verify readable format

**Required Sections**:
- Title header
- Database info
- Diversity metrics
- Island pairs

**Success Criteria**: ✅ Report complete and readable

---

### Test 7.2: Full Report with Strategies
**Objective**: Test comprehensive report.

**Test Steps**:
1. Generate with `include_all_strategies=True`
2. Verify strategy comparison section
3. Verify all strategies listed

**Success Criteria**: ✅ All strategies in report

---

### Test 7.3: Save Report to File
**Objective**: Test file saving.

**Test Steps**:
1. Generate report with output_path
2. Verify file created
3. Verify content matches returned string

**Success Criteria**: ✅ File saved correctly

---

## Test Execution Plan

### Phase 1: Core Functionality (Week 1)
- [ ] Unit tests for selection strategies
- [ ] Unit tests for embedding computation
- [ ] Unit tests for distance metrics
- [ ] Run: `pytest tests/test_diversity.py::TestProgramSelectionStrategies -v`
- [ ] Run: `pytest tests/test_diversity.py::TestEmbeddingComputation -v`
- [ ] Run: `pytest tests/test_diversity.py::TestDistanceMetrics -v`

### Phase 2: System-Wide Features (Week 1)
- [ ] Unit tests for system diversity
- [ ] Integration tests
- [ ] Run: `pytest tests/test_diversity.py::TestSystemDiversity -v`
- [ ] Run: `pytest tests/test_diversity.py::TestFullWorkflow -v`

### Phase 3: Edge Cases and Robustness (Week 2)
- [ ] Edge case tests
- [ ] Error handling validation
- [ ] Run: `pytest tests/test_diversity.py::TestEdgeCases -v`

### Phase 4: Performance and Optimization (Week 2)
- [ ] Performance benchmarks
- [ ] Cache validation
- [ ] Scalability tests
- [ ] Run: `pytest tests/test_diversity.py::TestPerformance -v`

### Phase 5: Statistical Validation (Week 2)
- [ ] Mathematical property tests
- [ ] Consistency validation
- [ ] Run: `pytest tests/test_diversity.py::TestStatisticalProperties -v`

### Phase 6: Comparison and Reporting (Week 3)
- [ ] Strategy comparison tests
- [ ] Report generation tests
- [ ] Run: `pytest tests/test_diversity.py::TestStrategyComparison -v`
- [ ] Run: `pytest tests/test_diversity.py::TestReportGeneration -v`

### Phase 7: Visualization (Week 3)
- [ ] Plot generation tests
- [ ] Run: `pytest tests/test_diversity_visualization.py -v`

---

## Continuous Integration

### Automated Testing
```bash
# Run all diversity tests
pytest tests/test_diversity*.py -v --tb=short

# Run with coverage
pytest tests/test_diversity*.py --cov=shinka.database.diversity --cov-report=html

# Run performance tests only
pytest tests/test_diversity.py::TestPerformance -v

# Run quick tests (exclude slow tests)
pytest tests/test_diversity.py -v -m "not slow"
```

### Test Metrics Targets
- **Code Coverage**: > 95%
- **Test Count**: > 100 tests
- **Passing Rate**: 100%
- **Performance**: All tests < 30s total

---

## Success Criteria Summary

### Functional Requirements
✅ All selection strategies work correctly  
✅ Distance metrics mathematically sound  
✅ System diversity computed accurately  
✅ Report generation produces valid output  

### Non-Functional Requirements
✅ Performance: < 10s for 500 programs  
✅ Memory: < 500MB for large databases  
✅ Robustness: Handles all edge cases  
✅ Accuracy: Within 1e-6 numerical precision  

### Integration Requirements
✅ Seamless database integration  
✅ Compatible with existing architecture  
✅ No breaking changes to existing code  

---

## Manual Testing Checklist

### Scenario 1: Circle Packing Example
- [ ] Run diversity analysis on circle_packing results
- [ ] Verify diversity increases over generations
- [ ] Check that elite strategy works
- [ ] Generate and inspect report

### Scenario 2: Agent Design Example
- [ ] Analyze diversity in agent design evolution
- [ ] Compare strategies on real data
- [ ] Validate that leaf strategy finds frontier

### Scenario 3: ALE-Bench Example
- [ ] Test with C++ code embeddings
- [ ] Verify language-agnostic operation
- [ ] Check performance on large codebase

---

## Appendix: Test Data Specifications

### Mock Program Properties
```python
Program:
  - id: str (UUID format)
  - code: str (50-500 characters)
  - embedding: List[float] (128-dimensional)
  - island_idx: int (0 to num_islands-1)
  - generation: int (0 to 100)
  - parent_id: Optional[str]
  - correct: bool (70% True, 30% False)
  - combined_score: float (0.0 to 1.0)
  - children_count: int (0 to 5)
```

### Database Configurations
```python
Small: 2 islands, 5 programs each
Medium: 4 islands, 15 programs each
Large: 5 islands, 100 programs each
XLarge: 10 islands, 200 programs each
```

---

**Document Version**: 1.0  
**Last Updated**: November 2, 2025  
**Author**: ShinkaEvolve Team  
**Status**: Ready for Implementation
