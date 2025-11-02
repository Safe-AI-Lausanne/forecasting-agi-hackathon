# Step-by-Step Visualization Plan for Merged Safety Benchmark Data

## Overview
Analyze the merged dataset (updated_data.csv + updated_data_v2.csv) to understand:
1. How baseline performance scales with model size
2. How safety techniques perform relative to baseline
3. Which techniques improve more at larger scales
4. Which benchmarks are "solved by scaling" vs requiring intervention

---

## Phase 1: Data Overview & Exploration

### Step 1.1: Dataset Statistics Dashboard
**Purpose**: Get familiar with the data

**Visualizations**:
- Bar chart: Number of data points per benchmark
- Bar chart: Number of data points per technique
- Histogram: Distribution of model scales
- Table: Coverage matrix (Technique × Benchmark)

**Insights to extract**:
- Which benchmarks have most/least data?
- Which techniques are most tested?
- What scale ranges do we have?
- Are there gaps in coverage?

---

### Step 1.2: Performance Distribution Overview
**Purpose**: Understand performance ranges

**Visualizations**:
- Box plots: Performance distribution per benchmark
- Violin plots: Performance by technique (all benchmarks combined)
- Heatmap: Mean performance (Benchmark × Technique)

**Insights to extract**:
- Which benchmarks are harder (lower scores)?
- Which techniques generally perform better?
- Performance variance across benchmarks

---

## Phase 2: Baseline Scaling Analysis

### Step 2.1: Baseline Performance vs Scale
**Purpose**: Identify which benchmarks saturate with scale

**Visualizations**:
- **Main plot**: Scatter + trend lines for each benchmark
  - X-axis: Model scale (log scale)
  - Y-axis: Baseline performance
  - Color: Different benchmark
  - Add: Spearman correlation coefficient and slope

**Insights to extract**:
- Which benchmarks improve with scale? (Saturated)
- Which benchmarks don't improve? (Not saturated)
- Correlation strength and statistical significance

---

### Step 2.2: Benchmark Categorization
**Purpose**: Classify benchmarks by saturation

**Visualizations**:
- Horizontal bar chart: Spearman correlation per benchmark
  - Color: Saturated (green) vs Not Saturated (red)
  - Add vertical line at threshold (0.5)
  - Annotate with slope values

**Insights to extract**:
- Clear categorization of benchmarks
- Magnitude of scaling effect

---

## Phase 3: Technique Performance Analysis

### Step 3.1: Technique vs Baseline Comparison
**Purpose**: See which techniques outperform baseline

**Visualizations**:
- Grouped bar charts per benchmark:
  - X-axis: Model scales
  - Y-axis: Performance
  - Bars: Baseline + each technique
  - One subplot per benchmark

**Insights to extract**:
- Which techniques consistently beat baseline?
- How much improvement do techniques provide?
- Performance gaps at different scales

---

### Step 3.2: Performance Gain Heatmap
**Purpose**: Overview of technique effectiveness

**Visualizations**:
- Heatmap: Mean performance gain (Technique × Benchmark)
  - Color scale: Red (negative) to Green (positive)
  - Annotate cells with exact values
  - Highlight best/worst combinations

**Insights to extract**:
- Which technique works best for which benchmark?
- Consistent performers vs specialized techniques

---

## Phase 4: Scaling Behavior Analysis

### Step 4.1: Performance Gain vs Scale (Individual Points)
**Purpose**: Analyze how technique advantage changes with scale

**Visualizations**:
- Scatter plots with trend lines:
  - One subplot per benchmark
  - X-axis: Model scale
  - Y-axis: Performance difference from baseline
  - Different colors/markers for each technique
  - Add zero reference line
  - Show correlation and slope for each technique

**Insights to extract**:
- Do techniques become MORE effective at larger scales? (positive slope)
- Do techniques become LESS effective? (negative slope)
- Are effects scale-independent? (near-zero slope)

---

### Step 4.2: Slope Analysis
**Purpose**: Categorize techniques by scaling behavior

**Visualizations**:
- Scatter plot: Slope vs Mean Gain
  - X-axis: Slope of (gain vs scale)
  - Y-axis: Mean performance gain
  - Color: Technique
  - Size: Number of data points
  - Quadrants:
    - Top-right: High gain + increases with scale (BEST)
    - Top-left: High gain + stable/decreases
    - Bottom-right: Low gain + increases with scale
    - Bottom-left: Low gain + decreases (WORST)

**Insights to extract**:
- Which techniques are in the "sweet spot"?
- Trade-offs between current performance and scaling potential

---

## Phase 5: Benchmark-Specific Deep Dives

### Step 5.1: Per-Benchmark Analysis
**Purpose**: Detailed view of each benchmark

**For each benchmark, create a 2x2 subplot grid**:
1. **Top-left**: Baseline scaling (scatter + trend)
2. **Top-right**: All techniques vs scale (line plot)
3. **Bottom-left**: Technique ranking at different scales (bar chart)
4. **Bottom-right**: Performance gain distribution (box plot)

**Insights to extract**:
- Comprehensive understanding of each benchmark
- Scale-dependent technique rankings
- Variability in technique performance

---

## Phase 6: Summary & Categorization

### Step 6.1: Overall Summary Dashboard
**Purpose**: One-page overview of key findings

**Layout** (4 panels):
1. **Benchmark Saturation** (bar chart with categories)
2. **Technique Effectiveness** (heatmap of mean gains)
3. **Scaling Trends** (slope scatter plot)
4. **Top Performers** (table of best technique per benchmark)

---

### Step 6.2: Interactive Recommendations Table
**Purpose**: Actionable insights

**Create table showing**:
- Benchmark name
- Saturation status
- Recommended technique (highest mean gain)
- Scaling behavior (gain increases/decreases/stable)
- Confidence (based on n_points and p-value)

---

## Phase 7: Advanced Visualizations (Optional)

### Step 7.1: Model Family Comparison
If multiple models at same scale:
- Compare performance across model families
- Identify model-specific effects

### Step 7.2: Statistical Significance
- Add confidence intervals to trend lines
- Show p-values for correlations
- Bootstrap analysis for robustness

### Step 7.3: Interactive Plots
- Plotly/Bokeh for interactive exploration
- Hover tooltips with detailed info
- Filter by benchmark/technique/scale

---

## Implementation Order

**Priority 1 (Essential)**: Steps 1.1, 2.1, 2.2, 4.1, 6.1
- Basic overview + key scaling analysis + summary

**Priority 2 (Important)**: Steps 3.1, 3.2, 4.2
- Technique comparison and categorization

**Priority 3 (Nice to have)**: Steps 1.2, 5.1, 6.2, 7.x
- Deep dives and advanced features

---

## Output Format

**Option A**: Single comprehensive Jupyter notebook
- All visualizations in one place
- Easy to run and share
- Can export to HTML

**Option B**: Separate notebooks
- `1_data_overview.ipynb`
- `2_baseline_scaling.ipynb`
- `3_technique_analysis.ipynb`
- `4_scaling_behavior.ipynb`
- `5_summary_dashboard.ipynb`

**Option C**: Python script + saved figures
- Generate all plots programmatically
- Save as PNG/PDF for reports
- Reusable for future data updates

**Recommendation**: Start with Option A (single notebook), then extract key figures for reports.

