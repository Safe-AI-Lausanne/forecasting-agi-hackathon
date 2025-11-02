# forecasting-agi-hackathon

## Analysis

Created comprehensive preprocessing and visualization pipeline for safety benchmark scaling analysis:

- **Data preprocessing** (`notebooks/preprocess_and_merge.ipynb`): Merged two datasets with different schemas, normalized benchmark directionality (WMDP benchmarks inverted), and created unique technique identifiers per paper
- **Visualization & analysis** (`notebooks/safety_benchmark_visualizations.ipynb`): Analyzed baseline scaling, technique effectiveness, and scaling behavior following SafetyWashing methodology

Key findings: Identified which benchmarks saturate with scale vs. require intervention, and which safety techniques scale well.

```some benchmark does not have enough baseline datapoints (e.g. only 2 models)```

- https://docs.google.com/spreadsheets/d/1uh_TqBLcSDB2_8Eyrww6Wl-4ZK3heV5mUBWqW2OZuRc/edit?usp=sharing

# Sources

- [Safetywashing: Do AI Safety Benchmarks
Actually Measure Safety Progress?](https://proceedings.neurips.cc/paper_files/paper/2024/file/7ebcdd0de471c027e67a11959c666d74-Paper-Datasets_and_Benchmarks_Track.pdf)
  - Argues that some safety benchmarks are naturally solved by scale and do not require effort to solve them (Bitter lesson) and are thus useless. 
