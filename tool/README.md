# abstraction-advisor

CLI decision support tool for abstraction layer selection in GPU-accelerated HPC.
See `project_spec.md §15` for full architecture description.

## Install

```bash
conda activate hpc-abstraction
pip install -e tool/   # once setup.py/pyproject.toml is added
# Or run directly:
python -m tool.abstraction_advisor.cli --help
```

## Usage

```bash
# Step 1: Profile
abstraction-advisor profile \
    --ncu-csv results/nvidia_a100/stream/profiles/ncu_kokkos_large.csv \
    --kernel stream_triad \
    --output workload_profile.json

# Step 2: Recommend
abstraction-advisor recommend \
    --profile workload_profile.json \
    --targets nvidia_a100,amd_mi250x

# Step 3: Query taxonomy
abstraction-advisor taxonomy --query "launch overhead"
abstraction-advisor taxonomy --pattern P001
```

## Status

- [ ] Profiler: ncu CSV parsing (partial)
- [ ] Profiler: rocprof/omniperf parsing (TODO)
- [ ] Profiler: VTune parsing (TODO)
- [ ] Recommender: decision logic implemented
- [ ] Taxonomy: 3 seed patterns (hypotheses — not yet validated)
- [ ] Validation: retrospective validation pending (Month 9)
