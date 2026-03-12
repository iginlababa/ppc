# Contributing

## Branch Naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feature/<short-desc>` | `feature/roofline-plot` |
| Experiment | `exp/<E1-stream>` | `exp/E3-stencil-kokkos` |
| Bug fix | `fix/<issue>` | `fix/ppc-divide-by-zero` |
| Paper section | `paper/<section>` | `paper/methodology` |

## Commit Messages

Prefix every commit with the experiment ID or scope:

```
[E2] Add Kokkos DGEMM kernel + correctness test
[scripts] Fix parse_results.py schema validation
[paper] Draft §4 methodology — measurement protocol
```

## Pull Request Rules

- Never commit to `main` directly — open a PR even if working solo
- One experiment per PR — keep diffs reviewable
- All correctness tests must pass before merge
- Paper PRs must build cleanly with `latexmk`

## Adding a New Dependency

1. Add to `environment.yml`
2. Document purpose in a comment
3. Run `conda env update --file environment.yml --prune` and verify the env builds cleanly
4. Include in the same PR as the code that requires it

## Data Policy

- Raw CSVs in `data/` are append-only — never overwrite or edit by hand
- Large files go in `data/raw/` and are tracked via Git LFS
- Do not commit proprietary benchmark data

## Adding a Taxonomy Pattern

1. Run the experiment and collect profiling evidence
2. Add entry to `data/taxonomy.json` following the schema in `project_spec.md §10.4`
3. Reference the `experiment_id` from `data/performance.csv`
4. Open a PR with the pattern + supporting evidence
