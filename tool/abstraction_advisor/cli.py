#!/usr/bin/env python3
"""
abstraction-advisor — CLI decision support tool.

Usage:
  # Step 1: Profile the application (currently: manual or ncu CSV input)
  abstraction-advisor profile \
      --ncu-csv results/nvidia_a100/stream/profiles/ncu_kokkos_large.csv \
      --kernel stream_triad \
      --output workload_profile.json

  # Step 2: Get recommendation
  abstraction-advisor recommend \
      --profile workload_profile.json \
      --targets nvidia_a100,amd_mi250x \
      --priority portability

  # Step 3: Query taxonomy
  abstraction-advisor taxonomy --query "launch overhead"
  abstraction-advisor taxonomy --pattern P001
"""

import json
import sys
from pathlib import Path

import click

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tool.abstraction_advisor.profiler import from_ncu_csv, from_manual, save, load
from tool.abstraction_advisor.recommender import get_recommendation
from tool.abstraction_advisor.database import TaxonomyDatabase


@click.group()
@click.version_option("0.1.0")
def cli():
    """Abstraction Advisor — empirically-grounded portability framework recommender."""


@cli.command()
@click.option("--ncu-csv", type=click.Path(exists=True), help="ncu CSV export")
@click.option("--kernel", required=True, help="Kernel name (for labeling)")
@click.option("--output", required=True, type=click.Path(), help="workload_profile.json")
@click.option("--platform", default="nvidia_a100")
def profile(ncu_csv, kernel, output, platform):
    """Extract workload characteristics from profiler output."""
    if ncu_csv:
        p = from_ncu_csv(Path(ncu_csv), kernel_name=kernel)
        p.platform = platform
    else:
        click.echo("No profiler input provided. Use --ncu-csv or extend for other backends.")
        sys.exit(1)

    save(p, Path(output))
    click.echo(f"Strategy: profile complete → {output}")


@cli.command()
@click.option("--profile", "profile_path", required=True, type=click.Path(exists=True))
@click.option("--targets", default="nvidia_a100,amd_mi250x,intel_pvc",
              help="Comma-separated target platforms")
@click.option("--taxonomy", default="data/taxonomy.json", type=click.Path())
@click.option("--output", type=click.Path(), help="Write recommendation to JSON")
@click.option("--priority", default="portability",
              type=click.Choice(["portability", "performance", "productivity"]))
def recommend(profile_path, targets, taxonomy, output, priority):
    """Get abstraction recommendation for a profiled workload."""
    p = load(Path(profile_path))
    target_list = [t.strip() for t in targets.split(",")]

    result = get_recommendation(p, target_list, taxonomy_path=Path(taxonomy))

    # Format output
    click.echo(f"\nKernel:   {result['kernel']}")
    click.echo(f"Targets:  {', '.join(result['targets'])}")
    click.echo(f"Strategy: {result['strategy']}  [{result['confidence']} confidence]")
    click.echo(f"\nRationale:\n  {result['rationale']}")

    if result["suggested_abstractions"]:
        click.echo(f"\nSuggested: {', '.join(result['suggested_abstractions'])}")
    if result["expected_ppc_range"]:
        lo, hi = result["expected_ppc_range"]
        click.echo(f"Expected PPC: {lo:.2f}–{hi:.2f}")
    if result["warnings"]:
        click.echo("\nWarnings:")
        for w in result["warnings"]:
            click.echo(f"  ! {w}")
    if result["known_patterns"]:
        click.echo(f"\nKnown patterns: {', '.join(result['known_patterns'])}")

    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        click.echo(f"\nFull output written to {output}")


@cli.command()
@click.option("--query", help="Search taxonomy by keyword")
@click.option("--pattern", help="Show a specific pattern by ID (e.g. P001)")
@click.option("--taxonomy", default="data/taxonomy.json", type=click.Path())
@click.option("--all", "show_all", is_flag=True, help="Show all patterns including hypotheses")
def taxonomy(query, pattern, taxonomy_path="data/taxonomy.json", show_all=False, **kwargs):
    """Query the taxonomy database."""
    tax_path = Path(kwargs.get("taxonomy", "data/taxonomy.json"))
    try:
        db = TaxonomyDatabase(tax_path)
    except FileNotFoundError as e:
        click.echo(f"ERROR: {e}"); sys.exit(1)

    summary = db.summary()
    click.echo(f"\nTaxonomy: {summary['total_patterns']} patterns "
               f"({summary['validated']} validated, {summary['hypotheses']} hypotheses)")

    if pattern:
        p = db.get_pattern(pattern)
        if p:
            click.echo(f"\n{json.dumps(p, indent=2)}")
        else:
            click.echo(f"Pattern '{pattern}' not found")
    elif query:
        matches = [
            p for p in db.all_patterns(include_hypotheses=show_all)
            if query.lower() in json.dumps(p).lower()
        ]
        click.echo(f"\nMatches for '{query}': {len(matches)}")
        for p in matches:
            click.echo(f"  [{p['id']}] {p['name']} ({p['type']}, {p.get('status', '?')})")
    else:
        for p in db.all_patterns(include_hypotheses=show_all):
            click.echo(f"  [{p['id']}] {p['name']} — {p['type']}")


def main():
    cli()


if __name__ == "__main__":
    main()
