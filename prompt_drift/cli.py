"""Click CLI for prompt-drift."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from . import __version__
from .comparator import compare_snapshots
from .models import TestInput
from .reporter import print_report
from .snapshot import create_snapshot, load_test_inputs
from .store import (
    delete_snapshot,
    list_snapshots,
    load_snapshot,
    save_snapshot,
    snapshot_exists,
)

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="prompt-drift")
def cli():
    """prompt-drift: Detect when prompt template changes cause output regressions."""
    pass


# ---------------------------------------------------------------------------
# snapshot command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--name", "-n", required=True, help="Unique name for this snapshot (e.g., 'v1').")
@click.option("--prompt", "-p", required=True, type=click.Path(exists=True), help="Path to the prompt template file.")
@click.option("--inputs", "-i", required=True, type=click.Path(exists=True), help="Path to test inputs JSON file.")
@click.option("--model", "-m", default="gpt-4o", help="Model to use (default: gpt-4o).")
@click.option("--dry-run", is_flag=True, default=False, help="Use mock LLM instead of real API calls.")
@click.option("--store-dir", default=None, help="Override snapshot storage directory.")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing snapshot with same name.")
def snapshot(name: str, prompt: str, inputs: str, model: str, dry_run: bool, store_dir: Optional[str], force: bool):
    """Take a snapshot of prompt outputs against test inputs."""
    # Validate that prompt file is readable before doing any work
    prompt_path = Path(prompt)
    if not prompt_path.is_file():
        console.print(f"[red]Error: Prompt file '{prompt}' does not exist.[/red]")
        sys.exit(1)
    try:
        prompt_text = prompt_path.read_text(encoding="utf-8")
    except PermissionError:
        console.print(f"[red]Error: Prompt file '{prompt}' is not readable (permission denied).[/red]")
        sys.exit(1)
    except OSError as exc:
        console.print(f"[red]Error: Cannot read prompt file '{prompt}': {exc}[/red]")
        sys.exit(1)

    # Validate that inputs file is readable
    inputs_path = Path(inputs)
    if not inputs_path.is_file():
        console.print(f"[red]Error: Inputs file '{inputs}' does not exist.[/red]")
        sys.exit(1)

    # Check for existing snapshot
    if snapshot_exists(name, store_dir) and not force:
        console.print(f"[red]Snapshot '{name}' already exists. Use --force to overwrite.[/red]")
        sys.exit(1)

    # Files validated -- report what was loaded
    console.print(f"[dim]Loaded prompt template from {prompt} ({len(prompt_text)} chars)[/dim]")

    # Load test inputs
    test_inputs = load_test_inputs(inputs)
    console.print(f"[dim]Loaded {len(test_inputs)} test inputs from {inputs}[/dim]")

    if dry_run:
        console.print("[yellow]Running in dry-run mode (mock LLM)[/yellow]")

    # Create snapshot
    with console.status(f"Running {len(test_inputs)} inputs through {model}..."):
        snap = create_snapshot(
            name=name,
            prompt_template=prompt_text,
            test_inputs=test_inputs,
            model=model,
            dry_run=dry_run,
        )

    # Save
    path = save_snapshot(snap, store_dir)
    console.print(f"[green]Snapshot '{name}' saved to {path}[/green]")
    console.print(f"[dim]  Model: {model} | Entries: {len(snap.entries)} | Created: {snap.created_at}[/dim]")


# ---------------------------------------------------------------------------
# compare command
# ---------------------------------------------------------------------------

def _validate_threshold(ctx, param, value):
    """Click callback to validate that threshold is in [0.0, 1.0]."""
    if not (0.0 <= value <= 1.0):
        raise click.BadParameter(
            f"Threshold must be between 0.0 and 1.0, got {value}."
        )
    return value


@cli.command()
@click.argument("baseline")
@click.argument("candidate")
@click.option("--threshold", "-t", default=0.2, type=float, callback=_validate_threshold, help="Drift threshold for regression (default: 0.2 = 20%).")
@click.option("--store-dir", default=None, help="Override snapshot storage directory.")
@click.option("--no-diff", is_flag=True, default=False, help="Hide text diffs.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show full output text in diffs.")
@click.option("--json-out", type=click.Path(), default=None, help="Save report as JSON to this path.")
def compare(baseline: str, candidate: str, threshold: float, store_dir: Optional[str], no_diff: bool, verbose: bool, json_out: Optional[str]):
    """Compare two snapshots and show a drift report."""
    try:
        snap_baseline = load_snapshot(baseline, store_dir)
    except FileNotFoundError:
        console.print(f"[red]Baseline snapshot '{baseline}' not found.[/red]")
        available = list_snapshots(store_dir)
        if available:
            console.print(f"[dim]Available snapshots: {', '.join(available)}[/dim]")
        sys.exit(1)

    try:
        snap_candidate = load_snapshot(candidate, store_dir)
    except FileNotFoundError:
        console.print(f"[red]Candidate snapshot '{candidate}' not found.[/red]")
        available = list_snapshots(store_dir)
        if available:
            console.print(f"[dim]Available snapshots: {', '.join(available)}[/dim]")
        sys.exit(1)

    report = compare_snapshots(snap_baseline, snap_candidate, threshold=threshold)
    print_report(report, console=console, show_diff=not no_diff, verbose=verbose)

    if json_out:
        Path(json_out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[dim]Report saved to {json_out}[/dim]")

    # Exit with non-zero status if regressions found
    if report.has_regressions:
        sys.exit(1)


# ---------------------------------------------------------------------------
# watch command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("prompt_file", type=click.Path(exists=True))
@click.option("--baseline", "-b", required=True, help="Baseline snapshot name to compare against.")
@click.option("--inputs", "-i", required=True, type=click.Path(exists=True), help="Path to test inputs JSON file.")
@click.option("--model", "-m", default="gpt-4o", help="Model to use (default: gpt-4o).")
@click.option("--interval", default=2.0, type=float, help="Polling interval in seconds (default: 2).")
@click.option("--threshold", "-t", default=0.2, type=float, callback=_validate_threshold, help="Drift threshold for regression (default: 0.2).")
@click.option("--dry-run", is_flag=True, default=False, help="Use mock LLM instead of real API calls.")
@click.option("--store-dir", default=None, help="Override snapshot storage directory.")
def watch(prompt_file: str, baseline: str, inputs: str, model: str, interval: float, threshold: float, dry_run: bool, store_dir: Optional[str]):
    """Watch a prompt file for changes and auto-compare against a baseline."""
    # Verify baseline exists
    try:
        snap_baseline = load_snapshot(baseline, store_dir)
    except FileNotFoundError:
        console.print(f"[red]Baseline snapshot '{baseline}' not found.[/red]")
        sys.exit(1)

    test_inputs = load_test_inputs(inputs)
    prompt_path = Path(prompt_file)
    last_mtime = prompt_path.stat().st_mtime
    last_content = prompt_path.read_text(encoding="utf-8")

    console.print(f"[cyan]Watching {prompt_file} for changes (Ctrl+C to stop)...[/cyan]")
    console.print(f"[dim]Baseline: {baseline} | Model: {model} | Threshold: {threshold:.0%}[/dim]")
    if dry_run:
        console.print("[yellow]Running in dry-run mode (mock LLM)[/yellow]")

    iteration = 0
    try:
        while True:
            time.sleep(interval)
            current_mtime = prompt_path.stat().st_mtime
            current_content = prompt_path.read_text(encoding="utf-8")

            if current_mtime != last_mtime or current_content != last_content:
                iteration += 1
                last_mtime = current_mtime
                last_content = current_content

                console.print(f"\n[cyan]Change detected (iteration {iteration})! Re-running...[/cyan]")

                with console.status(f"Running {len(test_inputs)} inputs..."):
                    snap_new = create_snapshot(
                        name=f"_watch_{iteration}",
                        prompt_template=current_content,
                        test_inputs=test_inputs,
                        model=model,
                        dry_run=dry_run,
                    )

                report = compare_snapshots(snap_baseline, snap_new, threshold=threshold)
                print_report(report, console=console, show_diff=True)

    except KeyboardInterrupt:
        console.print("\n[dim]Watch stopped.[/dim]")


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------

@cli.command("list")
@click.option("--store-dir", default=None, help="Override snapshot storage directory.")
def list_cmd(store_dir: Optional[str]):
    """List all saved snapshots."""
    names = list_snapshots(store_dir)
    if not names:
        console.print("[dim]No snapshots found.[/dim]")
        return

    table = click.echo  # simple output
    console.print(f"[bold]Snapshots ({len(names)}):[/bold]")
    for name in names:
        snap = load_snapshot(name, store_dir)
        console.print(
            f"  [cyan]{name}[/cyan]  "
            f"[dim]model={snap.model} entries={len(snap.entries)} created={snap.created_at}[/dim]"
        )


# ---------------------------------------------------------------------------
# delete command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("name")
@click.option("--store-dir", default=None, help="Override snapshot storage directory.")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation.")
def delete(name: str, store_dir: Optional[str], yes: bool):
    """Delete a saved snapshot."""
    if not snapshot_exists(name, store_dir):
        console.print(f"[red]Snapshot '{name}' not found.[/red]")
        sys.exit(1)

    if not yes:
        click.confirm(f"Delete snapshot '{name}'?", abort=True)

    deleted = delete_snapshot(name, store_dir)
    if deleted:
        console.print(f"[green]Snapshot '{name}' deleted.[/green]")


def main():
    cli()


if __name__ == "__main__":
    main()
