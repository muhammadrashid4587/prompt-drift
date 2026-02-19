"""Rich console reporter for drift reports."""

from __future__ import annotations

import difflib
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import ComparisonEntry, DriftReport


def _drift_color(drift: float, threshold: float) -> str:
    """Return a Rich color name based on drift severity."""
    if drift <= 0.05:
        return "green"
    elif drift <= threshold:
        return "yellow"
    else:
        return "red"


def _drift_label(drift: float, threshold: float) -> str:
    """Human-readable drift label."""
    if drift <= 0.05:
        return "SAME"
    elif drift <= threshold:
        return "MINOR"
    else:
        return "REGRESSION"


def _unified_diff(text_a: str, text_b: str, label_a: str = "before", label_b: str = "after") -> str:
    """Generate a unified diff between two texts."""
    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)
    diff = difflib.unified_diff(lines_a, lines_b, fromfile=label_a, tofile=label_b)
    return "".join(diff)


def print_report(
    report: DriftReport,
    console: Optional[Console] = None,
    show_diff: bool = True,
    verbose: bool = False,
) -> None:
    """Print a formatted drift report to the console.

    Args:
        report: The DriftReport to display.
        console: Optional Rich Console instance.
        show_diff: Whether to show text diffs for changed entries.
        verbose: Show full outputs instead of truncated versions.
    """
    if console is None:
        console = Console()

    # Header
    header_color = _drift_color(report.mean_drift, report.threshold)
    console.print()
    console.print(
        Panel(
            f"[bold]Drift Report:[/bold] {report.baseline_name} -> {report.candidate_name}\n"
            f"[dim]Threshold: {report.threshold:.0%}[/dim]",
            border_style=header_color,
        )
    )

    # Summary table
    summary = Table(title="Aggregate Metrics", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")

    mean_color = _drift_color(report.mean_drift, report.threshold)
    max_color = _drift_color(report.max_drift, report.threshold)

    summary.add_row("Mean Drift", f"[{mean_color}]{report.mean_drift:.2%}[/{mean_color}]")
    summary.add_row("Max Drift", f"[{max_color}]{report.max_drift:.2%}[/{max_color}]")
    summary.add_row("Min Drift", f"{report.min_drift:.2%}")
    summary.add_row("Total Inputs", str(len(report.entries)))

    reg_color = "red" if report.regression_count > 0 else "green"
    summary.add_row(
        "Regressions",
        f"[{reg_color}]{report.regression_count}[/{reg_color}]",
    )

    console.print(summary)
    console.print()

    # Per-entry details
    detail_table = Table(
        title="Per-Input Results",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    detail_table.add_column("#", style="dim", width=4)
    detail_table.add_column("Input", max_width=40)
    detail_table.add_column("Levenshtein", justify="right", width=12)
    detail_table.add_column("BLEU", justify="right", width=10)
    detail_table.add_column("Cosine", justify="right", width=10)
    detail_table.add_column("Drift", justify="right", width=10)
    detail_table.add_column("Status", justify="center", width=12)

    for i, entry in enumerate(report.entries, 1):
        color = _drift_color(entry.drift_score, report.threshold)
        label = _drift_label(entry.drift_score, report.threshold)
        input_preview = entry.input.content[:37] + "..." if len(entry.input.content) > 40 else entry.input.content

        detail_table.add_row(
            str(i),
            input_preview,
            f"{entry.similarity.levenshtein:.2%}",
            f"{entry.similarity.bleu:.2%}",
            f"{entry.similarity.cosine:.2%}",
            f"[{color}]{entry.drift_score:.2%}[/{color}]",
            f"[{color}]{label}[/{color}]",
        )

    console.print(detail_table)

    # Show diffs for entries with notable drift
    if show_diff:
        for i, entry in enumerate(report.entries, 1):
            if entry.drift_score > 0.05:
                console.print()
                color = _drift_color(entry.drift_score, report.threshold)
                console.print(
                    Panel(
                        f"[bold]Input #{i}:[/bold] {entry.input.content}",
                        border_style=color,
                        title=f"Drift: {entry.drift_score:.2%}",
                    )
                )

                diff_text = _unified_diff(
                    entry.output_before,
                    entry.output_after,
                    label_a=f"{report.baseline_name}",
                    label_b=f"{report.candidate_name}",
                )
                if diff_text.strip():
                    # Colorise diff lines
                    for line in diff_text.splitlines():
                        if line.startswith("+") and not line.startswith("+++"):
                            console.print(f"  [green]{line}[/green]")
                        elif line.startswith("-") and not line.startswith("---"):
                            console.print(f"  [red]{line}[/red]")
                        elif line.startswith("@@"):
                            console.print(f"  [cyan]{line}[/cyan]")
                        else:
                            console.print(f"  {line}")

                if verbose:
                    console.print(f"\n  [dim]Before:[/dim] {entry.output_before[:200]}")
                    console.print(f"  [dim]After:[/dim]  {entry.output_after[:200]}")

    # Final verdict
    console.print()
    if report.regression_count == 0:
        console.print("[bold green]No regressions detected.[/bold green]")
    else:
        console.print(
            f"[bold red]{report.regression_count} regression(s) detected "
            f"(drift > {report.threshold:.0%}).[/bold red]"
        )
    console.print()


def report_to_string(report: DriftReport, show_diff: bool = True, verbose: bool = False) -> str:
    """Render the drift report to a string (useful for tests and piping)."""
    from io import StringIO

    string_io = StringIO()
    console = Console(file=string_io, force_terminal=True, width=120)
    print_report(report, console=console, show_diff=show_diff, verbose=verbose)
    return string_io.getvalue()
