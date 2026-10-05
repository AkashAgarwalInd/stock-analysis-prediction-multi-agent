#!/usr/bin/env python3
"""End-to-end demo on three NSE stocks (Plan.md Phase 18 / §76).

For each stock: a point-in-time weekly backtest, the track record and
calibration reports, then today's forecast reviewed against the backtest's
history. Reports land in data/demo/<time>/<TICKER>/ and a summary is printed.

    python demo.py                          # RELIANCE INFY HDFCBANK, 12 weeks
    python demo.py --tickers TCS ITC --weeks 8
    python demo.py --no-llm                 # quant only: no API key needed
    python demo.py --show-reports           # also print each analysis report

Educational and research use only; not investment advice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from stock_analysis.config import get_settings
from stock_analysis.demo import DEFAULT_TICKERS, render_demo_summary, run_demo
from stock_analysis.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS), help="NSE tickers or company names")
    parser.add_argument("--weeks", type=int, default=12, help="backtest weeks per stock (default 12)")
    llm = parser.add_mutually_exclusive_group()
    llm.add_argument("--llm", dest="llm", action="store_true", default=None, help="use the Gemini LLM path")
    llm.add_argument("--no-llm", dest="llm", action="store_false", help="quant only (default without GEMINI_API_KEY)")
    parser.add_argument("--output-dir", type=Path, default=None, help="where to write databases and reports")
    parser.add_argument("--show-reports", action="store_true", help="print each stock's analysis report")
    args = parser.parse_args(argv)

    configure_logging()
    console = Console()
    use_llm = args.llm if args.llm is not None else bool(get_settings().gemini_api_key)
    if use_llm and not get_settings().gemini_api_key:
        console.print("GEMINI_API_KEY is not set: the LLM steps will degrade to the quant baseline.")
    console.print(f"Demo: {', '.join(args.tickers)} · {args.weeks} weeks · LLM {'on' if use_llm else 'off'}")

    results = run_demo(
        args.tickers,
        weeks=args.weeks,
        output_dir=args.output_dir,
        use_llm=use_llm,
        progress=lambda msg: console.print(msg, markup=False),
    )
    if args.show_reports:
        for result in results:
            if result.analysis is not None and result.analysis.report:
                console.print(Markdown(result.analysis.report))
    console.print(Markdown(render_demo_summary(results)))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
