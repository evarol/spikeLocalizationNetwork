"""Generate the convergence-trace plot from a SLN+DREDge history.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convergence_trace import ConvergenceStyle, load_history, save_panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True, type=Path,
                   help="Path to history.csv from a SLN+DREDge run.")
    p.add_argument("--config", default=None, type=Path,
                   help="Path to config.json (to read lambdas). Optional.")
    p.add_argument("--out", default="figures/convergence_trace.png", type=Path)
    p.add_argument("--label", default="SLN+DREDge convergence", help="title prefix")
    p.add_argument("--lambda_corr", type=float, default=1.0)
    p.add_argument("--lambda_ent", type=float, default=0.1)
    p.add_argument("--lambda_teth", type=float, default=0.01)
    args = p.parse_args()

    if args.config is not None and args.config.exists():
        with open(args.config) as f:
            cfg = json.load(f)
        args.lambda_corr = float(cfg.get("lambda_corr", args.lambda_corr))
        args.lambda_ent = float(cfg.get("lambda_ent", args.lambda_ent))
        args.lambda_teth = float(cfg.get("lambda_teth", args.lambda_teth))
        print(f"  loaded lambdas from {args.config}: "
              f"ρ={args.lambda_corr}, H={args.lambda_ent}, teth={args.lambda_teth}")

    df = load_history(args.history)
    print(f"  {len(df)} rows · outer_iters {sorted(df.outer_iter.unique())}")
    title = (f"{args.label}\n{args.history}  ·  "
             f"λ_ρ={args.lambda_corr}, λ_H={args.lambda_ent}, λ_teth={args.lambda_teth}")
    save_panel(args.out, df, args.lambda_corr, args.lambda_ent, args.lambda_teth,
               title=title, style=ConvergenceStyle())
    print(f"  wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
