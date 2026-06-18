import argparse
import sys
from src.download_github_data import download_github_data
from src.download_kaggle_data import download_kaggle_data
from src.audit_data import audit_data
from src.build_features import build_features
from src.predict_scorelines import predict_scorelines
from src.simulate_tournament import simulate_tournament

def main():
    parser = argparse.ArgumentParser(description="World Cup Score Predictor CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # download-data  (GitHub + Kaggle combined)
    parser_download = subparsers.add_parser(
        "download-data", help="Download all raw data (GitHub + Kaggle)"
    )
    parser_download.add_argument(
        "--force", action="store_true",
        help="Force re-download even if files already exist"
    )

    # download-kaggle  (Kaggle only)
    parser_kaggle = subparsers.add_parser(
        "download-kaggle",
        help="Download Kaggle player-scores dataset via direct HTTP (no credentials needed)"
    )
    parser_kaggle.add_argument(
        "--force", action="store_true",
        help="Force re-download even if files already exist"
    )

    # audit
    subparsers.add_parser("audit", help="Audit raw data and produce a JSON report")

    # build-features
    subparsers.add_parser("build-features", help="Build historical and squad features")

    # evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate predictions against actuals")
    parser_eval.add_argument("--as-of-date", default="auto", help="Evaluation date cutoff")
    parser_eval.add_argument("--actuals-source", default="martj42", help="Source for actuals")

    # predict-live
    parser_live = subparsers.add_parser("predict-live", help="Predict scorelines using all available data")
    parser_live.add_argument("--as-of-date", default="auto", help="Date cutoff for using historical results")

    # backtest
    parser_backtest = subparsers.add_parser("backtest", help="Evaluate pre-tournament model quality")
    parser_backtest.add_argument("--train-cutoff", required=True, help="Training cutoff date (e.g. 2026-06-10)")
    parser_backtest.add_argument("--actuals-source", default="martj42", help="Source for actuals")

    # simulate-tournament
    parser_sim = subparsers.add_parser("simulate-tournament", help="Simulate the entire tournament")
    parser_sim.add_argument("--runs", type=int, default=10000, help="Number of Monte Carlo simulations")

    args = parser.parse_args()

    # Parse dynamic dates
    from datetime import date
    def parse_date(d_str):
        if d_str == "auto" or d_str is None:
            return date.today().isoformat()
        return d_str

    if args.command == "download-data":
        download_github_data()
        ok = download_kaggle_data(force=args.force)
        if not ok:
            print("\n[cli] Kaggle download FAILED. See messages above.")
            sys.exit(1)
    elif args.command == "download-kaggle":
        ok = download_kaggle_data(force=args.force)
        if not ok:
            print("\n[cli] Kaggle download FAILED. See messages above.")
            sys.exit(1)
    elif args.command == "audit":
        audit_data()
    elif args.command == "build-features":
        build_features()
    elif args.command == "predict-live":
        from src.predict_scorelines import predict_scorelines
        predict_scorelines(mode="live", as_of_date=parse_date(args.as_of_date))
    elif args.command == "backtest":
        from src.predict_scorelines import predict_scorelines
        predict_scorelines(mode="backtest", train_cutoff=args.train_cutoff)
    elif args.command == "evaluate":
        from src.evaluate_predictions import evaluate_predictions
        evaluate_predictions(as_of_date=parse_date(args.as_of_date))
    elif args.command == "simulate-tournament":
        simulate_tournament(runs=args.runs)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
