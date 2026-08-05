#!/usr/bin/env python3
"""
Dry-run and materialize JIGSAWS LOSO/LOUO split CSVs.

Examples:
  python build_jigsaws_splits.py --split_scheme louo
  python build_jigsaws_splits.py --split_scheme louo --write
  python build_jigsaws_splits.py --split_scheme all
"""

import argparse

from jigsaws_splits import (
    JIGSAWSCrossValSplitBuilder,
    SPLIT_SCHEMES,
    iter_split_schemes,
    make_dataset_variant,
    resolve_split_root,
)


def _print_split_summary(
    builder: JIGSAWSCrossValSplitBuilder,
    split_scheme: str,
    show_train_records: bool = False,
) -> None:
    scheme_name = split_scheme.upper()
    for task in builder.tasks:
        print(f"\n{'=' * 72}")
        print(f"{scheme_name} | task={task}")
        print(f"{'=' * 72}")
        for fold_id in builder.get_fold_ids(split_scheme):
            dataset_variant = make_dataset_variant(task, split_scheme, fold_id)
            summary = builder.summarize_split(task, split_scheme, fold_id)
            held_out_prefix = "trial" if split_scheme == "loso" else "subject"
            held_out_label = f"T{fold_id:02d}" if split_scheme == "loso" else f"S{fold_id:02d}"

            print(
                f"\nFold {fold_id} ({dataset_variant}) -> hold out {held_out_prefix} {held_out_label}"
            )
            print(
                f"  train_count={summary['train_count']} test_count={summary['test_count']}"
            )
            print(f"  train_subjects={summary['train_subjects']}")
            print(f"  test_subjects={summary['test_subjects']}")
            print(f"  train_trials={summary['train_trials']}")
            print(f"  test_trials={summary['test_trials']}")
            print("  held_out_records:")
            for name in summary["test_records"]:
                print(f"    {name}")

            if show_train_records:
                print("  train_records:")
                for name in summary["train_records"]:
                    print(f"    {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run and materialize JIGSAWS LOSO/LOUO split CSVs")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--split_scheme",
        type=str,
        default="louo",
        choices=[*SPLIT_SCHEMES, "all"],
        help="Which cross-validation scheme to inspect or write.",
    )
    parser.add_argument(
        "--split_root",
        type=str,
        default=None,
        help="Output root for split CSVs. Defaults to ./LOSO or ./LOUO based on the scheme.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        default=False,
        help="Write the generated split CSVs after printing the dry-run summary.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing split CSVs when used together with --write.",
    )
    parser.add_argument(
        "--show_train_records",
        action="store_true",
        default=False,
        help="Also print every training record for each fold during the dry run.",
    )
    args = parser.parse_args()

    if args.split_scheme == "all" and args.split_root is not None:
        raise ValueError("--split_root can only be used with a single split scheme, not with --split_scheme all")

    builder = JIGSAWSCrossValSplitBuilder(data_root=args.data_root)

    for split_scheme in iter_split_schemes(args.split_scheme):
        _print_split_summary(builder, split_scheme, show_train_records=args.show_train_records)

        if args.write:
            output_root = resolve_split_root(split_scheme, split_root=args.split_root, repo_root=".")
            written_dirs = builder.materialize(
                split_scheme=split_scheme,
                output_root=output_root,
                force=args.force,
            )
            print(f"\n[INFO] Wrote {len(written_dirs)} split directories under {output_root}")

        resolved_root = resolve_split_root(split_scheme, split_root=args.split_root, repo_root=".")
        issues = builder.validate_materialized_splits(split_scheme, split_root=resolved_root)
        if issues:
            print(f"\n[WARN] Validation issues for {split_scheme.upper()}:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n[INFO] Validation passed for {split_scheme.upper()} at {resolved_root}")


if __name__ == "__main__":
    main()
