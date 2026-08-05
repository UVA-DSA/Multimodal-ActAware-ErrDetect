"""
Shared JIGSAWS split helpers for LOSO and LOUO.

LOSO in this repository is leave-one-trial-out:
  fold i -> hold out all `*_T0{i}.csv` videos.

LOUO in this repository is leave-one-user-out over the shared-subject set:
  folds = S02, S03, S04, S05, S06, S08, S09
  fold i -> hold out all `*_S0{i}_*.csv` videos.
"""

import os
import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple


TASKS: Tuple[str, ...] = ("Suturing", "Needle_Passing")
SPLIT_SCHEMES: Tuple[str, ...] = ("loso", "louo")
LOSO_FOLD_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
LOUO_FOLD_IDS: Tuple[int, ...] = (2, 3, 4, 5, 6, 8, 9)

SCHEME_DIR_NAMES = {
    "loso": os.path.join("splits", "LOSO"),
    "louo": os.path.join("splits", "LOUO"),
}

VIDEO_NAME_PATTERN = re.compile(
    r"^(?P<task>Suturing|Needle_Passing)_S(?P<subject>\d+)_T(?P<trial>\d+)\.csv$"
)


class JIGSAWSVideoRecord(NamedTuple):
    filename: str
    task: str
    subject_id: int
    trial_id: int

    @property
    def subject_token(self) -> str:
        return f"S{self.subject_id:02d}"

    @property
    def trial_token(self) -> str:
        return f"T{self.trial_id:02d}"


class JIGSAWSSplitSpec(NamedTuple):
    task: str
    split_scheme: str
    fold_id: int
    split_root: str

    @property
    def folder_name(self) -> str:
        return f"{self.fold_id}out"

    @property
    def split_dir(self) -> str:
        return os.path.join(self.split_root, self.task, self.folder_name)

    @property
    def dataset_variant(self) -> str:
        return make_dataset_variant(self.task, self.split_scheme, self.fold_id)

    @property
    def held_out_label(self) -> str:
        if self.split_scheme == "loso":
            return f"T{self.fold_id:02d}"
        return f"S{self.fold_id:02d}"


def normalize_split_scheme(split_scheme: str) -> str:
    normalized = str(split_scheme).strip().lower()
    if normalized not in SPLIT_SCHEMES:
        raise ValueError(f"Unsupported split scheme: {split_scheme}. Expected one of {SPLIT_SCHEMES}")
    return normalized


def get_split_fold_ids(split_scheme: str) -> Tuple[int, ...]:
    split_scheme = normalize_split_scheme(split_scheme)
    if split_scheme == "loso":
        return LOSO_FOLD_IDS
    return LOUO_FOLD_IDS


def _validate_task(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"Unsupported JIGSAWS task: {task}. Expected one of {TASKS}")
    return task


def make_dataset_variant(task: str, split_scheme: str, fold_id: int) -> str:
    task = _validate_task(task)
    split_scheme = normalize_split_scheme(split_scheme)
    if fold_id not in get_split_fold_ids(split_scheme):
        raise ValueError(
            f"Unsupported fold {fold_id} for split scheme {split_scheme}. "
            f"Expected one of {get_split_fold_ids(split_scheme)}"
        )
    prefix = "ls" if split_scheme == "loso" else "l"
    return f"{task}-{prefix}{fold_id}"


def build_dataset_variant_map(task: str) -> Dict[str, str]:
    task = _validate_task(task)
    variants: Dict[str, str] = {}
    for fold_id in LOSO_FOLD_IDS:
        variants[make_dataset_variant(task, "loso", fold_id)] = f"leave trial T{fold_id:02d} out"
    for fold_id in LOUO_FOLD_IDS:
        variants[make_dataset_variant(task, "louo", fold_id)] = f"leave subject S{fold_id:02d} out"
    return variants


def parse_dataset_variant(dataset_variant: str) -> Tuple[str, str, int]:
    task, fold_token = dataset_variant.split("-", 1)
    task = _validate_task(task)

    if fold_token.startswith("ls"):
        split_scheme = "loso"
        fold_str = fold_token[2:]
    elif fold_token.startswith("l"):
        split_scheme = "louo"
        fold_str = fold_token[1:]
    else:
        raise ValueError(
            f"Invalid dataset variant: {dataset_variant}. "
            "Expected suffixes like ls1..ls5 (LOSO) or l2/l3/.../l9 (LOUO)."
        )

    if not fold_str.isdigit():
        raise ValueError(f"Invalid fold token in dataset variant: {dataset_variant}")

    fold_id = int(fold_str)
    if fold_id not in get_split_fold_ids(split_scheme):
        raise ValueError(
            f"Unsupported fold {fold_id} for dataset variant {dataset_variant}. "
            f"Expected one of {get_split_fold_ids(split_scheme)}"
        )

    return task, split_scheme, fold_id


def resolve_split_root(split_scheme: str, split_root: Optional[str] = None, repo_root: str = ".") -> str:
    split_scheme = normalize_split_scheme(split_scheme)
    if split_root:
        return split_root
    return os.path.join(repo_root, SCHEME_DIR_NAMES[split_scheme])


def resolve_split_spec(
    dataset_variant: str,
    split_root: Optional[str] = None,
    repo_root: str = ".",
) -> JIGSAWSSplitSpec:
    task, split_scheme, fold_id = parse_dataset_variant(dataset_variant)
    resolved_root = resolve_split_root(split_scheme, split_root=split_root, repo_root=repo_root)
    return JIGSAWSSplitSpec(task=task, split_scheme=split_scheme, fold_id=fold_id, split_root=resolved_root)


def read_split_csv(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_split_csv(csv_path: str, records: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(f"{record}\n")


def load_split_records(
    dataset_variant: str,
    split_root: Optional[str] = None,
    repo_root: str = ".",
) -> Dict[str, object]:
    spec = resolve_split_spec(dataset_variant, split_root=split_root, repo_root=repo_root)
    train_csv = os.path.join(spec.split_dir, "train.csv")
    test_csv = os.path.join(spec.split_dir, "test.csv")
    return {
        "task": spec.task,
        "split_scheme": spec.split_scheme,
        "fold_id": spec.fold_id,
        "split_dir": spec.split_dir,
        "train": read_split_csv(train_csv),
        "test": read_split_csv(test_csv),
    }


def parse_video_record(filename: str) -> JIGSAWSVideoRecord:
    basename = os.path.basename(filename)
    match = VIDEO_NAME_PATTERN.match(basename)
    if match is None:
        raise ValueError(
            f"Could not parse JIGSAWS filename '{filename}'. "
            "Expected names like 'Suturing_S02_T01.csv'."
        )
    return JIGSAWSVideoRecord(
        filename=basename,
        task=match.group("task"),
        subject_id=int(match.group("subject")),
        trial_id=int(match.group("trial")),
    )


class JIGSAWSCrossValSplitBuilder:
    """
    Discover videos from `data/{task}/errors/*.csv` and generate LOSO/LOUO splits.
    """

    def __init__(
        self,
        data_root: str = "./data",
        tasks: Sequence[str] = TASKS,
        shared_subject_ids: Sequence[int] = LOUO_FOLD_IDS,
    ):
        self.data_root = data_root
        self.tasks = tuple(_validate_task(task) for task in tasks)
        self.shared_subject_ids = tuple(int(subject_id) for subject_id in shared_subject_ids)
        self.records_by_task = {
            task: self._discover_task_records(task)
            for task in self.tasks
        }
        self._records_lookup = {
            task: {record.filename: record for record in records}
            for task, records in self.records_by_task.items()
        }
        self._validate_shared_subjects()

    def _error_dir(self, task: str) -> str:
        return os.path.join(self.data_root, task, "errors")

    def _discover_task_records(self, task: str) -> List[JIGSAWSVideoRecord]:
        error_dir = self._error_dir(task)
        if not os.path.exists(error_dir):
            raise FileNotFoundError(f"Missing error CSV directory for task {task}: {error_dir}")

        records: List[JIGSAWSVideoRecord] = []
        for name in sorted(os.listdir(error_dir)):
            if not name.endswith(".csv"):
                continue
            record = parse_video_record(name)
            if record.task != task:
                raise ValueError(
                    f"Found mismatched task filename in {error_dir}: {name} belongs to {record.task}"
                )
            records.append(record)

        return sorted(records, key=lambda record: (record.subject_id, record.trial_id, record.filename))

    def _validate_shared_subjects(self) -> None:
        for task in self.tasks:
            available_subjects = {record.subject_id for record in self.records_by_task[task]}
            missing_subjects = [subject_id for subject_id in self.shared_subject_ids if subject_id not in available_subjects]
            if missing_subjects:
                missing_tokens = ", ".join(f"S{subject_id:02d}" for subject_id in missing_subjects)
                raise ValueError(
                    f"Task {task} is missing required shared LOUO subjects: {missing_tokens}"
                )

    def get_fold_ids(self, split_scheme: str) -> Tuple[int, ...]:
        split_scheme = normalize_split_scheme(split_scheme)
        if split_scheme == "loso":
            return LOSO_FOLD_IDS
        return self.shared_subject_ids

    def build_split_records(self, task: str, split_scheme: str, fold_id: int) -> Tuple[List[str], List[str]]:
        task = _validate_task(task)
        split_scheme = normalize_split_scheme(split_scheme)
        if fold_id not in self.get_fold_ids(split_scheme):
            raise ValueError(
                f"Unsupported fold {fold_id} for split scheme {split_scheme}. "
                f"Expected one of {self.get_fold_ids(split_scheme)}"
            )

        train_records: List[str] = []
        test_records: List[str] = []
        for record in self.records_by_task[task]:
            is_held_out = (
                record.trial_id == fold_id
                if split_scheme == "loso"
                else record.subject_id == fold_id
            )
            if is_held_out:
                test_records.append(record.filename)
            else:
                train_records.append(record.filename)

        if not test_records:
            raise ValueError(
                f"No held-out records found for task {task}, split scheme {split_scheme}, fold {fold_id}"
            )
        return train_records, test_records

    def summarize_split(self, task: str, split_scheme: str, fold_id: int) -> Dict[str, object]:
        train_records, test_records = self.build_split_records(task, split_scheme, fold_id)
        lookup = self._records_lookup[task]
        train_meta = [lookup[name] for name in train_records]
        test_meta = [lookup[name] for name in test_records]

        return {
            "train_count": len(train_records),
            "test_count": len(test_records),
            "train_subjects": sorted({record.subject_token for record in train_meta}),
            "test_subjects": sorted({record.subject_token for record in test_meta}),
            "train_trials": sorted({record.trial_token for record in train_meta}),
            "test_trials": sorted({record.trial_token for record in test_meta}),
            "train_records": train_records,
            "test_records": test_records,
        }

    def materialize(
        self,
        split_scheme: str,
        output_root: Optional[str] = None,
        repo_root: str = ".",
        force: bool = False,
    ) -> List[str]:
        split_scheme = normalize_split_scheme(split_scheme)
        root = resolve_split_root(split_scheme, split_root=output_root, repo_root=repo_root)
        written_dirs: List[str] = []

        for task in self.tasks:
            for fold_id in self.get_fold_ids(split_scheme):
                spec = JIGSAWSSplitSpec(task=task, split_scheme=split_scheme, fold_id=fold_id, split_root=root)
                train_records, test_records = self.build_split_records(task, split_scheme, fold_id)
                train_csv = os.path.join(spec.split_dir, "train.csv")
                test_csv = os.path.join(spec.split_dir, "test.csv")

                if not force and (os.path.exists(train_csv) or os.path.exists(test_csv)):
                    raise FileExistsError(
                        f"Split files already exist under {spec.split_dir}. "
                        "Pass force=True to overwrite them."
                    )

                write_split_csv(train_csv, train_records)
                write_split_csv(test_csv, test_records)
                written_dirs.append(spec.split_dir)

        return written_dirs

    def validate_materialized_splits(
        self,
        split_scheme: str,
        split_root: Optional[str] = None,
        repo_root: str = ".",
    ) -> List[str]:
        split_scheme = normalize_split_scheme(split_scheme)
        resolved_root = resolve_split_root(split_scheme, split_root=split_root, repo_root=repo_root)
        issues: List[str] = []

        for task in self.tasks:
            for fold_id in self.get_fold_ids(split_scheme):
                dataset_variant = make_dataset_variant(task, split_scheme, fold_id)
                expected_train, expected_test = self.build_split_records(task, split_scheme, fold_id)
                try:
                    actual_split = load_split_records(dataset_variant, split_root=resolved_root, repo_root=repo_root)
                except FileNotFoundError as exc:
                    issues.append(str(exc))
                    continue

                actual_train = list(actual_split["train"])
                actual_test = list(actual_split["test"])
                if actual_train != expected_train:
                    issues.append(
                        f"Train split mismatch for {dataset_variant}: "
                        f"expected {len(expected_train)} records, got {len(actual_train)}"
                    )
                if actual_test != expected_test:
                    issues.append(
                        f"Test split mismatch for {dataset_variant}: "
                        f"expected {len(expected_test)} records, got {len(actual_test)}"
                    )

        return issues


def iter_split_schemes(split_scheme: str) -> Tuple[str, ...]:
    if str(split_scheme).strip().lower() == "all":
        return SPLIT_SCHEMES
    return (normalize_split_scheme(split_scheme),)
