"""Typed paths for a reviewed collector campaign and its syncable artifacts."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReviewedPoolPaths:
    campaign_dir: Path
    jobs: Path
    output: Path
    control: Path
    worker_status: Path
    supervisor_journal: Path
    case_events: Path
    hard_abort: Path
    authorization: Path

    @classmethod
    def for_campaign(cls, campaign_dir, *, control_name='control', authorization_name='bundle/root-pool-authorization.json'):
        """Build sync-compatible artifact paths rooted in a dedicated campaign directory.

        A campaign owns the canonical sync names. `control_name` may be versioned
        when several reviewed transitions share the same campaign directory.
        """
        root = Path(campaign_dir).expanduser().resolve()
        if not control_name or Path(control_name).name != control_name or control_name in {'.', '..'}:
            raise ValueError('control_name must be one simple directory name')
        authorization = Path(authorization_name)
        if authorization.is_absolute() or '..' in authorization.parts:
            raise ValueError('authorization_name must be a relative campaign path')
        paths = cls(
            campaign_dir=root,
            jobs=root / 'jobs',
            output=root / 'results.jsonl',
            control=root / control_name,
            worker_status=root / 'worker-status.json',
            supervisor_journal=root / 'journal-pool.jsonl',
            case_events=root / 'journal-cases.jsonl',
            hard_abort=root / 'hard-abort.request',
            authorization=root / authorization,
        )
        paths.validate()
        return paths

    def validate(self):
        root = self.campaign_dir.resolve()
        if self.jobs.name != 'jobs' or self.output.name != 'results.jsonl':
            raise ValueError('jobs/output must use sync importer names `jobs/` and `results.jsonl`')
        if self.worker_status.name != 'worker-status.json':
            raise ValueError('worker status must use sync importer name `worker-status.json`')
        self.validate_journals(self.supervisor_journal, self.case_events)
        if self.jobs.resolve() != root / 'jobs' or self.output.resolve() != root / 'results.jsonl':
            raise ValueError('jobs/output must be direct children of the campaign root')
        if self.worker_status.resolve() != root / 'worker-status.json':
            raise ValueError('worker status must be a direct child of the campaign root')
        if self.supervisor_journal.parent != root or self.case_events.parent != root:
            raise ValueError('canonical journals must be direct children of the campaign root')
        if self.control.resolve().parent != root:
            raise ValueError('control directory must be a direct child of the campaign')
        if self.hard_abort.name != 'hard-abort.request' or self.hard_abort.resolve().parent != root:
            raise ValueError('hard abort file must use canonical campaign-root name `hard-abort.request`')
        if not self.authorization.resolve().is_relative_to(root):
            raise ValueError('authorization path must remain inside the campaign directory')
        file_paths = (self.output, self.worker_status, self.supervisor_journal,
                      self.case_events, self.hard_abort, self.authorization)
        if len(set(file_paths)) != len(file_paths):
            raise ValueError('reviewed pool artifact paths collide')
        return self

    @classmethod
    def from_runner_args(cls, args):
        """Validate a parsed reviewed_single_pool.py namespace."""
        supervisor = Path(args.journal).resolve()
        root = supervisor.parent
        return cls(
            campaign_dir=root,
            jobs=Path(args.jobs).resolve(),
            output=Path(args.output).resolve(),
            control=Path(args.control_dir).resolve(),
            worker_status=Path(args.status_file).resolve(),
            supervisor_journal=supervisor,
            case_events=Path(args.case_events_file).resolve(),
            hard_abort=Path(args.hard_abort_file).resolve() if args.hard_abort_file else root / 'hard-abort.request',
            authorization=Path(args.authorization).resolve(),
        ).validate()

    @staticmethod
    def validate_journals(supervisor_journal, case_events):
        supervisor = Path(supervisor_journal).resolve()
        cases = Path(case_events).resolve()
        if supervisor.name != 'journal-pool.jsonl' or cases.name != 'journal-cases.jsonl':
            raise ValueError('supervisor/case journal basenames must be journal-pool.jsonl and journal-cases.jsonl')
        if supervisor == cases:
            raise ValueError('supervisor and case journals must be distinct files')
        return supervisor, cases

    def authorization_fields(self):
        """Return the reviewed_single_pool.py authorization path fields."""
        return {
            'jobs_path': str(self.jobs),
            'output_path': str(self.output),
            'control_dir': str(self.control),
            'worker_status_path': str(self.worker_status),
            'supervisor_journal_path': str(self.supervisor_journal),
            'case_events_path': str(self.case_events),
            'hard_abort_file': str(self.hard_abort),
            'authorization_path': str(self.authorization),
            'pool_status_path': str(self.control / 'pool-status.json'),
        }

    def cli_args(self):
        """Return the path-related CLI arguments accepted by reviewed_single_pool.py."""
        self.validate()
        pairs = (
            ('--jobs', self.jobs), ('--output', self.output), ('--control-dir', self.control),
            ('--hard-abort-file', self.hard_abort), ('--status-file', self.worker_status),
            ('--journal', self.supervisor_journal), ('--case-events-file', self.case_events),
            ('--authorization', self.authorization),
        )
        return [value for flag, path in pairs for value in (flag, str(path))]
