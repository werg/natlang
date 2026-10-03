"""The S1 dataset ledger: every dataset in bgkit, Schnitzeljagd, natlang and on the external drive
with a disposition and reason (S1 §3, work item 1).

bgkit task stores and Schnitzeljagd corpora are enumerated from disk (rows and bytes measured);
everything else is listed explicitly. Dispositions:

* ``migrated`` — a converter exists and produces validated port records (scripts/neuralese_data).
* ``queued`` — assigned to a named converter or Natlang task adapter that does not exist yet.
* ``replay-only`` — ordinary-text replay against capability loss.
* ``eval-only`` — protected held-out evaluation; scanned out of training.
* ``source`` — raw upstream material that converters read; not itself a training product.
* ``excluded`` — with a reason (superseded, not portable, not data, retrieval framing, …).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import bgkit, schnitzel, schnitzel_turns

BGKIT_TASKS = bgkit.TASKS_DIR
SDKB_CORPORA = schnitzel.CORPORA_DIR
SDKB_RAW = Path("/mnt/external/sdkb-archive/raw")
LEDGER_VERSION = "natlang.neuralese-data-ledger/1"

# bgkit store family (version suffix removed) → (port family or None, disposition, converter/adapter, notes, licence)
BGKIT_FAMILIES = {
    "qa_squad": ("qa_extractive", "migrated", "bgkit.qa", "Group by article.", "CC-BY-SA-4.0"),
    "qa_squad_article": ("qa_extractive_article", "migrated", "bgkit.qa", "Article windows of SQuAD; same questions as qa_squad.", "CC-BY-SA-4.0"),
    "qa_hotpotqa": ("qa_multihop", "migrated", "bgkit.qa", "", "CC-BY-SA-4.0"),
    "qa_hotpotqa_distractor": ("qa_multihop_distractor", "migrated", "bgkit.qa", "", "CC-BY-SA-4.0"),
    "qa_triviaqa": ("qa_trivia", "migrated", "bgkit.qa", "", "Apache-2.0"),
    "qa_narrativeqa": ("qa_narrative", "migrated", "bgkit.qa", "", "Apache-2.0"),
    "qa_quality_mcq": ("qa_mcq", "migrated", "bgkit.qa", "Choice format re-derived from meta options.", "CC-BY-4.0"),
    "qa_quality": (None, "excluded", None, "Superseded by qa_quality_mcq (format defect in free-form variant).", "CC-BY-4.0"),
    "qa_multineedle": ("qa_needles", "migrated", "bgkit.qa", "SQuAD passages among Wikipedia filler; grouped by needle document.", "CC-BY-SA-4.0"),
    "qa_multineedle_long": (None, "excluded", None, "Superseded by qa_multineedle_long_fam.", "LicenseRef-mixed"),
    "qa_multineedle_long_fam": ("qa_needles_long", "migrated", "bgkit.qa", "SQuAD passages among Wikipedia filler; grouped by needle document.", "CC-BY-SA-4.0"),
    "babi_filler": ("qa_babi_filler", "migrated", "bgkit.qa", "bAbI stories in wikitext filler; grouped by story.", "LicenseRef-mixed"),
    "babilong": (None, "eval-only", None, "Eval-only upstream.", "Apache-2.0"),
    "chatqa2_long_sft": ("qa_long_context", "migrated", "bgkit.slot_qa", "Slots become sources, <|reserved_6|> sentinels dropped; licence not verified locally (no card).", "LicenseRef-unverified"),
    "chatqa2_narrativeqa": ("qa_narrative_long", "migrated", "bgkit.slot_qa", "NarrativeQA in ChatQA2 framing; slots become sources; licence not verified locally.", "LicenseRef-unverified"),
    "compaction": (None, "excluded", None, "bgkit segment/spine construction; continuation records are rebuilt from upstream trajectories (swe_compaction).", "CC-BY-4.0"),
    "compaction_chunks_5src_100k": (None, "excluded", None, "bgkit spine chunks; rebuilt from upstream trajectories.", "LicenseRef-mixed"),
    "compaction_chunks_long_5src_fam": (None, "excluded", None, "bgkit spine chunks; rebuilt from upstream trajectories.", "LicenseRef-mixed"),
    "chunks_agenttrove": (None, "excluded", None, "bgkit chunking of agenttrove; rebuild from trajectories_ext/agenttrove.", "LicenseRef-mixed"),
    "chunks_nebius": (None, "excluded", None, "bgkit chunking of SWE-rebench OpenHands; rebuilt by swe_compaction.", "CC-BY-4.0"),
    "chunks_openresearcher": (None, "excluded", None, "bgkit chunking; rebuild from trajectories_ext/openresearcher.", "LicenseRef-noncommercial"),
    "chunks_open_swe": (None, "excluded", None, "bgkit chunking; rebuild from trajectories_ext/open_swe_traces.", "LicenseRef-mixed"),
    "chunks_long_agenttrove": (None, "excluded", None, "bgkit chunking; rebuild from upstream.", "LicenseRef-mixed"),
    "chunks_long_nebius": (None, "excluded", None, "bgkit chunking; rebuilt by swe_compaction.", "CC-BY-4.0"),
    "chunks_long_openresearcher": (None, "excluded", None, "bgkit chunking; rebuild from upstream.", "LicenseRef-noncommercial"),
    "chunks_long_open_swe": (None, "excluded", None, "bgkit chunking; rebuild from upstream.", "LicenseRef-mixed"),
    "chunks_long_swezero": (None, "excluded", None, "bgkit chunking; rebuild from capability_packaging/trajectories_ext/swe_zero.", "LicenseRef-mixed"),
    "tool_digest": ("tool_digest", "migrated", "bgkit.tool_digest", "", "LicenseRef-repository-content"),
    "tool_digest_ext": ("tool_digest", "migrated", "bgkit.tool_digest", "Promptable digests: compare material.", "LicenseRef-repository-content"),
    "tool_digest_lfm_swe": ("tool_digest", "queued", "bgkit.tool_digest", "LFM2.5-8B teacher digests in bgkit's harness.", "LicenseRef-repository-content"),
    "tool_slots": (None, "excluded", None, "bgkit slot/spine framing of upstream trajectories; continuation records are rebuilt from upstream (agents, swe_compaction).", "LicenseRef-mixed"),
    "tool_slots_merged": (None, "excluded", None, "Intermediate merge of tool_slots variants.", "LicenseRef-mixed"),
    "tool_slots_ours": (None, "excluded", None, "Superseded by tool_slots_synth_v4_mapped.", "LicenseRef-mixed"),
    "tool_slots_ours2": (None, "excluded", None, "Superseded by tool_slots_synth_v4_mapped.", "LicenseRef-mixed"),
    "tool_slots_synth_ours": (None, "excluded", None, "Superseded by tool_slots_synth_v4_mapped.", "LicenseRef-mixed"),
    "tool_slots_synth": (None, "excluded", None, "Superseded by tool_slots_synth_v4_mapped.", "LicenseRef-mixed"),
    "tool_slots_synth_mapped": (None, "excluded", None, "bgkit slot/spine framing of upstream trajectories; continuation records are rebuilt from upstream (agents, swe_compaction).", "LicenseRef-mixed"),
    "agent_tool_slots": (None, "excluded", None, "Synthetic walks in bgkit's tool/zoom harness; the underlying repo QA is migrated as repo_qa.", "LicenseRef-mixed"),
    "web_search_r1": (None, "excluded", None, "bgkit framing; rebuilt from web_sft Search-R1 by agents.search_r1.", "LicenseRef-unverified"),
    "web_openresearcher": (None, "excluded", None, "bgkit framing; rebuilt from web_sft OpenResearcher (MIT) by agents.openresearcher.", "MIT"),
    "web_openseeker": (None, "excluded", None, "bgkit framing; rebuilt from web_sft openseeker-miroverse by agents.openseeker_miroverse.", "LicenseRef-openseeker"),
    "web_webshaper": (None, "excluded", None, "bgkit framing; rebuilt from web_sft WebShaper by agents.webshaper.", "LicenseRef-unverified"),
    "web_sds": ("trajectory_continuation_research", "queued", "upstream.simple_deep_searcher", "SimpleDeepSearcher upstream uses a custom format; adapter not written.", "LicenseRef-unverified"),
    "web_miroverse_nc": (None, "excluded", None, "bgkit framing; MiroVerse rows rebuilt from web_sft openseeker-miroverse (CC-BY-NC-4.0).", "CC-BY-NC-4.0"),
    "memory_recall": ("memory_recall", "migrated", "bgkit.slot_qa", "Conversation Chronicles sessions become sources; memory-tool protocol dropped; teacher questions. Licence to be confirmed.", "LicenseRef-unverified"),
    "memory_qa_pairs": ("memory_qa", "migrated", "bgkit.slot_qa", "Conversation Chronicles sessions become sources; memory-tool protocol dropped. Licence to be confirmed.", "LicenseRef-unverified"),
    "memory_qa_user_assistant": ("memory_qa_user_assistant", "migrated", "bgkit.slot_qa", "Conversation Chronicles sessions become sources; memory-tool protocol dropped. Licence to be confirmed.", "LicenseRef-unverified"),
    "memory_qa_s2_ua": ("memory_qa_user_assistant", "migrated", "bgkit.slot_qa", "S2-era user/assistant memory QA; same treatment as memory_qa_user_assistant. Licence to be confirmed.", "LicenseRef-unverified"),
    "memory_agent": (None, "excluded", None, "Same episodes as memory_qa; the rest is bgkit's index/search tool protocol.", "LicenseRef-mixed"),
    "repo_qa_file": ("repo_qa", "migrated", "bgkit.repo", "Repo-disjoint held-out list applied.", "LicenseRef-repository-content"),
    "repo_localisation": ("repo_localisation", "migrated", "bgkit.repo", "Repo-disjoint held-out list applied.", "LicenseRef-repository-content"),
    "repo_overview": ("repo_overview", "migrated", "bgkit.overview", "Locate the file for a question in a repository map; held-out repos to test.", "LicenseRef-repository-content"),
    "file_overview": ("file_overview", "migrated", "bgkit.overview", "File overviews written by a bgkit teacher.", "LicenseRef-repository-content"),
    "repo_tree": ("repo_tree", "queued", "bgkit.repo_tree", "Questions and file lists; .roots.safetensors are encoder outputs (excluded).", "LicenseRef-repository-content"),
    "code_swe_tsjs": (None, "excluded", None, "bgkit framing; rebuilt from trajectories_ext/open_swe_traces_ts_think by agents.open_swe_traces.", "CC-BY-4.0"),
    "code_swe_opencode": (None, "excluded", None, "bgkit framing; rebuilt from trajectories_ext/nemotron_sft_swe_v35_filtered by agents.nemotron_swe.", "CC-BY-4.0"),
    "agent_swe_lfm": (None, "excluded", None, "LFM2.5-8B teacher runs inside bgkit's agent harness; harness-specific protocol, no harness-free upstream.", "LicenseRef-mixed"),
    "chat_replay": (None, "replay-only", "replay", "Ordinary chat replay.", "LicenseRef-mixed"),
    "read_replay": (None, "replay-only", "replay", "Ordinary reading replay.", "LicenseRef-mixed"),
    "read_replay_long": (None, "replay-only", "replay", "Long-context reading replay.", "LicenseRef-mixed"),
    "reasoning_replay": (None, "replay-only", "replay", "OpenThoughts.", "Apache-2.0"),
}
SIDECAR_SUFFIXES = (".stats.json", ".report.json", ".repos.parquet", ".roots.safetensors")


def _base_and_version(name: str) -> tuple[str, int, str]:
    """Store family and version: `tool_slots_synth_v3b_mapped.parquet` → ('tool_slots_synth_mapped', 3, 'b')."""
    stem = name
    for suffix in (".parquet", ".arrow", ".jsonl", ".txt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = stem.removesuffix("_heldout")
    m = re.search(r"_v(\d+)([a-z]?)(?=_|$)", stem)
    if not m:
        return stem, 0, ""
    return stem[: m.start()] + stem[m.end():], int(m.group(1)), m.group(2)


def _rows(path: Path) -> int | None:
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            return pq.ParquetFile(str(path)).metadata.num_rows
        if path.suffix == ".arrow":
            import pyarrow as pa

            with pa.memory_map(str(path)) as source:  # count batch by batch; never materialise the store
                try:
                    reader = pa.ipc.open_file(source)
                    return sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
                except pa.ArrowInvalid:
                    source.seek(0)
                    return sum(batch.num_rows for batch in pa.ipc.open_stream(source))
        if path.suffix in (".jsonl", ".txt"):
            with open(path, "rb") as stream:
                return sum(1 for _ in stream)
    except Exception:  # noqa: BLE001 - ledger records unreadable files instead of failing
        return None
    return None


def bgkit_task_entries(tasks_dir: Path = BGKIT_TASKS) -> list[dict]:
    files = sorted(p for p in tasks_dir.iterdir() if p.is_file() and not p.name.endswith(SIDECAR_SUFFIXES))
    groups: dict[str, list[Path]] = {}
    for p in files:
        if p.suffix == ".arrow" and p.with_suffix(".parquet").exists():
            continue  # identical mirror of the parquet store
        base, _, _ = _base_and_version(p.name)
        groups.setdefault(base, []).append(p)
    entries = []
    for base, paths in sorted(groups.items()):
        lookup = base
        if lookup not in BGKIT_FAMILIES:
            lookup = re.sub(r"_\d+$", "", base)
        if base in ("eval_repos", "train_repos_used"):
            info = (None, "source", "splits", "Repository split list used for closure.", "LicenseRef-none")
        elif base.startswith("repo_") and base.endswith("_tasks"):
            info = (None, "source", "bgkit.repo", "Task definitions (repo, commit, verifier) used by the repo converters for lineage.", "LicenseRef-repository-content")
        elif base in ("eval_repos_heldout", "train_repos_used"):
            info = (None, "source", "splits", "Repository split list used for closure.", "LicenseRef-none")
        else:
            info = BGKIT_FAMILIES.get(lookup)
        ordered = sorted(paths, key=lambda q: (_base_and_version(q.name)[1:], "heldout" in q.name))
        non_heldout = [q for q in ordered if "heldout" not in q.name and _rows(q) != 0]
        latest = non_heldout[-1] if non_heldout else ordered[-1]
        for p in ordered:
            family, disposition, converter, notes, spdx = info or (None, "queued", "review", "Unclassified store: needs review.", "LicenseRef-unknown")
            reason = notes
            rows = _rows(p)
            if rows == 0:
                disposition, reason, converter, family = "excluded", "Empty store.", None, None
            elif p != latest and "heldout" not in p.name and disposition not in ("source",):
                disposition, reason, converter = "excluded", f"Superseded by {latest.name}.", None
            if "heldout" in p.name and disposition in ("migrated", "queued"):
                disposition, converter = "eval-only", None
                reason = "Repo-disjoint held-out store; protected from training."
            entries.append({
                "id": f"bgkit:tasks:{p.name}", "project": "bgkit", "location": str(p), "kind": "task-store",
                "bytes": p.stat().st_size, "rows": rows, "family": family, "disposition": disposition,
                "converter": converter, "reason": reason, "license": spdx,
                "mirrors": [str(p.with_suffix(".arrow"))] if p.suffix == ".parquet" and p.with_suffix(".arrow").exists() else [],
            })
    for d in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        entries.append({"id": f"bgkit:tasks:{d.name}/", "project": "bgkit", "location": str(d), "kind": "shards",
                        "bytes": None, "rows": None, "family": None, "disposition": "excluded", "converter": None,
                        "reason": "Shard directory of an earlier store build; the consolidated store is listed instead.",
                        "license": "LicenseRef-mixed", "mirrors": []})
    return entries


SDKB_EXCLUDED_PREFIXES = {
    "memory-": "Transcript render of the matching tasks-* corpus in Schnitzeljagd's memory-tool protocol; the tasks-* corpus is converted instead.",
    "squad-": "SQuAD bank/four-space/routing framings; upstream questions are covered by bgkit QA records.",
    "hotpot-": "HotpotQA four-space/spatial bank framings; covered by bgkit QA records.",
    "public": "Retrieval bank corpora (2WikiMultihopQA, HoVer, MuSiQue retrieval framing); retrieval-key training is not used.",
    "r5-mixed": "Mixed bank corpus for KB retrieval.",
    "r6-mixed": "Mixed bank corpus for KB retrieval.",
    "four-space": "Four-space bank configuration.",
}
SDKB_MULTI_TURN = {
    "tasks-agentbank-20260927": "natlang.function_calling",
    "tasks-agentinstruct-20260927": "natlang.function_calling",
    "tasks-apigen-mt-20260927": "natlang.function_calling",
    "tasks-alfworld-20260927": "natlang.worlds",
    "tasks-alfworld-bg-20260927": "natlang.worlds",
    "tasks-scienceworld-20260927": "natlang.worlds",
    "tasks-scienceworld-bg-20260927": "natlang.worlds",
    "tasks-webshop-20260927": "natlang.worlds",
}
SDKB_ADAPTERS = {
    "spider": "natlang.text_to_sql", "bird": "natlang.text_to_sql", "xlam": "natlang.function_calling",
    "kodcode": "natlang.code", "knights": "natlang.reasoning", "synlogic": "natlang.reasoning",
    "reasoning-gym": "natlang.reasoning",
}


def sdkb_corpus_entries(root: Path = SDKB_CORPORA) -> list[dict]:
    entries = []
    converted = set(schnitzel.CORPORA)
    for p in sorted(root.iterdir()):
        name = p.name
        entry = {"id": f"schnitzeljagd:corpora:{name}", "project": "schnitzeljagd", "location": str(p),
                 "kind": "corpus" if p.is_dir() else "file", "bytes": None if p.is_dir() else p.stat().st_size,
                 "rows": None, "family": None, "converter": None, "license": None}
        if p.is_file():
            entry.update(disposition="excluded", reason="Build log or configuration, not data.")
        elif name in converted:
            manifest = json.loads((p / "manifest.json").read_text()) if (p / "manifest.json").exists() else {}
            episodes = manifest.get("episodes") or {}
            domain = manifest.get("domain")
            family, _, lic = schnitzel.DOMAINS.get(domain, (None, None, ("LicenseRef-unknown", False, "")))
            adapter = next((a for k, a in SDKB_ADAPTERS.items() if k in name), None)
            entry.update(rows=sum(episodes.values()) if episodes else None, family=family, disposition="migrated",
                         converter="schnitzel.episodes" + (f"; also queued for {adapter}" if adapter else ""),
                         reason="Single-turn episodes converted to consume records; supports become sources.",
                         license=lic[0], episodes=episodes)
        elif name in SDKB_MULTI_TURN:
            manifest = json.loads((p / "manifest.json").read_text()) if (p / "manifest.json").exists() else {}
            domain = manifest.get("domain") or name.split("-")[1]
            family, _, lic, _ = schnitzel_turns.DOMAINS.get(domain.replace("-", "_"), (None, None, ("LicenseRef-unknown", False, ""), None))
            entry.update(rows=sum((manifest.get("episodes") or {}).values()) or None, disposition="migrated",
                         family=family,
                         converter=f"schnitzel_turns; also queued for {SDKB_MULTI_TURN[name]}",
                         reason="Multi-turn episodes converted to causal continuation windows; supports become background sources. An executable adapter is still queued.",
                         license=lic[0], episodes=manifest.get("episodes"))
        elif "inverse-cloze" in name:
            entry.update(disposition="excluded", reason="Inverse-cloze retrieval-key transcripts; retrieval-key training is not used.")
        else:
            reason = next((r for prefix, r in SDKB_EXCLUDED_PREFIXES.items() if name.startswith(prefix)), None)
            if reason:
                entry.update(disposition="excluded", reason=reason)
            else:
                entry.update(disposition="queued", converter="review", reason="Unclassified corpus: needs review.")
        entries.append(entry)
    return entries


def sdkb_raw_entries(root: Path = SDKB_RAW) -> list[dict]:
    entries = []
    for group in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(q for q in group.iterdir() if q.is_dir() and not q.name.startswith("_")):
            entries.append({"id": f"schnitzeljagd:raw:{group.name}/{p.name}", "project": "schnitzeljagd",
                            "location": str(p), "kind": "raw-source", "bytes": None, "rows": None, "family": None,
                            "disposition": "source", "converter": "Natlang task adapters and port converters",
                            "reason": "Raw upstream download behind Schnitzeljagd corpora; adapters read from here (see group README for licences).",
                            "license": "see README"})
        if not any(group.iterdir()):
            continue
    return entries


MANUAL = [
    # bgkit, outside the task stores
    ("bgkit:benchmarks", "bgkit", "/home/werg/bgkit-data-nvme/bgkit2/benchmarks", "benchmarks", "eval-only", None,
     "Protected held-out evaluation (benchpress, browsecomp_plus, hotpotqa_distractor, nq_open, squad, triviaqa_rc, swebench_lite, web pool, showcase). Scanned out of training by question hash and id.", "LicenseRef-mixed"),
    ("bgkit:corpus", "bgkit", "/home/werg/bgkit-data-nvme/bgkit2/corpus", "tokenized-corpus", "excluded", None,
     "LFM2.5-350M token ids (fineweb, code, books, dialogue); rebuilt from raw text as reconstruct records and replay.", "LicenseRef-mixed"),
    ("bgkit:filler", "bgkit", "/home/werg/bgkit-data-nvme/bgkit2/filler", "raw-text", "queued", "reconstruct+replay",
     "fineweb_edu, pg19, wikitext raw text for reconstruct records and replay.", "LicenseRef-mixed"),
    ("bgkit:docs", "bgkit", "/home/werg/bgkit-data-nvme/bgkit2/docs", "raw-text", "queued", "reconstruct+replay",
     "Library documentation dumps (pages.parquet).", "LicenseRef-mixed"),
    ("bgkit:raw", "bgkit", "/home/werg/bgkit-data-nvme/bgkit2/raw", "raw-text", "queued", "replay",
     "babi and openthoughts raw.", "LicenseRef-mixed"),
    ("bgkit:teacher", "bgkit", "/home/werg/bgkit-data-nvme/bgkit2/teacher", "teacher-trajectories", "queued", "bgkit.teacher",
     "run1-3, fix1-4, lfm8b_swe_ov, synth_* (rows.parquet, scores.jsonl, trajectories.jsonl): port records with outcome labels from scores; failures kept as labelled feedback.", "LicenseRef-mixed"),
    ("bgkit:trajectories_ext/swe_rebench_openhands", "upstream", "/mnt/external/bgkit-data/trajectories_ext/swe_rebench_openhands", "raw-trajectories", "migrated", "swe_compaction",
     "SWE-rebench OpenHands trajectories (67,074); continuation records rebuilt at message boundaries.", "CC-BY-4.0"),
    ("bgkit:trajectories_ext/open_swe_traces", "upstream", "/mnt/external/bgkit-data/trajectories_ext/open_swe_traces", "raw-trajectories", "migrated", "agents.open_swe_traces",
     "open_swe_traces_ts_think: coding continuation records; resolved label checked; repository licence in notes.", "CC-BY-4.0"),
    ("bgkit:trajectories_ext/nemotron_sft_swe_v35", "upstream", "/mnt/external/bgkit-data/trajectories_ext/nemotron_sft_swe_v35", "raw-trajectories", "migrated", "agents.nemotron_swe",
     "nemotron_sft_swe_v35_filtered (5,115 rows): coding continuation records.", "CC-BY-4.0"),
    ("bgkit:trajectories_ext/agenttrove", "upstream", "/mnt/external/bgkit-data/trajectories_ext/agenttrove", "raw-trajectories", "migrated", "agents.agenttrove",
     "Terminal-agent continuation records from the last episode of each trial.", "Apache-2.0"),
    ("bgkit:trajectories_ext/openresearcher", "upstream", "/mnt/external/bgkit-data/trajectories_ext/openresearcher", "raw-trajectories", "excluded", None,
     "Seeds 42/43 copy, superseded by web_sft OpenResearcher (all 16 seeds, agents.openresearcher).", "MIT"),
    ("bgkit:trajectories_ext/bench_dedupe", "upstream", "/mnt/external/bgkit-data/trajectories_ext/bench_dedupe", "benchmarks", "eval-only", None,
     "SWE benchmark instance/PR/commit keys; protected from training.", "LicenseRef-mixed"),
    ("bgkit:web_sft", "upstream", "/mnt/external/bgkit-data/web_sft", "raw-trajectories", "migrated", "agents (openresearcher, openseeker_miroverse, webshaper, search_r1)",
     "OpenResearcher (MIT), OpenSeeker/MiroVerse (other / CC-BY-NC-4.0), WebShaper and Search-R1 (unverified) converted; ASearcher (RL prompts, no trajectories) and SimpleDeepSearcher (custom format) still queued.", "LicenseRef-mixed"),
    ("bgkit:repos", "bgkit", "/mnt/external/bgkit-data/repos", "git-repositories", "source", "natlang.repository + repo records",
     "12,083 bare git repositories; tokenizer-independent source for repository tasks and git-history context.", "LicenseRef-repository-content"),
    ("bgkit:browse_trees", "bgkit", "/mnt/external/bgkit-data/browse_trees", "raw-structured", "queued", "bgkit.git_history",
     "git_commit_repro, git_history, kilt, lognav, babilong, xref; KB-navigation framing dropped.", "LicenseRef-mixed"),
    ("bgkit:qa_pairs", "bgkit", "/mnt/external/bgkit-data/qa_pairs", "generated-qa", "queued", "bgkit.repo_metadata",
     "Per-repo LLM-generated QA; teacher recorded as unknown where provenance is missing.", "LicenseRef-repository-content"),
    ("bgkit:qa_pairs_filtered", "bgkit", "/mnt/external/bgkit-data/qa_pairs_filtered", "generated-qa", "queued", "bgkit.repo_metadata",
     "Filtered variant of qa_pairs.", "LicenseRef-repository-content"),
    ("bgkit:descriptions", "bgkit", "/mnt/external/bgkit-data/descriptions", "generated-descriptions", "queued", "bgkit.repo_metadata",
     "Per-repo LLM-generated descriptions; teacher unknown.", "LicenseRef-repository-content"),
    ("bgkit:descriptions.old", "bgkit", "/mnt/external/bgkit-data/descriptions.old", "generated-descriptions", "excluded", None,
     "Superseded by descriptions.", "LicenseRef-repository-content"),
    ("bgkit:structural", "bgkit", "/mnt/external/bgkit-data/structural", "generated-structure", "queued", "bgkit.repo_metadata",
     "Per-repo structural metadata.", "LicenseRef-repository-content"),
    ("bgkit:arxiv_v1", "bgkit", "/mnt/external/bgkit-data/arxiv_v1", "raw-text", "queued", "reconstruct+summarise",
     "Raw text source; low priority.", "LicenseRef-arXiv-mixed"),
    ("bgkit:pubmed_v1", "bgkit", "/mnt/external/bgkit-data/pubmed_v1", "raw-text", "queued", "reconstruct+summarise", "Raw text source; low priority.", "LicenseRef-mixed"),
    ("bgkit:multi_news_v1", "bgkit", "/mnt/external/bgkit-data/multi_news_v1", "raw-text", "queued", "reconstruct+summarise", "Raw text source; low priority.", "LicenseRef-mixed"),
    ("bgkit:chatqa2_long_sft", "bgkit", "/mnt/external/bgkit-data/chatqa2_long_sft", "raw-qa", "source", "bgkit.slot_qa", "Upstream of the chatqa2 stores, which are migrated; licence not verified locally.", "LicenseRef-unverified"),
    ("bgkit:browsecomp_plus", "bgkit", "/mnt/external/bgkit-data/browsecomp_plus", "benchmarks", "eval-only", None, "BrowseComp-Plus corpus and queries.", "LicenseRef-mixed"),
    ("bgkit:swe_rebench", "bgkit", "/mnt/external/bgkit-data/swe_rebench", "raw-tasks", "queued", "natlang.swe", "SWE-rebench task instances for container environments.", "CC-BY-4.0"),
    ("bgkit:swe_repos", "bgkit", "/mnt/external/bgkit-data/swe_repos", "git-repositories", "source", "natlang.swe", "12 SWE-bench repositories.", "LicenseRef-repository-content"),
    ("bgkit:bgkit2_walk", "bgkit", "/mnt/external/bgkit-data/bgkit2_walk", "raw-structured", "queued", "bgkit.repo_tree", "Repository walks.", "LicenseRef-repository-content"),
    ("bgkit:bgkit2_intermediate", "bgkit", "/mnt/external/bgkit-data/bgkit2_intermediate", "intermediate", "excluded", None, "Intermediate coding builds; upstream traces are converted instead.", "LicenseRef-mixed"),
    ("bgkit:processed", "bgkit", "/mnt/external/bgkit-data/processed", "tokenized-corpus", "excluded", None,
     "bgkit1 Qwen3.5/Falcon-H1 token ids with no text; same content available from raw sources. Re-decoding possible but not pursued.", "LicenseRef-mixed"),
    ("bgkit:processed_v2", "bgkit", "/mnt/external/bgkit-data/processed_v2", "tokenized-corpus", "excluded", None, "bgkit1 token ids; see processed.", "LicenseRef-mixed"),
    ("bgkit:fineweb_edu_v1", "bgkit", "/mnt/external/bgkit-data/fineweb_edu_v1", "tokenized-corpus", "excluded", None, "bgkit1 token ids; raw fineweb_edu text is in filler/.", "ODC-BY-1.0"),
    ("bgkit:mmap", "bgkit", "/mnt/external/bgkit-data/mmap", "tokenized-corpus", "excluded", None, "bgkit1 memory-mapped token stores (browse trees, qa_conditioned, structural).", "LicenseRef-mixed"),
    ("bgkit:synthetic", "bgkit", "/mnt/external/bgkit-data/synthetic", "intermediate", "queued", "review", "bgkit1 synthetic data; needs review.", "LicenseRef-unknown"),
    ("bgkit:trajectories", "bgkit", "/mnt/external/bgkit-data/trajectories", "raw-trajectories", "queued", "review", "bgkit1 trajectory collection; needs review against trajectories_ext.", "LicenseRef-unknown"),
    ("bgkit:taxonomies", "bgkit", "/mnt/external/bgkit-data/taxonomies", "metadata", "excluded", None, "Classification metadata, not training data.", "LicenseRef-none"),
    ("bgkit:provenance", "bgkit", "/mnt/external/bgkit-data/provenance", "metadata", "source", "lineage", "Provenance records consulted for lineage.", "LicenseRef-none"),
    ("bgkit:staging", "bgkit", "/mnt/external/bgkit-data/staging", "intermediate", "excluded", None, "Staging area of earlier builds.", "LicenseRef-mixed"),
    ("bgkit:diagnostics", "bgkit", "/mnt/external/bgkit-data/diagnostics", "logs", "excluded", None, "Diagnostics, not data.", "LicenseRef-none"),
    ("bgkit:logs", "bgkit", "/mnt/external/bgkit-data/logs", "logs", "excluded", None, "Logs, not data.", "LicenseRef-none"),
    ("bgkit:crawl_state", "bgkit", "/mnt/external/bgkit-data/crawl_state.db", "metadata", "excluded", None, "Crawler state database.", "LicenseRef-none"),
    ("bgkit:models", "bgkit", "/mnt/external/bgkit-data/models", "models", "excluded", None, "Model weights (GGUF, LFM2.5-8B-A1B, Qwen3.6-35B-NVFP4), not data.", "LicenseRef-model"),
    ("bgkit:checkpoints", "bgkit", "/mnt/external/bgkit-checkpoints", "checkpoints", "excluded", None,
     "bgkit checkpoints, tree caches and survivor caches: model outputs tied to bgkit encoders.", "LicenseRef-model"),
    ("bgkit:ckpt-fast", "bgkit", "/home/werg/bgkit-ckpt-fast", "checkpoints", "excluded", None, "bgkit 8B-A checkpoints and diagnostics.", "LicenseRef-model"),
    ("bgkit:capability_packaging", "bgkit", "/home/werg/bgkit-data-nvme/capability_packaging", "mixed", "queued", "review",
     "loghub, lognav_qa, babilong/benchpress/ruler, swe_zero and toucan trajectories, Phase-1 base checkpoint (excluded).", "LicenseRef-mixed"),
    ("bgkit:nvme_archive", "bgkit", "/mnt/external/nvme_archive_2026_09_26", "archive", "source", None,
     "HDD archive of bgkit-data-nvme (arxiv_v1, bgkit2, browse_trees, fineweb_edu_v1, mmap, multi_news_v1, processed); entries above cover its contents.", "LicenseRef-mixed"),
    ("bgkit:bgkit2-clone", "bgkit", "/home/werg/bgkit2", "code", "excluded", None, "Stale clone of the bgkit repository; no data.", "LicenseRef-none"),
    # Schnitzeljagd, outside corpora and raw
    ("schnitzeljagd:banks", "schnitzeljagd", "/mnt/external/sdkb-archive/banks", "latent-banks", "excluded", None, "Latent knowledge-base banks: Schnitzeljagd model outputs.", "LicenseRef-model"),
    ("schnitzeljagd:runs", "schnitzeljagd", "/mnt/external/sdkb-archive/runs", "runs", "excluded", None, "Training run outputs.", "LicenseRef-model"),
    ("schnitzeljagd:sdkb-runs", "schnitzeljagd", "/mnt/external/natlang-storage-20261002/sdkb-runs", "runs", "excluded", None, "Training run outputs (167 GB).", "LicenseRef-model"),
    ("schnitzeljagd:hf-models", "schnitzeljagd", "/mnt/external/sdkb-archive/hf-models", "models", "excluded", None, "Model weights.", "LicenseRef-model"),
    ("schnitzeljagd:retained-checkpoints", "schnitzeljagd", "/mnt/external/sdkb-archive/retained-checkpoints", "checkpoints", "excluded", None, "Checkpoints.", "LicenseRef-model"),
    ("schnitzeljagd:validation", "schnitzeljagd", "/mnt/external/sdkb-archive/validation", "eval-outputs", "excluded", None, "Validation outputs of Schnitzeljagd runs.", "LicenseRef-model"),
    ("schnitzeljagd:probes", "schnitzeljagd", "/mnt/external/sdkb-archive/probes", "eval-outputs", "excluded", None, "Probe outputs.", "LicenseRef-model"),
    ("schnitzeljagd:cache", "schnitzeljagd", "/mnt/external/sdkb-archive/cache", "cache", "excluded", None, "Caches.", "LicenseRef-none"),
    ("schnitzeljagd:diagnostics", "schnitzeljagd", "/mnt/external/sdkb-archive/diagnostics", "logs", "excluded", None, "Diagnostics.", "LicenseRef-none"),
    ("schnitzeljagd:profiles", "schnitzeljagd", "/mnt/external/sdkb-archive/profiles", "logs", "excluded", None, "Profiles.", "LicenseRef-none"),
    ("schnitzeljagd:run-hash-dirs", "schnitzeljagd", "/mnt/external/sdkb-archive/<32-hex>", "runs", "excluded", None, "Hash-named run directories.", "LicenseRef-model"),
    ("schnitzeljagd:teacher-datasets", "schnitzeljagd", "/home/werg/sdkb/docs/datasets.md", "catalogue", "queued", "schnitzel.teacher_prefixes",
     "hermes, ultrachat, swe_smith, openhands prefix-memory and cross-experience protocols. Not materialised locally "
     "(the hub cache holds only the SWE-smith README); openhands is covered by swe_compaction. Hermes (Apache-2.0), "
     "UltraChat and SWE-smith (MIT) need a download before conversion; cross-experience episodes handed to S2.", "LicenseRef-mixed"),
    # natlang
    ("natlang:v13-corpus", "natlang", "runs/lfm25-350m-broad-20261002", "model-neutral-corpus", "queued", "api-migration neuralese-language (ts-host/scripts/migrate-neuralese-language.mjs)",
     "111,301 rows (105,869 train / 5,432 test); to be migrated through eager typing and explicit captures. Builder skeleton "
     "and inventory exist; rewriting waits on the S4 compiler passes.", "LicenseRef-mixed"),
    ("natlang:teacher-ir", "natlang", "data/teacher", "program-ir", "source", "collector",
     "natlang.program/2 IR sources; unchanged as IR, re-collected when runtime or prompts change.", "LicenseRef-mixed"),
    ("natlang:retired-python-sources", "natlang", "data/external", "raw-tasks", "queued", "natlang.retired_sources",
     "SCONE, Schema-Guided Dialogue, FinQA, CLEVR, sales, support: checkable results.", "LicenseRef-mixed"),
    ("natlang:external_pilot", "natlang", "data/external_pilot", "pre-ir-traces", "excluded", None, "Pre-IR traces: not canonical (PROGRAM_IR_PIPELINE.md).", "LicenseRef-mixed"),
    ("natlang:development-data", "natlang", "/mnt/external/natlang-development-data", "development-data", "source", "natlang ledgers",
     "Code corpora, ref-v7/v8, weakness sets; inventoried by the existing natlang ledgers, no change of disposition.", "LicenseRef-mixed"),
    ("natlang:native-directory", "natlang", "/mnt/external/natlang-native-directory-20261002", "model+preflight", "excluded", None,
     "MiniCPM5-2B weights and editor-preflight data; covered by natlang's own ledgers.", "LicenseRef-model"),
    # external drive, other
    ("external:natlang-vllm-qwen36", "external", "/mnt/external/natlang-vllm-qwen36", "cache", "excluded", None, "vLLM compile caches.", "LicenseRef-none"),
    ("external:outputs-proto1", "external", "/mnt/external/outputs", "checkpoints", "excluded", None, "Old proto1 checkpoints (Dec 2025).", "LicenseRef-model"),
    ("external:hf-cache", "external", "/mnt/external/hf-cache", "models", "excluded", None, "Hugging Face model cache (Ling-3.0-tiny, Gemma-4, Qwen3.5, Falcon-H1-Tiny).", "LicenseRef-model"),
    ("external:ComfyUI", "external", "/mnt/external/ComfyUI", "unrelated", "excluded", None, "Image-generation tooling; not training data.", "LicenseRef-none"),
    ("external:SparkyUI", "external", "/mnt/external/SparkyUI", "unrelated", "excluded", None, "UI project; not training data.", "LicenseRef-none"),
]


def manual_entries() -> list[dict]:
    out = []
    for ident, project, location, kind, disposition, converter, reason, spdx in MANUAL:
        out.append({"id": ident, "project": project, "location": location, "kind": kind, "bytes": None, "rows": None,
                    "family": None, "disposition": disposition, "converter": converter, "reason": reason, "license": spdx})
    return out


def build() -> dict:
    entries = bgkit_task_entries() + sdkb_corpus_entries() + sdkb_raw_entries() + manual_entries()
    counts: dict = {}
    for e in entries:
        counts[e["disposition"]] = counts.get(e["disposition"], 0) + 1
    missing_reason = [e["id"] for e in entries if e["disposition"] != "migrated" and not e.get("reason")]
    return {"version": LEDGER_VERSION,
            "policy": "Every dataset has a disposition; exclusion requires a recorded reason (training/data_sources.json). "
                      "Generated by scripts/neuralese_port_records.py ledger; manual entries live in scripts/neuralese_data/inventory.py.",
            "counts": counts, "missing_reason": missing_reason, "entries": entries}
