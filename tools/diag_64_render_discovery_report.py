#!/usr/bin/env python3
"""Render every diagnostic result into one TXT and package source/log evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tarfile
import tempfile
import sys
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    INVALID_PROTOCOL,
    PASS,
    ProtocolError,
    diagnostic_result,
    load_spec,
    read_json,
    sha256_file,
    write_json_exclusive,
)
from diagnostics.common.reporting import utc_now_iso


TEXT_RESULT_SUFFIXES = {".json", ".csv", ".yaml", ".yml"}
RAW_DIRECTORY_NAMES = {"models", "rollouts", "legacy", "checkpoints"}


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _diagnostics(spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for stage in spec.get("stages", []):
        for item in stage.get("diagnostics", []):
            values.append({**dict(item), "stage": str(stage.get("id"))})
    if not values:
        raise ProtocolError("suite spec contains no diagnostics")
    return values


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"<UNREADABLE: {exc}>\n"


def _result_files(output_dir: Path, spec: Mapping[str, Any]) -> list[Path]:
    selected: set[Path] = set()
    for item in _diagnostics(spec):
        for relative in [item.get("status_artifact"), *item.get("expected_outputs", [])]:
            if not relative:
                continue
            path = (output_dir / str(relative)).resolve()
            if path.is_file() and path.suffix.lower() in TEXT_RESULT_SUFFIXES:
                selected.add(path)
    for directory in (output_dir / "tables", output_dir / "edge_runs"):
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix.lower() in TEXT_RESULT_SUFFIXES:
                    selected.add(path.resolve())
    for relative in (
        "status.json",
        "manifest.json",
        "postfreeze_diagnostic_corrections.json",
    ):
        path = (output_dir / relative).resolve()
        if path.is_file():
            selected.add(path)
    return sorted(selected, key=lambda path: path.relative_to(output_dir).as_posix())


def _binary_evidence(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    excluded = {"diagnostic_results.txt", "discovery_report.md", "source_and_logs.tar.gz"}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(output_dir)
        if path.suffix.lower() in TEXT_RESULT_SUFFIXES or "logs" in relative.parts:
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _status_rows(output_dir: Path, spec: Mapping[str, Any]) -> list[dict[str, str]]:
    runner = read_json(output_dir / "status.json")
    observed = {
        str(item.get("diagnostic_id")): item
        for item in runner.get("diagnostics", [])
        if isinstance(item, Mapping)
    }
    rows: list[dict[str, str]] = []
    for item in _diagnostics(spec):
        identifier = str(item["id"])
        execution = observed.get(identifier, {})
        artifact_path = output_dir / str(item["status_artifact"])
        artifact: Mapping[str, Any] = {}
        if artifact_path.is_file() and artifact_path.suffix == ".json":
            raw = read_json(artifact_path)
            if isinstance(raw, Mapping):
                artifact = raw
        rows.append(
            {
                "id": identifier,
                "stage": str(item["stage"]),
                "name": str(item["name"]),
                "status": str(execution.get("status", artifact.get("status", "NOT_RECORDED"))),
                "summary": str(artifact.get("summary", execution.get("reason", ""))),
                "status_artifact": str(item["status_artifact"]),
            }
        )
    return rows


def _completed_suite_status(
    suite_status: Mapping[str, Any], status_rows: Iterable[Mapping[str, str]]
) -> dict[str, Any]:
    """Return the final 52-node status, including the report node itself.

    ``status.json`` is necessarily one runner update behind while diag_64 is
    executing.  Rendering that intermediate count would make the TXT claim 51
    diagnostics while listing 52.  Recompute only the aggregate/count fields;
    the runner remains the authority for each preceding execution record.
    """

    statuses = [str(row["status"]) for row in status_rows]
    allowed = {"PASS", "FAIL", "INVALID_PROTOCOL", "SKIPPED_DEPENDENCY"}
    if not statuses or any(value not in allowed for value in statuses):
        raise ProtocolError("final report contains an unrecorded diagnostic status")
    counts = {value: statuses.count(value) for value in sorted(allowed)}
    overall = "PASS"
    for candidate in ("INVALID_PROTOCOL", "FAIL", "SKIPPED_DEPENDENCY"):
        if counts[candidate]:
            overall = candidate
            break
    completed = dict(suite_status)
    completed["overall_status"] = overall
    completed["status_counts"] = counts
    return completed


def _render_txt(
    *,
    output_dir: Path,
    spec: Mapping[str, Any],
    suite_status: Mapping[str, Any],
    decision: Mapping[str, Any],
    status_rows: list[Mapping[str, str]],
    result_files: list[Path],
    binary_records: list[Mapping[str, Any]],
) -> str:
    lines = [
        "AMP RESEARCH DISCOVERY SUITE — 完整诊断结果",
        "=" * 78,
        f"生成时间: {utc_now_iso()}",
        f"suite_id: {spec.get('suite_id')}",
        f"suite_version: {spec.get('suite_version')}",
        f"任务: {spec.get('frozen_research_boundaries', {}).get('task_isolation', {}).get('task_name')}",
        f"动作: {spec.get('frozen_research_boundaries', {}).get('task_isolation', {}).get('motion_name')}",
        "范围: 只诊断 AMP 及其直接改进；不实现新算法。",
        "旧代码隔离: A_mix/FCAMP/H4 已按仓库所有者指令淘汰，未加载、未采集、未参与任何 gate。",
        "正式五域: K, T_u200, T_u500, A_amp, B。",
        "术语限制: overlap 均为 empirical effective overlap，不是数学 support 或可达性证明。",
        "",
        "一、总状态与冻结决策",
        "-" * 78,
        f"overall_status: {suite_status.get('overall_status')}",
        f"status_counts: {_json_text(suite_status.get('status_counts', {}))}",
        f"R/T/E/L/F/X: {_json_text(decision.get('evidence', {}).get('decision_variables', {}))}",
        f"decision: {_json_text(decision.get('evidence', {}).get('decision', {}))}",
        "",
        "二、协议修订（全部发生在正式采集之前）",
        "-" * 78,
        _json_text(spec.get("protocol_amendments", [])),
        "",
        "三、逐项诊断状态",
        "-" * 78,
    ]
    for row in status_rows:
        lines.append(
            f"[{row['id']}] {row['status']} | {row['name']} | {row['summary']} "
            f"| {row['status_artifact']}"
        )
    lines.extend(
        [
            "",
            "四、全部文本结果原文",
            "-" * 78,
            "以下按路径逐个嵌入所有诊断 JSON/CSV/YAML；原始 rollout/model/tensor 不复制，见第五节哈希索引。",
        ]
    )
    for path in result_files:
        relative = path.relative_to(output_dir).as_posix()
        lines.extend(
            [
                "",
                f">>> BEGIN {relative}",
                _safe_read(path).rstrip("\n"),
                f"<<< END {relative}",
            ]
        )
    lines.extend(
        [
            "",
            "五、二进制/图像/视频/模型/原始数据证据索引",
            "-" * 78,
            _json_text(binary_records),
            "",
            "六、结论边界",
            "-" * 78,
            "本文件只汇总冻结诊断所得事实与预注册 Go/No-Go。它没有提出、训练或追认任何完整新算法。",
            "SKIPPED_DEPENDENCY 表示缺少真实证据，INVALID_PROTOCOL 表示协议不可解释；二者绝不当作科学 FAIL 或 PASS。",
        ]
    )
    return "\n".join(lines) + "\n"


def _archive_members(repo_root: Path, output_dir: Path) -> Iterable[tuple[Path, str]]:
    allowed_output_roots = {"logs", "tables", "plots", "videos", "edge_runs"}
    fixed_output_files = {
        "manifest.json", "status.json", "family_decision_matrix.json",
        "policy_class_gate.json", "domain_triangle_gate.json",
        "reward_validity_gate.json", "edge_gate.json", "diagnostic_results.txt",
        "discovery_report.md", "source_snapshot.tar.gz",
        "postfreeze_analysis_source_snapshot.tar.gz",
        "postfreeze_diagnostic_corrections.json",
    }
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "source_and_logs.tar.gz":
            continue
        relative = path.relative_to(output_dir)
        root_name = relative.parts[0]
        include_root_file = root_name in allowed_output_roots and (
            root_name in {"logs", "tables", "plots"}
            or path.suffix.lower() in TEXT_RESULT_SUFFIXES | {".log", ".txt"}
        )
        if relative.name in fixed_output_files or include_root_file:
            yield path, (Path("suite_output") / relative).as_posix()
    for relative in (
        "diagnostics/specs/largebox_discovery_v1.yaml",
        "configs/fixed_reward_largebox.yaml",
    ):
        path = repo_root / relative
        if path.is_file():
            yield path, (Path("repository") / relative).as_posix()


def _write_archive(path: Path, members: Iterable[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for source, arcname in members:
                archive.add(source, arcname=arcname, recursive=False)
        # Exclusive publication: os.link fails if another result exists and
        # does not expose a partially-written archive.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_target = output_dir / "discovery_report.status.json"
    try:
        spec = load_spec(args.spec)
        suite_status = read_json(output_dir / "status.json")
        decision = read_json(output_dir / "family_decision_matrix.json")
        if decision.get("diagnostic_id") != "63" or decision.get("status") != PASS:
            raise ProtocolError("diag_63 is not a valid completed decision artifact")
        status_rows = _status_rows(output_dir, spec)
        result_files = _result_files(output_dir, spec)
        binary_records = _binary_evidence(output_dir)
        archive_path = output_dir / "source_and_logs.tar.gz"
        _write_archive(archive_path, _archive_members(repo_root, output_dir))
        binary_records.append(
            {
                "path": "source_and_logs.tar.gz",
                "bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
            }
        )
        # At this point every potentially failing prerequisite of diag_64 has
        # succeeded.  Record the report node itself in the one-file summary;
        # the runner will independently validate the declared files afterward.
        for row in status_rows:
            if row["id"] == "64":
                row["status"] = PASS
                row["summary"] = "all diagnostic results consolidated into this TXT"
        suite_status = _completed_suite_status(suite_status, status_rows)
        text = _render_txt(
            output_dir=output_dir,
            spec=spec,
            suite_status=suite_status,
            decision=decision,
            status_rows=status_rows,
            result_files=result_files,
            binary_records=binary_records,
        )
        txt_path = output_dir / "diagnostic_results.txt"
        markdown_path = output_dir / "discovery_report.md"
        _write_text_exclusive(txt_path, text)
        _write_text_exclusive(
            markdown_path,
            "# AMP Research Discovery Suite\n\n"
            "完整、无省略的诊断汇总见 [`diagnostic_results.txt`](diagnostic_results.txt)。\n\n"
            f"- suite status: `{suite_status.get('overall_status')}`\n"
            f"- frozen decision: `{decision.get('evidence', {}).get('decision', {}).get('selected_family')}`\n"
            "- deprecated A_mix/FCAMP/H4: `quarantined; not loaded`\n",
        )
        result = diagnostic_result(
            "64",
            PASS,
            summary="all diagnostic results were consolidated into one TXT",
            evidence={
                "diagnostic_results": str(txt_path),
                "diagnostic_results_sha256": sha256_file(txt_path),
                "discovery_report": str(markdown_path),
                "discovery_report_sha256": sha256_file(markdown_path),
                "source_and_logs": str(archive_path),
                "source_and_logs_sha256": sha256_file(archive_path),
                "diagnostic_count": len(status_rows),
                "embedded_text_result_count": len(result_files),
                "indexed_binary_evidence_count": len(binary_records),
            },
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, TypeError, ValueError, ProtocolError) as exc:
        result = diagnostic_result(
            "64", INVALID_PROTOCOL, summary="discovery report rendering failed closed", errors=[str(exc)]
        )
    write_json_exclusive(status_target, result)
    print(f"[diag_64] {result['status']} {status_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
