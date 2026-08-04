#!/usr/bin/env python3
"""Build an anonymous, deterministically randomized bank of real PhysX videos."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    INVALID_PROTOCOL,
    PASS,
    SKIPPED_DEPENDENCY,
    DependencyUnavailable,
    ProtocolError,
    canonical_sha256,
    diagnostic_result,
    load_spec,
    read_json,
    sha256_file,
    write_json_exclusive,
)
from diagnostics.common.policy_class_probe import output_dir_from_args
from diagnostics.common.reward_stage4 import (
    BLIND_VIDEO_SOURCE_SCHEMA,
    RewardValidityProtocol,
    deterministic_blind_order,
)


def _copy_exclusive(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.set_defaults(execute_real=True)
    parser.add_argument("--execute-real", dest="execute_real", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-execute-real", dest="execute_real", action="store_false")
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(root, spec, args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = video_dir / "blind_pair_manifest.json"
    answer_path = video_dir / "blind_pair_answer_key.json"
    status_path = video_dir / "blind_pair_manifest.status.json"
    try:
        protocol = RewardValidityProtocol.from_spec(spec)
        branching_path = output_dir / "branching.json"
        if not branching_path.is_file():
            raise DependencyUnavailable("diag_42 branching status is absent")
        branching = read_json(branching_path)
        if branching.get("status") != PASS:
            raise DependencyUnavailable("diag_42 has no real PhysX branches to render")
        branch_path = Path(str(branching["evidence"]["branch_bank"])).expanduser().resolve()
        branch_hash = str(branching["evidence"].get("branch_bank_sha256", ""))
        if sha256_file(branch_path) != branch_hash:
            raise ProtocolError("branch bank changed after diag_42")
        source_path = (
            args.source_manifest or video_dir / "video_pair_sources.json"
        ).expanduser().resolve()
        if args.execute_real and not source_path.is_file():
            from diagnostics.common.stage4_sim import render_blind_video_sources_real

            completed = render_blind_video_sources_real(
                repo_root=root,
                output_dir=output_dir,
                spec=spec,
                branch_bank_path=branch_path,
                source_manifest_path=source_path,
            )
            if completed is not True:
                raise ProtocolError("real video renderer did not confirm completion")
        if not source_path.is_file():
            raise DependencyUnavailable(
                "real PhysX pair videos/source manifest are absent; placeholder videos are forbidden"
            )
        source = read_json(source_path)
        if not isinstance(source, dict) or source.get("schema") != BLIND_VIDEO_SOURCE_SCHEMA:
            raise ProtocolError("blind-video source schema is invalid")
        if source.get("source_branch_bank_sha256") != branch_hash:
            raise ProtocolError("video sources are not rendered from the validated branch bank")
        for key in ("real_physx_videos", "common_camera", "common_timing", "identity_overlays_absent"):
            if source.get(key) is not True:
                raise ProtocolError(f"video source lacks guarantee {key}")
        if source.get("A_mix") not in (None, "legacy_quarantined"):
            raise ProtocolError("A_mix/FCAMP entered blind-video sources")
        raw_pairs = source.get("pairs")
        expected_count = int(protocol.blind_video["pair_count"])
        if not isinstance(raw_pairs, list) or len(raw_pairs) != expected_count:
            raise DependencyUnavailable(
                f"blind-video bank requires exactly {expected_count} real pairs"
            )
        ordering_seed = int(protocol.blind_video["ordering_seed"])
        public_pairs: list[dict[str, object]] = []
        private_pairs: list[dict[str, object]] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_pairs):
            if not isinstance(raw, dict):
                raise ProtocolError("video source pair is not a mapping")
            pair_id = str(raw.get("pair_id", ""))
            if not pair_id or pair_id in seen:
                raise ProtocolError("video pair identity is empty or duplicated")
            seen.add(pair_id)
            if raw.get("strict_pareto_verified") is not True:
                raise ProtocolError("blind pair has no strict independent outcome preference")
            winner = Path(str(raw.get("winner_video", ""))).expanduser().resolve()
            loser = Path(str(raw.get("loser_video", ""))).expanduser().resolve()
            if not winner.is_file() or not loser.is_file():
                raise DependencyUnavailable(f"real source video is missing for {pair_id}")
            if sha256_file(winner) != str(raw.get("winner_video_sha256", "")):
                raise ProtocolError("winner source-video hash changed")
            if sha256_file(loser) != str(raw.get("loser_video_sha256", "")):
                raise ProtocolError("loser source-video hash changed")
            winner_left = deterministic_blind_order(pair_id, ordering_seed)
            left_source, right_source = (winner, loser) if winner_left else (loser, winner)
            opaque = f"pair_{index:03d}"
            left_target = video_dir / f"{opaque}_left.mp4"
            right_target = video_dir / f"{opaque}_right.mp4"
            _copy_exclusive(left_source, left_target)
            _copy_exclusive(right_source, right_target)
            public_pairs.append(
                {
                    "anonymous_pair_id": opaque,
                    "left_video": str(left_target),
                    "right_video": str(right_target),
                    "left_sha256": sha256_file(left_target),
                    "right_sha256": sha256_file(right_target),
                }
            )
            private_pairs.append(
                {
                    "anonymous_pair_id": opaque,
                    "source_pair_id": pair_id,
                    "preferred_side_by_strict_outcomes": "left" if winner_left else "right",
                    "winner_branch_id": str(raw.get("winner_branch_id", "")),
                    "loser_branch_id": str(raw.get("loser_branch_id", "")),
                    "snapshot_id": str(raw.get("snapshot_id", "")),
                }
            )
        public = {
            "schema": "largebox_blind_pair_manifest_v1",
            "anonymous_labels": True,
            "ordering_seed_sha256": canonical_sha256({"ordering_seed": ordering_seed}),
            "pair_count": len(public_pairs),
            "pairs": public_pairs,
            "instructions": "Choose left, right, or tie based only on visible motion quality.",
        }
        private = {
            "schema": "largebox_blind_pair_answer_key_v1",
            "ordering_seed": ordering_seed,
            "source_manifest": str(source_path),
            "source_manifest_sha256": sha256_file(source_path),
            "pairs": private_pairs,
        }
        write_json_exclusive(manifest_path, public)
        write_json_exclusive(answer_path, private)
        result = diagnostic_result(
            "46",
            PASS,
            summary="real PhysX comparison videos were anonymously randomized",
            evidence={
                "public_manifest": str(manifest_path),
                "public_manifest_sha256": sha256_file(manifest_path),
                "private_answer_key": str(answer_path),
                "private_answer_key_sha256": sha256_file(answer_path),
                "pair_count": len(public_pairs),
                "external_preferences": "optional; no preferences were fabricated",
                "A_mix": "legacy_quarantined",
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "46",
            SKIPPED_DEPENDENCY,
            summary="blind-video bank awaits real PhysX renders",
            evidence={"executor_seam": "diagnostics.common.stage4_sim.render_blind_video_sources_real"},
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, RuntimeError) as exc:
        result = diagnostic_result(
            "46", INVALID_PROTOCOL, summary="blind-video protocol failed closed", errors=[str(exc)]
        )
    write_json_exclusive(status_path, result)
    print(f"[diag_46] {result['status']} {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
