"""
Raga-wise motif mining and lookup utilities.

This module provides a standalone CLI:

    python -m raga_pipeline.motifs mine ...
    python -m raga_pipeline.motifs score ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac")

GENDER_ALIASES = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
}

INSTRUMENT_ALIASES = {
    "sitar": "sitar",
    "sarod": "sarod",
    "flute": "bansuri",
    "bansuri": "bansuri",
    "mohan veena": "slide_guitar",
    "slide guitar": "slide_guitar",
    "slide_guitar": "slide_guitar",
}


@dataclass(frozen=True)
class GroundTruthRow:
    row_num: int
    filename: str
    raga: str
    tonic: str
    instrument: str
    gender: str


@dataclass
class TranscriptionCandidate:
    recording_dir: Path
    stem: str
    original_csv: Optional[Path] = None
    edited_csv: Optional[Path] = None

    def resolve(self, source_mode: str) -> Tuple[Optional[Path], str]:
        mode = (source_mode or "auto").strip().lower()
        if mode == "edited":
            return self.edited_csv, "edited"
        if mode == "original":
            return self.original_csv, "original"
        if self.edited_csv is not None:
            return self.edited_csv, "edited"
        return self.original_csv, "original"


@dataclass(frozen=True)
class TokenNote:
    start: float
    end: float
    token: str


def _normalize_header(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _find_column(fieldnames: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    normalized: Dict[str, str] = {}
    for name in fieldnames:
        key = _normalize_header(name)
        if key and key not in normalized:
            normalized[key] = name
    for alias in aliases:
        resolved = normalized.get(_normalize_header(alias))
        if resolved:
            return resolved
    return None


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _normalize_gender(value: str) -> str:
    lowered = _clean(value).lower()
    return GENDER_ALIASES.get(lowered, lowered)


def _normalize_instrument(value: str) -> str:
    lowered = _clean(value).lower()
    if lowered in INSTRUMENT_ALIASES:
        return INSTRUMENT_ALIASES[lowered]
    return lowered


def _normalize_sargam(value: str) -> str:
    return _clean(value).replace("'", "").replace("`", "").replace("’", "").replace("·", "").lower()


def _parse_filter_values(values: Optional[Sequence[str]]) -> set[str]:
    if not values:
        return set()
    out: set[str] = set()
    for raw in values:
        for token in str(raw or "").split(","):
            cleaned = token.strip().lower()
            if cleaned:
                out.add(cleaned)
    return out


def _discover_candidates(results_dir: Path) -> Tuple[Dict[str, List[TranscriptionCandidate]], Dict[str, List[TranscriptionCandidate]]]:
    by_dir: Dict[Path, TranscriptionCandidate] = {}

    for path in results_dir.rglob("transcribed_notes.csv"):
        rec_dir = path.parent.resolve()
        stem = rec_dir.name.lower()
        cand = by_dir.get(rec_dir)
        if cand is None:
            cand = TranscriptionCandidate(recording_dir=rec_dir, stem=stem)
            by_dir[rec_dir] = cand
        cand.original_csv = path.resolve()

    for path in results_dir.rglob("transcription_edited.csv"):
        path_resolved = path.resolve()
        if path.parent.name == "transcription_edits":
            rec_dir = path.parent.parent.resolve()
        else:
            rec_dir = path.parent.resolve()
        stem = rec_dir.name.lower()
        cand = by_dir.get(rec_dir)
        if cand is None:
            cand = TranscriptionCandidate(recording_dir=rec_dir, stem=stem)
            by_dir[rec_dir] = cand
        cand.edited_csv = path_resolved

    stem_map: Dict[str, List[TranscriptionCandidate]] = defaultdict(list)
    basename_map: Dict[str, List[TranscriptionCandidate]] = defaultdict(list)

    for cand in by_dir.values():
        stem_key = cand.stem.lower()
        stem_map[stem_key].append(cand)
        basename_map[stem_key].append(cand)
        for ext in AUDIO_EXTS:
            basename_map[f"{stem_key}{ext}"].append(cand)

    return stem_map, basename_map


def _match_row_to_candidate(
    filename: str,
    stem_map: Dict[str, List[TranscriptionCandidate]],
    basename_map: Dict[str, List[TranscriptionCandidate]],
) -> Tuple[Optional[TranscriptionCandidate], Optional[str]]:
    token = _clean(filename)
    if not token:
        return None, "empty filename token"

    base = os.path.basename(token)
    base_lower = base.lower()
    stem = Path(base).stem if Path(base).suffix else base
    stem_lower = stem.lower()

    stem_matches = stem_map.get(stem_lower, [])
    if len(stem_matches) == 1:
        return stem_matches[0], None
    if len(stem_matches) > 1:
        return None, f"ambiguous stem match '{filename}' ({len(stem_matches)} candidates)"

    base_matches = basename_map.get(base_lower, [])
    if len(base_matches) == 1:
        return base_matches[0], None
    if len(base_matches) > 1:
        return None, f"ambiguous basename match '{filename}' ({len(base_matches)} candidates)"

    return None, f"no transcription candidate found for '{filename}'"


def _read_ground_truth_rows(csv_path: Path) -> List[GroundTruthRow]:
    rows: List[GroundTruthRow] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError("Ground-truth CSV has no headers.")

        filename_col = _find_column(fieldnames, ["filename", "file", "audio", "name"])
        if not filename_col:
            raise ValueError("Ground-truth CSV is missing a filename column.")

        raga_col = _find_column(fieldnames, ["raga"])
        tonic_col = _find_column(fieldnames, ["tonic"])
        instrument_col = _find_column(fieldnames, ["instrument_type", "instrument"])
        gender_col = _find_column(fieldnames, ["vocalist_gender", "gender"])

        for row_num, raw in enumerate(reader, start=2):
            rows.append(
                GroundTruthRow(
                    row_num=row_num,
                    filename=_clean(raw.get(filename_col)),
                    raga=_clean(raw.get(raga_col)) if raga_col else "",
                    tonic=_clean(raw.get(tonic_col)) if tonic_col else "",
                    instrument=_clean(raw.get(instrument_col)) if instrument_col else "",
                    gender=_clean(raw.get(gender_col)) if gender_col else "",
                )
            )
    return rows


def _load_token_notes(transcription_csv: Path) -> Tuple[List[TokenNote], Optional[str]]:
    notes: List[TokenNote] = []
    try:
        with transcription_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                return [], "CSV has no headers"

            sargam_col = _find_column(fieldnames, ["sargam"])
            pitch_class_col = _find_column(fieldnames, ["pitch_class", "pitchclass"])
            start_col = _find_column(fieldnames, ["start", "start_time", "starttime"])
            end_col = _find_column(fieldnames, ["end", "end_time", "endtime"])
            pitch_midi_col = _find_column(fieldnames, ["pitch_midi", "pitchmidi", "midi"])

            for idx, raw in enumerate(reader):
                sargam_raw = _clean(raw.get(sargam_col)) if sargam_col else ""
                sargam = _normalize_sargam(sargam_raw)
                if not sargam:
                    continue

                pitch_class = None
                if pitch_class_col:
                    value = _clean(raw.get(pitch_class_col))
                    if value:
                        try:
                            pitch_class = int(round(float(value))) % 12
                        except Exception:
                            pitch_class = None
                if pitch_class is None and pitch_midi_col:
                    value = _clean(raw.get(pitch_midi_col))
                    if value:
                        try:
                            pitch_class = int(round(float(value))) % 12
                        except Exception:
                            pitch_class = None
                if pitch_class is None:
                    continue

                start = float(idx)
                end = float(idx + 1)
                if start_col:
                    value = _clean(raw.get(start_col))
                    if value:
                        try:
                            start = float(value)
                        except Exception:
                            pass
                if end_col:
                    value = _clean(raw.get(end_col))
                    if value:
                        try:
                            end = float(value)
                        except Exception:
                            pass
                if end < start:
                    start, end = end, start
                if not math.isfinite(start) or not math.isfinite(end):
                    continue

                notes.append(TokenNote(start=start, end=end, token=f"{sargam}:{pitch_class}"))
    except Exception as exc:
        return [], f"failed to parse CSV: {exc}"

    if not notes:
        return [], "no tokenizable notes found (needs sargam + pitch class)"

    notes.sort(key=lambda n: (n.start, n.end))
    return notes, None


def _extract_ngrams(tokens: Sequence[str], min_len: int, max_len: int) -> Counter[Tuple[str, ...]]:
    counts: Counter[Tuple[str, ...]] = Counter()
    if not tokens:
        return counts
    upper = max(min(max_len, len(tokens)), 0)
    for n in range(min_len, upper + 1):
        for i in range(0, len(tokens) - n + 1):
            counts[tuple(tokens[i : i + n])] += 1
    return counts


def _extract_ngram_positions(tokens: Sequence[str], min_len: int, max_len: int) -> Dict[Tuple[str, ...], List[Tuple[int, int]]]:
    out: Dict[Tuple[str, ...], List[Tuple[int, int]]] = defaultdict(list)
    if not tokens:
        return out
    upper = max(min(max_len, len(tokens)), 0)
    for n in range(min_len, upper + 1):
        for i in range(0, len(tokens) - n + 1):
            out[tuple(tokens[i : i + n])].append((i, i + n))
    return out


def _compute_entropy(counts_by_raga: Dict[str, int]) -> float:
    total = sum(counts_by_raga.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts_by_raga.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_num",
        "filename",
        "raga",
        "tonic",
        "gender",
        "instrument",
        "matched_recording_dir",
        "transcription_file",
        "transcription_source",
        "token_count",
        "status",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def mine_motifs(
    ground_truth: str,
    results_dir: str,
    index_out: str,
    transcription_source: str = "auto",
    min_len: int = 3,
    max_len: int = 8,
    min_recording_support: int = 3,
    raga_filter: Optional[Sequence[str]] = None,
    gender_filter: Optional[Sequence[str]] = None,
    instrument_filter: Optional[Sequence[str]] = None,
    summary_out: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    if min_len < 3:
        raise ValueError("min_len must be >= 3.")
    if max_len < min_len:
        raise ValueError("max_len must be >= min_len.")
    if min_recording_support < 1:
        raise ValueError("min_recording_support must be >= 1.")

    ground_truth_path = Path(ground_truth).resolve()
    results_dir_path = Path(results_dir).resolve()
    index_out_path = Path(index_out).resolve()

    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {ground_truth_path}")
    if not results_dir_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir_path}")

    raga_filters = _parse_filter_values(raga_filter)
    gender_filters = _parse_filter_values(gender_filter)
    instrument_filters = _parse_filter_values(instrument_filter)

    rows = _read_ground_truth_rows(ground_truth_path)
    stem_map, basename_map = _discover_candidates(results_dir_path)

    summary_rows: List[Dict[str, Any]] = []
    missing_files: List[str] = []

    valid_records: List[Dict[str, Any]] = []

    for row in rows:
        normalized_raga = row.raga.strip().lower()
        normalized_gender = _normalize_gender(row.gender)
        normalized_instrument = _normalize_instrument(row.instrument)
        status = "processed"
        message = ""
        matched_dir = ""
        transcription_file = ""
        transcription_mode = ""
        token_notes: List[TokenNote] = []

        if not row.filename:
            status = "missing_transcription"
            message = "empty filename in CSV row"
        elif not row.raga:
            status = "missing_raga"
            message = "missing raga in CSV row"
        elif raga_filters and normalized_raga not in raga_filters:
            status = "filtered_out"
            message = "raga filter"
        elif gender_filters and normalized_gender not in gender_filters:
            status = "filtered_out"
            message = "gender filter"
        elif instrument_filters and normalized_instrument not in instrument_filters:
            status = "filtered_out"
            message = "instrument filter"
        else:
            matched, reason = _match_row_to_candidate(row.filename, stem_map, basename_map)
            if matched is None:
                status = "missing_transcription"
                message = reason or "no candidate"
            else:
                matched_dir = str(matched.recording_dir)
                selected_csv, selected_source = matched.resolve(transcription_source)
                if selected_csv is None:
                    status = "missing_transcription"
                    message = f"{selected_source} transcription missing"
                else:
                    transcription_file = str(selected_csv)
                    transcription_mode = selected_source
                    token_notes, parse_error = _load_token_notes(selected_csv)
                    if parse_error:
                        status = "invalid_transcription"
                        message = parse_error
                    elif len(token_notes) < min_len:
                        status = "too_short"
                        message = f"token count {len(token_notes)} < min_len {min_len}"
                    else:
                        valid_records.append(
                            {
                                "row": row,
                                "raga": row.raga.strip(),
                                "recording_id": str(matched.recording_dir),
                                "token_notes": token_notes,
                            }
                        )

        if status == "missing_transcription":
            missing_files.append(row.filename or f"row:{row.row_num}")
            if not quiet:
                print(f"[WARN] Row {row.row_num}: {row.filename} -> {message}")

        summary_rows.append(
            {
                "row_num": row.row_num,
                "filename": row.filename,
                "raga": row.raga,
                "tonic": row.tonic,
                "gender": normalized_gender,
                "instrument": normalized_instrument,
                "matched_recording_dir": matched_dir,
                "transcription_file": transcription_file,
                "transcription_source": transcription_mode,
                "token_count": len(token_notes),
                "status": status,
                "message": message,
            }
        )

    raga_recordings: Dict[str, set[str]] = defaultdict(set)
    per_raga_stats: Dict[str, Dict[Tuple[str, ...], Dict[str, Any]]] = defaultdict(dict)
    motif_raga_supports: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    motif_global_recordings: Dict[Tuple[str, ...], set[str]] = defaultdict(set)

    for rec in valid_records:
        raga = rec["raga"]
        recording_id = rec["recording_id"]
        token_notes = rec["token_notes"]
        tokens = [item.token for item in token_notes]

        raga_recordings[raga].add(recording_id)
        ngram_counts = _extract_ngrams(tokens, min_len=min_len, max_len=max_len)
        for motif, count in ngram_counts.items():
            stats = per_raga_stats[raga].setdefault(
                motif,
                {
                    "recording_ids": set(),
                    "total_occurrences": 0,
                },
            )
            first_in_recording = recording_id not in stats["recording_ids"]
            stats["recording_ids"].add(recording_id)
            stats["total_occurrences"] += int(count)
            motif_global_recordings[motif].add(recording_id)
            if first_in_recording:
                motif_raga_supports[motif][raga] += 1

    ragas_payload: Dict[str, Any] = {}
    total_kept_motifs = 0
    for raga, motif_map in per_raga_stats.items():
        recording_count = len(raga_recordings.get(raga, set()))
        kept: List[Dict[str, Any]] = []
        for motif, stats in motif_map.items():
            recording_support = len(stats["recording_ids"])
            if recording_support < min_recording_support:
                continue

            total_occurrences = int(stats["total_occurrences"])
            global_support = len(motif_global_recordings.get(motif, set()))
            counts_by_raga = dict(motif_raga_supports.get(motif, {}))
            entropy = _compute_entropy(counts_by_raga)
            specificity = (recording_support / global_support) if global_support > 0 else 0.0
            weight = specificity / (1.0 + entropy)
            coverage = (recording_support / recording_count) if recording_count > 0 else 0.0

            kept.append(
                {
                    "motif": list(motif),
                    "motif_str": " ".join(motif),
                    "length": len(motif),
                    "recording_support": recording_support,
                    "total_occurrences": total_occurrences,
                    "coverage": round(coverage, 6),
                    "global_recording_support": global_support,
                    "specificity": round(specificity, 6),
                    "entropy": round(entropy, 6),
                    "weight": round(weight, 6),
                }
            )

        kept.sort(
            key=lambda item: (
                int(item["recording_support"]),
                float(item["weight"]),
                int(item["total_occurrences"]),
                int(item["length"]),
            ),
            reverse=True,
        )
        total_kept_motifs += len(kept)
        ragas_payload[raga] = {
            "recording_count": recording_count,
            "kept_motif_count": len(kept),
            "motifs": kept,
        }

    created_at = datetime.now(timezone.utc).isoformat()
    index_payload: Dict[str, Any] = {
        "version": "1",
        "created_at": created_at,
        "params": {
            "ground_truth": str(ground_truth_path),
            "results_dir": str(results_dir_path),
            "transcription_source": transcription_source,
            "min_len": min_len,
            "max_len": max_len,
            "min_recording_support": min_recording_support,
            "raga_filter": sorted(raga_filters),
            "gender_filter": sorted(gender_filters),
            "instrument_filter": sorted(instrument_filters),
        },
        "totals": {
            "csv_rows": len(rows),
            "valid_recordings": len(valid_records),
            "kept_ragas": len(ragas_payload),
            "kept_motifs": total_kept_motifs,
            "missing_transcriptions": sum(1 for row in summary_rows if row["status"] == "missing_transcription"),
        },
        "ragas": ragas_payload,
    }

    index_out_path.parent.mkdir(parents=True, exist_ok=True)
    with index_out_path.open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, indent=2)

    if summary_out:
        _write_summary_csv(Path(summary_out).resolve(), summary_rows)

    if not quiet:
        print(
            f"Motif mining complete. Rows={len(rows)} valid={len(valid_records)} "
            f"ragas={len(ragas_payload)} motifs={total_kept_motifs}"
        )
        if missing_files:
            print("\nMissing transcription rows:")
            for item in missing_files:
                print(f"- {item}")

    return {
        "index": index_payload,
        "summary_rows": summary_rows,
        "missing_files": missing_files,
    }


def _infer_phrase_indices(notes: Sequence[TokenNote], phrase_gap: float) -> Tuple[List[int], Dict[int, Tuple[float, float]]]:
    if not notes:
        return [], {}
    idxs: List[int] = []
    ranges: Dict[int, Tuple[float, float]] = {}
    phrase_idx = 0
    prev_end = notes[0].end
    for note in notes:
        if (note.start - prev_end) > phrase_gap:
            phrase_idx += 1
        idxs.append(phrase_idx)
        start, end = ranges.get(phrase_idx, (note.start, note.end))
        ranges[phrase_idx] = (min(start, note.start), max(end, note.end))
        prev_end = note.end
    return idxs, ranges


def score_transcription(
    index_path: str,
    transcription_path: str,
    top_k: int = 5,
    out_path: Optional[str] = None,
    phrase_gap: float = 1.0,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
) -> Dict[str, Any]:
    index_file = Path(index_path).resolve()
    transcription_file = Path(transcription_path).resolve()
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    if not transcription_file.exists():
        raise FileNotFoundError(f"Transcription CSV not found: {transcription_file}")

    with index_file.open("r", encoding="utf-8") as handle:
        index = json.load(handle)

    params = index.get("params", {})
    effective_min_len = int(min_len if min_len is not None else params.get("min_len", 3))
    effective_max_len = int(max_len if max_len is not None else params.get("max_len", 8))
    if effective_min_len < 3:
        raise ValueError("min_len must be >= 3.")
    if effective_max_len < effective_min_len:
        raise ValueError("max_len must be >= min_len.")

    token_notes, parse_error = _load_token_notes(transcription_file)
    if parse_error:
        raise ValueError(f"Cannot score transcription: {parse_error}")
    tokens = [item.token for item in token_notes]
    ngram_positions = _extract_ngram_positions(tokens, min_len=effective_min_len, max_len=effective_max_len)

    phrase_idxs, phrase_ranges = _infer_phrase_indices(token_notes, phrase_gap=phrase_gap)

    ranked: List[Dict[str, Any]] = []
    phrase_overlay_accumulator: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for raga, payload in (index.get("ragas") or {}).items():
        motifs = payload.get("motifs") or []
        score = 0.0
        hit_count = 0
        motif_hits: List[Dict[str, Any]] = []

        for entry in motifs:
            motif_tokens = tuple(entry.get("motif") or [])
            if not motif_tokens:
                continue
            positions = ngram_positions.get(motif_tokens, [])
            if not positions:
                continue

            weight = float(entry.get("weight", 1.0))
            occurrences = len(positions)
            motif_score = weight * occurrences
            score += motif_score
            hit_count += occurrences

            motif_str = str(entry.get("motif_str") or " ".join(motif_tokens))
            motif_hits.append(
                {
                    "motif": motif_str,
                    "length": int(entry.get("length", len(motif_tokens))),
                    "count": occurrences,
                    "weight": weight,
                    "score": round(motif_score, 6),
                }
            )

            for start_idx, end_idx in positions:
                phrase_idx = phrase_idxs[start_idx] if start_idx < len(phrase_idxs) else 0
                span_start = token_notes[start_idx].start
                span_end = token_notes[end_idx - 1].end
                key = (phrase_idx, raga)
                agg = phrase_overlay_accumulator.setdefault(
                    key,
                    {
                        "phrase_idx": phrase_idx,
                        "raga": raga,
                        "start": span_start,
                        "end": span_end,
                        "hit_count": 0,
                        "score": 0.0,
                        "motifs": set(),
                    },
                )
                agg["hit_count"] += 1
                agg["score"] += weight
                agg["start"] = min(float(agg["start"]), span_start)
                agg["end"] = max(float(agg["end"]), span_end)
                agg["motifs"].add(motif_str)

        motif_hits.sort(key=lambda item: (float(item["score"]), int(item["count"])), reverse=True)
        ranked.append(
            {
                "raga": raga,
                "score": round(score, 6),
                "hit_count": hit_count,
                "matched_motif_count": len(motif_hits),
                "top_motifs": motif_hits[:20],
            }
        )

    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    ranked = ranked[: max(top_k, 1)]
    top_ragas = {item["raga"] for item in ranked}

    phrase_overlays: List[Dict[str, Any]] = []
    for (phrase_idx, raga), entry in phrase_overlay_accumulator.items():
        if raga not in top_ragas:
            continue
        phrase_start, phrase_end = phrase_ranges.get(phrase_idx, (entry["start"], entry["end"]))
        phrase_overlays.append(
            {
                "phrase_idx": int(phrase_idx),
                "raga": raga,
                "start": round(float(phrase_start), 6),
                "end": round(float(phrase_end), 6),
                "hit_count": int(entry["hit_count"]),
                "score": round(float(entry["score"]), 6),
                "motifs": sorted(entry["motifs"]),
            }
        )
    phrase_overlays.sort(key=lambda item: (int(item["phrase_idx"]), -float(item["score"]), item["raga"]))

    payload = {
        "transcription": str(transcription_file),
        "token_count": len(tokens),
        "min_len": effective_min_len,
        "max_len": effective_max_len,
        "phrase_gap": phrase_gap,
        "ranked_ragas": ranked,
        "phrase_overlays": phrase_overlays,
    }

    if out_path:
        out_file = Path(out_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raga-wide motif mining and lookup.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine = subparsers.add_parser("mine", help="Mine raga-wide motifs from transcriptions.")
    mine.add_argument("--ground-truth", required=True, help="Path to ground-truth CSV.")
    mine.add_argument("--results-dir", required=True, help="Root directory containing transcription outputs.")
    mine.add_argument("--index-out", required=True, help="Output path for motif index JSON.")
    mine.add_argument(
        "--transcription-source",
        choices=["auto", "edited", "original"],
        default="auto",
        help="Transcription selection mode (default: auto).",
    )
    mine.add_argument("--min-len", type=int, default=3, help="Minimum motif length (default: 3).")
    mine.add_argument("--max-len", type=int, default=8, help="Maximum motif length (default: 8).")
    mine.add_argument(
        "--min-recording-support",
        type=int,
        default=3,
        help="Minimum unique recording support per raga motif (default: 3).",
    )
    mine.add_argument("--raga-filter", action="append", default=[], help="Optional raga filter (repeat or comma-separated).")
    mine.add_argument(
        "--gender-filter",
        action="append",
        default=[],
        help="Optional gender filter (repeat or comma-separated, e.g. male,female).",
    )
    mine.add_argument(
        "--instrument-filter",
        action="append",
        default=[],
        help="Optional instrument filter (repeat or comma-separated).",
    )
    mine.add_argument("--summary-out", default=None, help="Optional summary CSV output path.")
    mine.add_argument("--quiet", action="store_true", help="Suppress warning and summary prints.")

    score = subparsers.add_parser("score", help="Score a transcription against a motif index.")
    score.add_argument("--index", required=True, help="Path to motif index JSON.")
    score.add_argument("--transcription", required=True, help="Path to transcription CSV.")
    score.add_argument("--top-k", type=int, default=5, help="Number of top ragas to return (default: 5).")
    score.add_argument("--out", default=None, help="Optional JSON output path.")
    score.add_argument(
        "--phrase-gap",
        type=float,
        default=1.0,
        help="Gap threshold (seconds) for inferred phrase buckets in overlay output.",
    )
    score.add_argument("--min-len", type=int, default=None, help="Override minimum motif length for scoring.")
    score.add_argument("--max-len", type=int, default=None, help="Override maximum motif length for scoring.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "mine":
        mine_motifs(
            ground_truth=args.ground_truth,
            results_dir=args.results_dir,
            index_out=args.index_out,
            transcription_source=args.transcription_source,
            min_len=args.min_len,
            max_len=args.max_len,
            min_recording_support=args.min_recording_support,
            raga_filter=args.raga_filter,
            gender_filter=args.gender_filter,
            instrument_filter=args.instrument_filter,
            summary_out=args.summary_out,
            quiet=args.quiet,
        )
        return 0

    if args.command == "score":
        result = score_transcription(
            index_path=args.index,
            transcription_path=args.transcription,
            top_k=args.top_k,
            out_path=args.out,
            phrase_gap=args.phrase_gap,
            min_len=args.min_len,
            max_len=args.max_len,
        )
        print(json.dumps(result, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
