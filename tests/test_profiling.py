"""Interval arithmetic behind the overlap analysis.

The overlap number is only as trustworthy as this arithmetic, and it is easy to
get subtly wrong (double-counting concurrent kernels, or reporting overlap
between an event and itself). These tests pin it down on hand-built traces
where the right answer is obvious by inspection.
"""

from __future__ import annotations

import json

import pytest

from minidist.profiling import _intersect, _merge, _subtract, _total, analyze_trace


def test_merge_combines_overlapping_intervals():
    assert _merge([(0, 2), (1, 3), (5, 6)]) == [(0, 3), (5, 6)]
    assert _merge([(0, 1), (1, 2)]) == [(0, 2)], "touching intervals are contiguous"
    assert _merge([]) == []


def test_intersect():
    a = [(0, 10)]
    b = [(2, 4), (6, 12)]
    assert _intersect(a, b) == [(2, 4), (6, 10)]
    assert _intersect([(0, 1)], [(2, 3)]) == []


def test_subtract():
    assert _subtract([(0, 10)], [(2, 4)]) == [(0, 2), (4, 10)]
    assert _subtract([(0, 10)], [(0, 10)]) == []
    assert _subtract([(0, 10)], []) == [(0, 10)]
    assert _total(_subtract([(0, 10)], [(0, 3), (7, 10)])) == 4


def _write_trace(path, events):
    path.write_text(json.dumps({"traceEvents": events}))


def _kernel(name, ts_us, dur_us):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts_us, "dur": dur_us}


def test_analyze_trace_fully_overlapped(tmp_path):
    """A collective running entirely alongside compute is fully hidden."""
    trace = tmp_path / "t.json"
    _write_trace(trace, [
        _kernel("ampere_sgemm_128x64", 0, 1000),
        _kernel("nccl:all_reduce", 200, 400),
    ])
    report = analyze_trace(trace)
    assert report.device == "cuda"
    assert report.comm_s == pytest.approx(400e-6)
    assert report.overlapped_s == pytest.approx(400e-6)
    assert report.overlap_fraction == 1.0
    assert report.exposed_comm_s == 0.0


def test_analyze_trace_fully_exposed(tmp_path):
    """A collective with no concurrent compute is entirely a stall."""
    trace = tmp_path / "t.json"
    _write_trace(trace, [
        _kernel("ampere_sgemm_128x64", 0, 500),
        _kernel("nccl:all_gather", 500, 500),
    ])
    report = analyze_trace(trace)
    assert report.overlapped_s == 0.0
    assert report.overlap_fraction == 0.0
    assert report.exposed_comm_s == pytest.approx(500e-6)
    assert report.exposed_fraction_of_step == pytest.approx(0.5)


def test_analyze_trace_partial_overlap_and_classification(tmp_path):
    trace = tmp_path / "t.json"
    _write_trace(trace, [
        _kernel("ampere_sgemm_128x64", 0, 600),
        _kernel("nccl:all_reduce", 400, 400),
        _kernel("ncclDevKernel_ReduceScatter", 800, 100),
    ])
    report = analyze_trace(trace)
    assert report.comm_s == pytest.approx(500e-6)
    assert report.overlapped_s == pytest.approx(200e-6), "only 400-600us has compute alongside"
    assert report.exposed_comm_s == pytest.approx(300e-6)
    assert set(report.per_kind_s) == {"all_reduce", "reduce_scatter"}
    assert report.per_kind_s["reduce_scatter"] == pytest.approx(100e-6)


def test_concurrent_kernels_are_not_double_counted(tmp_path):
    """Two kernels on different streams over the same window are one busy interval."""
    trace = tmp_path / "t.json"
    _write_trace(trace, [
        _kernel("sgemm_a", 0, 1000),
        _kernel("sgemm_b", 0, 1000),
    ])
    report = analyze_trace(trace)
    assert report.compute_s == pytest.approx(1000e-6), "summing durations would give 2000us"


def test_cpu_fallback_is_labelled(tmp_path):
    """A Gloo trace has no kernels; the report must say the number is degenerate."""
    trace = tmp_path / "t.json"
    _write_trace(trace, [
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 500},
        {"ph": "X", "cat": "cpu_op", "name": "c10d::allreduce_", "ts": 500, "dur": 300},
    ])
    report = analyze_trace(trace)
    assert report.device == "cpu"
    assert "Gloo" in report.note
    assert report.comm_s == pytest.approx(300e-6)
    assert report.overlapped_s == 0.0
    assert "exposed comm" in report.format()
