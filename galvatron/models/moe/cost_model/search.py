"""Programmatic search interface over :class:`PPCostModel`.

The cost model answers "how much does this one config cost?". The
search layer enumerates a structured config space, scores each entry
via :meth:`PPCostModel.query`, applies feasibility filters (OOM,
trust-source), and returns ranked results.

Two layers, mirroring the cost-model split:

  - :func:`enumerate_configs` — pure structural enumerator. Yields the
    parallel layouts ``{pp, dp, tp, ep, dp_mode, fsep}`` that satisfy
    ``dp × pp × tp × ep == num_gpus`` and the FSEP feasibility
    constraints. Stateless; safe to call many times.
  - :class:`MoESearcher` — object that holds a long-lived
    :class:`PPCostModel` and exposes per-call entry points an outer
    loop can drive. Single-process by design — composition into
    parallel pipelines is the *outer* loop's job, not ours.

External callers (planners, design-space probes, "compose results
across partial models" loops) build one :class:`MoESearcher` per
model (or share one across many calls) and invoke:

  - :meth:`MoESearcher.score` for a single ``(cfg, workload)`` pair
    → :class:`SearchResult`.
  - :meth:`MoESearcher.rank` for an enumerated batch
    → :class:`RankedSearch` with viable + infeasible buckets.

Neither method spawns processes or threads; the cost model itself is
already fast enough that adding parallelism cascades CPU usage
without measurable speedup (see ``doc/cost_model_guide.md`` §5 for
the multiprocessing benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from .base import CostQuery
from .pp import PPCostModel

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """One scored config from a search.

    Either ``query`` is populated (config is viable, possibly with
    filter caveats), or ``error`` is populated (the cost-model raised
    or a feasibility filter rejected the config). Both can never be
    populated simultaneously; ``viable`` reports the disposition.

    Attributes:
        cfg: parallel layout dict ``{pp, dp, tp, ep, dp_mode, fsep}``.
        query: focused cost result. ``None`` if the config errored.
        error: human-readable rejection reason. ``None`` for viable.
        num_attention_layers / num_expert_layers: layer layout this
            config was scored against.
    """

    cfg: Dict[str, Any]
    query: Optional[CostQuery] = None
    error: Optional[str] = None
    num_attention_layers: int = 0
    num_expert_layers: int = 0

    @property
    def viable(self) -> bool:
        """True if this config has a usable cost estimate (no error,
        passed every filter)."""
        return self.error is None and self.query is not None

    @property
    def iter_ms(self) -> float:
        """Iteration time, NaN when not viable."""
        return self.query.iter_ms if self.query is not None else float("nan")

    @property
    def max_stage_ms(self) -> float:
        return self.query.max_stage_ms if self.query is not None else float("nan")

    @property
    def peak_memory_mb(self) -> float:
        return self.query.peak_memory_mb if self.query is not None else float("nan")


@dataclass
class RankedSearch:
    """Output of :meth:`MoESearcher.rank` — sorted viable configs plus
    a bucket of infeasible ones.

    Useful for outer loops composing results across multiple searches
    (e.g. one search per "partial model"): pull ``best`` /
    ``viable[:k]`` from each, combine externally.

    Attributes:
        viable: every config that passed all filters, sorted by
            ``sort_key`` ascending (default ``(iter_ms,
            peak_memory_mb)``).
        infeasible: every config that errored or got filtered out,
            with ``error`` populated explaining why.
        num_total: total configs enumerated (= ``len(viable) +
            len(infeasible) + skipped``); informational.
        sort_key: the sort function that produced this ranking, so
            callers re-sorting by a different criterion can call
            :meth:`resort` explicitly.
    """

    viable: List[SearchResult] = field(default_factory=list)
    infeasible: List[SearchResult] = field(default_factory=list)
    num_total: int = 0
    sort_key: Optional[Callable[[SearchResult], Any]] = None

    @property
    def best(self) -> Optional[SearchResult]:
        """Top viable result, or ``None`` when the search admitted no
        viable configs."""
        return self.viable[0] if self.viable else None

    def top(self, k: int) -> List[SearchResult]:
        """First ``k`` viable results."""
        return self.viable[:k]

    def resort(self, key: Callable[[SearchResult], Any]) -> "RankedSearch":
        """Return a new :class:`RankedSearch` with ``viable`` re-sorted
        by ``key``. ``infeasible`` is preserved as-is."""
        return RankedSearch(
            viable=sorted(self.viable, key=key),
            infeasible=list(self.infeasible),
            num_total=self.num_total,
            sort_key=key,
        )


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def _divisors(value: int) -> List[int]:
    return [d for d in range(1, value + 1) if value % d == 0]


def enumerate_configs(
    num_gpus: int,
    num_layers: int,
    num_experts: int,
    dp_modes: Iterable[str] = ("zero2sdp", "zero3"),
    fsep_modes: Iterable[str] = ("on", "off"),
) -> Iterator[Dict[str, Any]]:
    """Yield every ``{pp, dp, tp, ep, dp_mode, fsep}`` tuple that
    satisfies the structural constraints:

      - ``dp × pp × tp × ep == num_gpus``
      - ``num_layers % pp == 0``
      - ``ep ≤ num_experts``
      - FSEP-on requires ``tp × ep == per_stage_world`` AND
        ``ep | num_experts``

    Viability checks (memory budget, trust filter) are applied by the
    caller (typically :meth:`MoESearcher.rank`)."""
    dp_modes = list(dp_modes)
    fsep_modes = list(fsep_modes)
    for pp in _divisors(num_gpus):
        if num_layers % pp != 0:
            continue
        per_stage_world = num_gpus // pp
        for dp in _divisors(per_stage_world):
            for tp in _divisors(per_stage_world // dp):
                ep = per_stage_world // (dp * tp)
                if ep < 1 or ep > num_experts:
                    continue
                for dp_mode in dp_modes:
                    for fsep in fsep_modes:
                        if fsep == "on":
                            # FSEP requires tp × ep == per_stage_world
                            # AND ep | num_experts (so capacity_per_device
                            # is integer and ≥ 1, satisfying ep × cap ≥ E).
                            if tp * ep != per_stage_world:
                                continue
                            if num_experts % ep != 0:
                                continue
                        yield dict(
                            pp=pp,
                            dp=dp,
                            tp=tp,
                            ep=ep,
                            dp_mode=dp_mode,
                            fsep=fsep,
                        )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _trust_keep(query: CostQuery, trust: str) -> bool:
    """Return True if ``query`` passes the trust-source filter.

    - ``any``: no filter.
    - ``calibrated``: both time and memory must come from
      ``runtime_profile`` (rules out fully-analytical configs without
      a calibration anchor).
    - ``sample``: both must come from an exact-N profile sample
      (strictest — only configs with a real measurement at the
      requested ``num_layers`` survive).
    """
    if trust == "any":
        return True
    if trust == "calibrated":
        return query.time_source.startswith(
            "runtime_profile"
        ) and query.memory_source.startswith("runtime_profile")
    if trust == "sample":
        return "+sample" in query.time_source and "+sample" in query.memory_source
    raise ValueError(f"unknown trust mode: {trust!r}")


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------


def _default_sort_key(result: SearchResult) -> Tuple[float, float]:
    """Default ranking: iter_ms ascending, ties broken by lower peak
    memory."""
    return (result.iter_ms, result.peak_memory_mb)


class MoESearcher:
    """Object-style search API.

    Build once per model, call :meth:`score` / :meth:`rank` many times.
    The underlying :class:`PPCostModel` is held for the searcher's
    lifetime; profile JSONs load once at construction.

    Single-process by design. To parallelize across models, build one
    :class:`MoESearcher` per model in the outer loop. To parallelize
    across configs within a single search, see the multiprocessing
    note in :func:`cost_model_guide.md` §5 — generally not worth the
    IPC overhead since :meth:`score` is ~20 µs.

    Example::

        searcher = MoESearcher("mixtral-8x7b-e8k2")

        # Score one config
        cfg = {"pp": 1, "dp": 1, "tp": 1, "ep": 4,
               "dp_mode": "zero2sdp", "fsep": "off"}
        result = searcher.score(
            cfg, num_layers=4, num_gpus=4, global_bsz=4, seq_len=4096,
        )
        print(result.iter_ms, result.peak_memory_mb)

        # Rank the full enumerated space
        ranked = searcher.rank(
            num_gpus=4, num_layers=4, global_bsz=4,
            gpu_memory_mb=45000, trust="calibrated",
        )
        print(ranked.best.cfg, ranked.best.query.iter_ms)

    Composition pattern (outer loop over partial models)::

        results_per_partial = []
        for partial_n in (4, 8, 12, 16):
            ranked = searcher.rank(
                num_gpus=8, num_layers=partial_n, global_bsz=8,
                gpu_memory_mb=80000, trust="calibrated",
            )
            results_per_partial.append((partial_n, ranked.best))
        # … combine externally
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        cost_model: Optional[PPCostModel] = None,
        mixed_precision: str = "bf16",
    ):
        if cost_model is None:
            if model_name is None:
                raise ValueError(
                    "MoESearcher requires either ``cost_model`` or ``model_name``"
                )
            cost_model = PPCostModel(model_name, mixed_precision=mixed_precision)
        self.cost_model = cost_model
        # Convenience accessor for outer loops that want the underlying
        # model name without reaching through the cost model.
        self.model_name = self.cost_model.model_name

    # ------------------------------------------------------------------
    # Single-config entry point
    # ------------------------------------------------------------------

    def score(
        self,
        cfg: Dict[str, Any],
        *,
        num_layers: int,
        num_gpus: int,
        global_bsz: int,
        seq_len: int = 4096,
        recompute: bool = True,
        num_attention_layers: Optional[int] = None,
        num_expert_layers: Optional[int] = None,
        num_stages_behind: int = 0,
    ) -> SearchResult:
        """Score one parallel layout against one workload.

        ``cfg`` is the parallel layout dict ``{pp, dp, tp, ep,
        dp_mode, fsep}`` — same shape :func:`enumerate_configs`
        yields. Workload params (``num_layers``, ``num_gpus``,
        ``global_bsz``, ``seq_len``, ``recompute``, asymmetric layer
        counts) are kwargs.

        Returns a :class:`SearchResult`. On error (invalid micro-batch
        partition, missing profile, asymmetric request without a
        per-component profile) the result has ``error`` populated and
        ``query=None``; the caller can decide whether to surface or
        ignore it.

        ``num_attention_layers`` / ``num_expert_layers`` default to
        ``num_layers`` (1:1, the runtime layout). Setting them
        differently asks the cost model to project a hypothetical
        layout — useful for "should we add one extra MoE layer?"
        probes.

        ``num_stages_behind`` is forwarded to
        :meth:`PPCostModel.estimate` — adds a uniform activation-memory
        reserve to every non-last stage (count of extra microbatches,
        default 0 = no reserve, calibrated baseline). See
        :class:`PPCostModel` for details.
        """
        if num_attention_layers is None:
            num_attention_layers = num_layers
        if num_expert_layers is None:
            num_expert_layers = num_layers

        # ``micro_bsz`` is the GLOBAL microbatch size (total samples per
        # forward-backward step across all data-dim-sharing ranks). With
        # ``dp × ep`` ranks sharing the data dim, each rank gets
        # ``micro_bsz // (dp × ep)`` samples per microbatch step. We
        # default to ``micro_bsz = global_bsz`` (one microbatch per
        # optimizer step), which preserves the calibrated single-
        # microbatch physics of today's runtime profile sweep. Configs
        # where ``dp × ep > micro_bsz`` are filtered out — per-rank
        # batch would be < 1 sample.
        micro_bsz = global_bsz
        if cfg["dp"] * cfg["ep"] > micro_bsz:
            return SearchResult(
                cfg=cfg,
                error=f"dp*ep ({cfg['dp']}*{cfg['ep']}) > micro_bsz " f"({micro_bsz})",
                num_attention_layers=num_attention_layers,
                num_expert_layers=num_expert_layers,
            )
        if micro_bsz % (cfg["dp"] * cfg["ep"]) != 0:
            return SearchResult(
                cfg=cfg,
                error=f"micro_bsz ({micro_bsz}) not divisible by dp*ep "
                f"({cfg['dp']}*{cfg['ep']}={cfg['dp']*cfg['ep']})",
                num_attention_layers=num_attention_layers,
                num_expert_layers=num_expert_layers,
            )

        try:
            query = self.cost_model.query(
                num_layers=num_layers,
                num_gpus=num_gpus,
                dp=cfg["dp"],
                pp=cfg["pp"],
                tp=cfg["tp"],
                ep=cfg["ep"],
                num_attention_layers=num_attention_layers,
                num_expert_layers=num_expert_layers,
                num_stages_behind=num_stages_behind,
                micro_batch_size=micro_bsz,
                global_batch_size=global_bsz,
                seq_len=seq_len,
                sequence_parallel=True,
                zero_stage=2 if cfg["dp_mode"] == "zero2sdp" else 3,
                sdp=(cfg["dp_mode"] in ("zero2sdp", "zero3")),
                recompute=recompute,
                bwd_mult=2.0,
                fsep=(cfg["fsep"] == "on"),
            )
        except (ValueError, KeyError) as err:
            return SearchResult(
                cfg=cfg,
                error=str(err),
                num_attention_layers=num_attention_layers,
                num_expert_layers=num_expert_layers,
            )

        return SearchResult(
            cfg=cfg,
            query=query,
            num_attention_layers=query.num_attention_layers or num_attention_layers,
            num_expert_layers=query.num_expert_layers or num_expert_layers,
        )

    # ------------------------------------------------------------------
    # Batch entry point
    # ------------------------------------------------------------------

    def rank(
        self,
        *,
        num_gpus: int,
        num_layers: int,
        global_bsz: int,
        seq_len: int = 4096,
        gpu_memory_mb: Optional[float] = None,
        num_experts: int = 8,
        dp_modes: Iterable[str] = ("zero2sdp", "zero3"),
        fsep_modes: Iterable[str] = ("on", "off"),
        recompute: bool = True,
        trust: str = "any",
        num_attention_layers: Optional[int] = None,
        num_expert_layers: Optional[int] = None,
        num_stages_behind: int = 0,
        sort_key: Optional[Callable[[SearchResult], Any]] = None,
        configs: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> RankedSearch:
        """Enumerate the structural config space, score each entry,
        apply feasibility filters, return a sorted :class:`RankedSearch`.

        Filters (in order):

          1. ``score`` errors (invalid layout, missing profile, etc.)
             → infeasible bucket.
          2. ``gpu_memory_mb`` budget — if set, configs with
             ``peak_memory_mb > budget`` go to infeasible.
          3. ``trust`` filter — ``any`` keeps everything; ``calibrated``
             requires runtime-profile-anchored time + memory; ``sample``
             requires an exact-N profile sample.

        ``sort_key`` defaults to ``(iter_ms, peak_memory_mb)``
        ascending; pass a custom key for alternate rankings (e.g.
        ``lambda r: r.max_stage_ms`` for "minimize per-stage compute"
        searches).

        ``configs`` lets the outer loop supply a pre-filtered config
        list instead of running the full enumerator — useful when
        composing across partial models where the layout space is
        constrained externally.

        ``num_stages_behind`` is forwarded to every scored config; the
        OOM filter (``gpu_memory_mb``) sees the inflated peak so configs
        on the edge get filtered out as the reserve dial goes up.
        Useful for "what optima survive a more conservative memory
        budget?" probes.
        """
        if num_attention_layers is None:
            num_attention_layers = num_layers
        if num_expert_layers is None:
            num_expert_layers = num_layers

        if configs is None:
            configs = enumerate_configs(
                num_gpus=num_gpus,
                num_layers=num_layers,
                num_experts=num_experts,
                dp_modes=dp_modes,
                fsep_modes=fsep_modes,
            )

        viable: List[SearchResult] = []
        infeasible: List[SearchResult] = []
        num_total = 0

        for cfg in configs:
            num_total += 1
            result = self.score(
                cfg,
                num_layers=num_layers,
                num_gpus=num_gpus,
                global_bsz=global_bsz,
                seq_len=seq_len,
                recompute=recompute,
                num_attention_layers=num_attention_layers,
                num_expert_layers=num_expert_layers,
                num_stages_behind=num_stages_behind,
            )
            if result.error is not None or result.query is None:
                infeasible.append(result)
                continue
            if gpu_memory_mb is not None and result.peak_memory_mb > gpu_memory_mb:
                result.error = (
                    f"OOM: peak_memory_mb={result.peak_memory_mb:.0f} "
                    f"> budget {gpu_memory_mb:.0f}"
                )
                infeasible.append(result)
                continue
            if not _trust_keep(result.query, trust):
                result.error = f"filtered (trust={trust})"
                infeasible.append(result)
                continue
            viable.append(result)

        key = sort_key or _default_sort_key
        viable.sort(key=key)
        return RankedSearch(
            viable=viable,
            infeasible=infeasible,
            num_total=num_total,
            sort_key=key,
        )

    # ------------------------------------------------------------------
    # Convenience: enumerate configs without running the cost model
    # ------------------------------------------------------------------

    @staticmethod
    def enumerate(
        num_gpus: int,
        num_layers: int,
        num_experts: int = 8,
        dp_modes: Iterable[str] = ("zero2sdp", "zero3"),
        fsep_modes: Iterable[str] = ("on", "off"),
    ) -> Iterator[Dict[str, Any]]:
        """Static alias for :func:`enumerate_configs`. Useful for
        outer loops that want to inspect / filter the config space
        before scoring."""
        return enumerate_configs(
            num_gpus=num_gpus,
            num_layers=num_layers,
            num_experts=num_experts,
            dp_modes=dp_modes,
            fsep_modes=fsep_modes,
        )


__all__ = [
    "SearchResult",
    "RankedSearch",
    "MoESearcher",
    "enumerate_configs",
]
