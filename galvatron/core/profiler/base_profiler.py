import os


class BaseProfiler:
    def __init__(self, args):
        self.args = args
        self.time_path = None
        self.mem_path = None

    def set_path(self, path):
        self.path = path

    def set_model_name(self, name):
        self.model_name = name

    def _parallel_suffix(self, tp=None, ep=None):
        """Suffix to disambiguate profile output files across different TP/EP
        sweeps. Keeps the legacy filename when both degrees are 1 so existing
        non-FSEP workflows stay unchanged. Pass explicit tp/ep to compute the
        suffix for a specific sweep point (used by the aggregator to iterate
        over per-tp/ep files the inner launcher wrote).
        """
        if tp is None:
            tp = getattr(self.args, "global_tp_deg", 1) or 1
        if ep is None:
            ep = getattr(self.args, "global_ep_deg", 1) or 1
        parts = []
        if tp > 1 or ep > 1:
            parts.append(f"tp{tp}")
            parts.append(f"ep{ep}")
        return ("_" + "_".join(parts)) if parts else ""

    def time_profiling_path_for(self, tp, ep):
        """Uncached variant of time_profiling_path() for an explicit (tp, ep)."""
        assert self.model_name is not None, "Should specify the model name!"
        fname = "configs/computation_profiling_%s_%s%s.json" % (
            self.args.mixed_precision,
            self.model_name,
            self._parallel_suffix(tp=tp, ep=ep),
        )
        return os.path.join(self.path, fname)

    def memory_profiling_path_for(self, tp, ep):
        """Uncached variant of memory_profiling_path() for an explicit (tp, ep)."""
        assert self.model_name is not None, "Should specify the model name!"
        fname = "configs/memory_profiling_%s_%s%s.json" % (
            self.args.mixed_precision,
            self.model_name,
            self._parallel_suffix(tp=tp, ep=ep),
        )
        return os.path.join(self.path, fname)

    def memory_profiling_path(self):
        """Get memory profiling path

        Returns:
            str: Path to memory profiling config file
        """
        if self.mem_path is not None:
            return self.mem_path
        assert self.model_name is not None, "Should specify the model name!"
        args = self.args
        memory_config_path = "configs/memory_profiling_%s_%s%s.json" % (
            args.mixed_precision,
            self.model_name,
            self._parallel_suffix(),
        )
        self.mem_path = os.path.join(self.path, memory_config_path)
        return self.mem_path

    def time_profiling_path(self):
        """Get time profiling path

        Returns:
            str: Path to time profiling config file
        """
        if self.time_path is not None:
            return self.time_path
        assert self.model_name is not None, "Should specify the model name!"
        args = self.args
        time_config_path = "configs/computation_profiling_%s_%s%s.json" % (
            args.mixed_precision,
            self.model_name,
            self._parallel_suffix(),
        )
        self.time_path = os.path.join(self.path, time_config_path)
        return self.time_path
