try:
    import torch
    from megatron.legacy.model.rms_norm import RMSNorm as MoERMSNorm

    from .MoEModel_hybrid_parallel import (
        construct_hybrid_parallel_model,
        get_hybrid_parallel_configs,
        moe_model_hp,
    )

    def rms_reset_parameters(self):
        with torch.no_grad():
            torch.nn.init.ones_(self.weight)

    MoERMSNorm.reset_parameters = rms_reset_parameters
except ImportError:
    # The runtime stack pulls in compiled CUDA kernels
    # (moe_all_to_all_kernels, greedy_balancer). On CPU-only environments
    # — e.g. the cost-model regression suite, the search driver — those
    # kernels aren't built, but the cost-model subpackage doesn't need
    # them. Surface the import error only at use site, not at package
    # load time, so ``from galvatron.models.moe.cost_model import ...``
    # works without a GPU build.
    pass
