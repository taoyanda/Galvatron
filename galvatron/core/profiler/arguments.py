from galvatron.core.common_args import galvatron_common_model_args


def galvatron_profile_args(parser):
    galvatron_common_model_args(parser)
    group = parser.add_argument_group(title="Galvatron Profiling Arguments")

    group.add_argument(
        "--profile_unit",
        choices=["attention", "mlp", "all"],
        default="all",
        help="Profile granularity",
    )
    group.add_argument(
        "--profile_metric",
        type=str,
        default="memory",
        help="Galvatron profiling metric (which pass the driver runs).",
        choices=["memory", "computation"],
    )
    group.add_argument(
        "--set_layernum_manually",
        type=int,
        default=1,
        help="Whether to set layernum config manually (doesn't overwrite other model configs).",
    )
    group.add_argument(
        "--mlp_profile_mode",
        choices=["prof_mlp", "prof_all"],
        default="prof_all",
        help="Selects MoE-MLP profiling variant when --profile_unit=mlp. "
        "'prof_all' (recommended) builds the real per-rank expert stack so "
        "the cost-model's mlp slope matches FSEP-off training. "
        "'prof_mlp' uses the legacy single dense ParallelMLP (undercounts).",
    )
    group.add_argument(
        "--set_experts_manually",
        type=int,
        default=0,
        help="Whether to set experts config manually (doesn't overwrite other model configs).",
    )
    group.add_argument("--profile_batch_size", type=int, default=None, help="Galvatron profiling batch size")
    group.add_argument("--profile_min_batch_size", type=int, default=None, help="Galvatron profiling min batch size")
    group.add_argument("--profile_max_batch_size", type=int, default=None, help="Galvatron profiling max batch size")
    group.add_argument("--profile_batch_size_step", type=int, default=1, help="Galvatron profiling batch size step")
    group.add_argument(
        "--profile_seq_length_list", type=str, default=None, help="Galvatron profiling sequence length step"
    )
    group.add_argument(
        "--profile_min_seq_length", type=int, default=None, help="Galvatron profiling max sequence length"
    )
    group.add_argument(
        "--profile_max_seq_length", type=int, default=None, help="Galvatron profiling max sequence length"
    )
    group.add_argument(
        "--profile_seq_length_step", type=int, default=128, help="Galvatron profiling sequence length step"
    )
    group.add_argument("--layernum_min", type=int, default=1, help="Layernum min for profiling.")
    group.add_argument("--layernum_max", type=int, default=2, help="Layernum min for profiling.")
    group.add_argument("--max_tp_deg", type=int, default=8, help="Maximum tensor parallel degree to profile.")
    group.add_argument(
        "--profile_dp_type", type=str, default="zero3", help="Use zero3 or ddp to profile.", choices=["zero3", "ddp"]
    )
    group.add_argument("--use-flash-attn", action="store_true", help="Use FlashAttention implementation of attention.")
    group.add_argument("--extra_args_str", type=str, default="", help="Extra arguments for megatron initilization.")

    group.add_argument(
        "--sequence_parallel",
        action="store_true",
        help="Whether to use sequence parallel",
    )

    group.add_argument(
        "--make-vocab-size-divisible-by",
        type=int,
        default=128,
        help="Pad the vocab size to be divisible by this value." "This is added for computational efficieny reasons.",
    )

    return parser
