def galvatron_common_model_args(parser):
    group = parser.add_argument_group(title="Galvatron Common Model Arguments")

    group.add_argument(
        "--profile_mode",
        type=str,
        default="static",
        help="Galvatron profiling mode",
        choices=["static", "batch", "sequence"],
    )
    group.add_argument(
        "--set_model_config_manually",
        type=int,
        default=0,
        help="Whether to set model config manually. If set to 1, model config set by 'model_size' will be overwritten.",
    )
    group.add_argument(
        "--set_seqlen_manually",
        type=int,
        default=0,
        help="Whether to set sequence length config manually (doesn't overwrite other model configs).",
    )
    group.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        help="Mixed precision option.",
        choices=["fp32", "fp16", "bf16"],
    )
    group.add_argument(
        "--shape_order",
        type=str,
        default="SBH",
        help="Model shape order.",
        choices=["SBH", "BSH"],
    )

    group.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout rate.")

    # Deterministic-input / solver-freeze knobs. Shared between training,
    # evaluation, and profiling so performance numbers stay reproducible.
    group.add_argument(
        "--static_input",
        action="store_true",
        help="Reuse a single deterministic batch across all iterations. "
        "The batch is built once on rank 0 and broadcast across the DP group. "
        "Combine with --dropout_prob 0 to keep routing bit-identical across iters.",
    )
    group.add_argument(
        "--static_input_path",
        type=str,
        default="",
        help="Optional path to a .pt file holding the frozen synthetic batch. "
        "If the file exists it is loaded (all ranks); otherwise the batch is "
        "built from --seed/--vocab_size and saved to this path by rank 0. "
        "Only meaningful when --static_input is set.",
    )
    group.add_argument(
        "--use_fsep",
        action="store_true",
        help="Enable fused-sequence expert parallelism / smart routing dispatcher. "
        "Required for the LAER async LP solver to be invoked (ENABLE_SOLVER=1 alone "
        "is not enough).",
    )
    group.add_argument(
        "--global_ep_deg",
        type=int,
        default=1,
        help="Experts parallel degree.",
    )
    group.add_argument(
        "--global_tp_of_ep_deg",
        type=int,
        default=1,
        help="Tensor parallel degree of experts.",
    )
    group.add_argument(
        "--expert_capacity_per_device",
        type=int,
        default=1,
        help="Expert capacity per device. FSEP requires "
        "global_ep_deg * expert_capacity_per_device >= num_global_experts.",
    )
    group.add_argument(
        "--laer_freeze_after_iter",
        type=int,
        default=-1,
        help="Freeze the LAER expert layout once each dispatcher has submitted this many "
        "solver iterations. -1 disables freezing (default). Only meaningful when "
        "ENABLE_SOLVER=1.",
    )
    return parser
