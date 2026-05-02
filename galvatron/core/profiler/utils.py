import os

import torch

from galvatron.utils.config_utils import num2str, read_json_config, write_json_config


def print_peak_memory(prefix, device, type="allocated"):
    if type == "allocated":
        print(prefix, "[Allocated]")
        max_mem = torch.cuda.max_memory_allocated(device) / 2**20
        cur_mem = torch.cuda.memory_allocated(device) / 2**20
        print("\tMax memory: %.2f MB\tCurrent memory : %.2f MB" % (max_mem, cur_mem))
    elif type == "reserved":
        print(prefix, "[Reserved]")
        max_mem = torch.cuda.max_memory_reserved(device) / 2**20
        cur_mem = torch.cuda.memory_reserved(device) / 2**20
        print("\tMax memory: %.2f MB\tCurrent memory : %.2f MB" % (max_mem, cur_mem))
    return max_mem, cur_mem


def _is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _is_rank0():
    return (not _is_distributed()) or torch.distributed.get_rank() == 0


def _allreduce_max(value):
    """Collapse a scalar across ranks using MAX to get the conservative result.
    Fixes the multi-rank read-modify-write race on the shared JSON file.
    """
    if not _is_distributed():
        return value
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return t.item()


def save_profiled_memory(
    path,
    pp_deg,
    tp_deg,
    world_size,
    layer_num,
    bsz,
    rank,
    model_states,
    activation,
    activation_peak,
    cpt,
    sequence_parallel=False,
    vocab_tp=1,
    seq=None,
    profile_unit="all",
):
    # All ranks participate in the reduction; only rank 0 writes the file.
    # Keys retain the rank index for backward compatibility with the memory
    # aggregator, but every rank's key stores the same (conservative MAX)
    # value since ranks have been collapsed via all_reduce.
    model_states = _allreduce_max(model_states)
    activation = _allreduce_max(activation)
    activation_peak = _allreduce_max(activation_peak)
    if not _is_rank0():
        return
    config = read_json_config(path) if os.path.exists(path) else {}
    key = "%d_%d_%d" % (pp_deg, tp_deg, world_size // pp_deg // tp_deg)
    if cpt:
        key += "_c"
    if vocab_tp == tp_deg and tp_deg != 1:
        key += "_vtp"
    if sequence_parallel:
        key += "_sp"
    if key not in config.keys():
        config[key] = {}
    layernum_info = num2str(layer_num, "layernum")
    seq_info = num2str(seq, "seq")
    # Write both first-stage (rank 0) and last-stage (rank world_size-1) keys
    # so the PP-aware aggregator in _process_memory_data finds both indices.
    ranks_to_write = {0, max(0, world_size - 1)}
    for r in ranks_to_write:
        if profile_unit == "all":
            config[key][
                "%s_bsz%d_%s_rank%d_ms" % (layernum_info, bsz, seq_info, r)
            ] = model_states
            config[key][
                "%s_bsz%d_%s_rank%d_act" % (layernum_info, bsz, seq_info, r)
            ] = activation
            config[key][
                "%s_bsz%d_%s_rank%d_act_peak" % (layernum_info, bsz, seq_info, r)
            ] = activation_peak
        else:
            config[key][
                "%s_bsz%d_%s_%s_rank%d_ms"
                % (layernum_info, bsz, seq_info, profile_unit, r)
            ] = model_states
            config[key][
                "%s_bsz%d_%s_%s_rank%d_act"
                % (layernum_info, bsz, seq_info, profile_unit, r)
            ] = activation
            config[key][
                "%s_bsz%d_%s_%s_rank%d_act_peak"
                % (layernum_info, bsz, seq_info, profile_unit, r)
            ] = activation_peak
    write_json_config(config, path)
    print("Already written profiled memory into config file %s!\n" % (path))


def save_profiled_time(path, time, bsz, layer_num, seq, profile_unit):
    time = _allreduce_max(time)
    if not _is_rank0():
        return
    config = read_json_config(path) if os.path.exists(path) else {}
    layernum_info = num2str(layer_num, "layernum")
    seq_info = num2str(seq, "seq")
    key = "%s_bsz%d_%s" % (layernum_info, bsz, seq_info)
    if profile_unit != "all":
        key += "_%s" % (profile_unit)
    config[key] = time
    write_json_config(config, path)
    print("Already written profiled time into config file %s!\n" % (path))
