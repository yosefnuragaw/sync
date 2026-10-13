from typing import Dict, Optional
from collections import deque
import torch

try:  # typing.override is 3.12+
    from typing import override
except ImportError:  # pragma: no cover - older interpreters
    def override(func):
        return func


MODEL_TEMPLATE_MAP: Dict[str, str] = {
    'meta-llama/Llama-2-7b-chat-hf': 'llama-3',
    'mistralai/Mistral-7B-Instruct-v0.2': 'mistral',
}

VALID_APPLY_TYPES = {'billy', 'sync', 'simultaneous', 'sync-static'}

# Apply types that run the sensitivity gate: per-layer cosine levels, de-biased per
# persona, standardised across personas.
LEVEL_APPLY_TYPES = {'sync'}



class GateTrace:
    """Opt-in recorder for what the gate emits, per layer per step.

        from models import wrapper
        wrapper.GATE_TRACE = wrapper.GateTrace(vec_names=[...])
        ...generate...
        wrapper.GATE_TRACE.to_csv("trace.csv")

    Columns: layer, step, warmup, then tau_<name>, gamma_<name> and level_<name> per
    behaviour, averaged over the batch; keep_batch=True keeps each item its own row.
    """

    def __init__(self, vec_names=None, keep_batch: bool = False, max_rows: int = 2_000_000):
        self.vec_names = vec_names
        self.keep_batch = keep_batch
        self.max_rows = max_rows
        self.rows = []
        self._step = {}

    def record(self, layer_idx: int, gamma, tau, warmup: bool, level=None) -> None:
        """gamma (B, N, 1) or (B, N); tau (B, N) or None during warm-up; level (B, N)."""
        if len(self.rows) >= self.max_rows:
            return
        step = self._step.get(layer_idx, 0)
        self._step[layer_idx] = step + 1
        g = gamma.detach().float().reshape(gamma.shape[0], -1).cpu()
        t = None if tau is None else tau.detach().float().cpu()
        lv = None if level is None else level.detach().float().reshape(g.shape[0], -1).cpu()
        names = self.vec_names or [f"v{i}" for i in range(g.shape[1])]
        batch = range(g.shape[0]) if self.keep_batch else [None]
        for bi in batch:
            row = {"layer": layer_idx, "step": step, "warmup": int(warmup)}
            if self.keep_batch:
                row["batch"] = bi
            for i, n in enumerate(names):
                row[f"gamma_{n}"] = float(g[bi, i]) if self.keep_batch else float(g[:, i].mean())
                if t is not None:
                    row[f"tau_{n}"] = float(t[bi, i]) if self.keep_batch else float(t[:, i].mean())
                if lv is not None:
                    row[f"level_{n}"] = float(lv[bi, i]) if self.keep_batch else float(lv[:, i].mean())
            self.rows.append(row)

    def reset(self) -> None:
        self.rows.clear()
        self._step.clear()

    def to_csv(self, path: str) -> str:
        import csv
        if not self.rows:
            raise RuntimeError("GateTrace is empty -- was an apply_type with a gate used?")
        # Warm-up rows carry no tau_ columns, so the header is the union over all rows.
        keys, seen = [], set()
        for r in self.rows:
            for k in r:
                if k not in seen:
                    seen.add(k); keys.append(k)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(self.rows)
        return path


# None disables tracing entirely; set it to a GateTrace to start recording.
GATE_TRACE: Optional[GateTrace] = None

# Apply types that place personas under a cross-layer budget. A budget constrains layers
# against each other, so the whole (persona x layer) level matrix is needed before any
# layer can decide. Layers run 0 -> 31 within a forward pass, so the matrix completes one
# step late: the assignment used at step t is built from the board as of step t-1.
BUDGET_APPLY_TYPES = {'sync'}

# Apply types that use 'simultaneous's offline assignment during decode and a dense
# all-layer pass during prefill.
BP_APPLY_TYPES = {'sync-static'}

# Apply types whose prompt pass is ramped rather than flat. Membership here is the only
# thing that decides it.
WARM_PREFILL_APPLY_TYPES = {'sync-static', 'sync'}

# Apply types that keep the de-bias offset per prompt and per batch item: cleared at every
# prefill and seeded from that prompt's own tokens, so no prompt's offset reaches another.
PERPROMPT_BASELINE_APPLY_TYPES = {'sync'}

LEVEL_BOARD: Dict[int, torch.Tensor] = {}        # layer_idx -> (B, N) de-biased level margin
BUDGET_ASSIGNMENT: Dict[int, torch.Tensor] = {}  # layer_idx -> (B,) persona index, -1 = idle
BUDGET_LAYERS: set = set()                       # layers running this apply_type
BUDGET_K: int = 1                                # layers each persona may claim
BUDGET_STATS = {"steps": 0, "placed": 0, "dropped": 0}


def reset_budget_state() -> None:
    """Clear the board between prompts. Stale rows would place on last prompt's levels."""
    LEVEL_BOARD.clear()
    BUDGET_ASSIGNMENT.clear()


def _rebuild_budget_assignment() -> None:
    """Place each behaviour on its best free layer(s) from the lagged board.

    Same rule as ablation_layer.assign_layers, so only the scores differ between the two:
    candidates are (score, behaviour, layer) for every pair, consumed best-score-first
    across all behaviours at once. Ties go to the lower behaviour index (its position in
    vec_dir), then the lower layer. A behaviour that loses a layer falls through to its
    next-best free one rather than being dropped, so the realised layer count is always
    budget x n_behaviours while layers remain.
    """
    if not LEVEL_BOARD:
        return
    layers = sorted(LEVEL_BOARD)
    M = torch.stack([LEVEL_BOARD[l] for l in layers], dim=-1)   # (B, N, L)
    B, N, L = M.shape
    assign = torch.full((B, len(layers)), -1, dtype=torch.long)
    for b in range(B):
        mb = M[b]
        cands = [(-float(mb[p, j]), p, j) for p in range(N) for j in range(L)]
        cands.sort()                       # score desc, behaviour asc, layer asc
        taken: Dict[int, int] = {}
        count = [0] * N
        for _, p, j in cands:
            if count[p] >= BUDGET_K or j in taken:
                continue
            taken[j] = p
            count[p] += 1
        for p in range(N):
            BUDGET_STATS["placed" if count[p] else "dropped"] += 1
        for j, p in taken.items():
            assign[b, j] = p
    BUDGET_STATS["steps"] += 1
    for i, l in enumerate(layers):
        BUDGET_ASSIGNMENT[l] = assign[:, i]


class CAABlockWrapper(torch.nn.Module):
    def __init__(self, block, hidden_dim, vec: Optional[torch.Tensor] = None, apply_type: str = 'billy', K:int = 16, l:int = 2, b:int = 2, temp: float = 1.0, layer_idx: int = 0, dir_idx: Optional[int] = None, level_baseline: Optional[torch.Tensor] = None, level_eps: float = 1e-3, rho: float = 1.0,
                 warmup_reverse: bool = False ):
        super().__init__()
        # (N,) mean cos(h, v_p) for this layer. A tensor is a fixed calibration; the
        # string "running" estimates it online instead. None disables de-biasing.
        self.level_baseline = level_baseline
        self.level_eps = level_eps
        self._lvl_sum = None       # running-mean accumulator, persists across prompts
        self._lvl_n = 0
        self.multiplier = 1.0
        self.block = block
        self.is_extract = False
        self.drift_history: list[dict] = []
        self.vec_names: list[str] = []
        
        self.K = K
        self.l = l
        self.b = b
        self.temp = temp          # inert: no apply_type takes a softmax
        self.layer_idx = layer_idx
        self.dir_idx = dir_idx    # persona index this layer injects, used by 'simultaneous' only
        self.steer = True         # False on a wrapped-but-unsteered layer, see set_steer()

        try:
            ref_param = next(block.parameters())
            self.init_dtype = ref_param.dtype
        except StopIteration:
            self.init_dtype = torch.float32

        if vec is not None:
            self.vec = vec.to(self.init_dtype)
        else:
            self.vec = torch.zeros(hidden_dim, dtype=self.init_dtype)

        self.caa_buffer = {'pos': [], 'neg': []}

        if apply_type not in VALID_APPLY_TYPES:
            raise ValueError(f"apply_type must be one of {sorted(VALID_APPLY_TYPES)}, got {apply_type!r}")
        self.apply_type = apply_type
        if apply_type in BUDGET_APPLY_TYPES:
            BUDGET_LAYERS.add(layer_idx)
        # Shape of the prefill ramp, r_t = ((t + 1) / T) ** rho. rho = 0 makes the ramp
        # identically 1 in both directions, i.e. a flat full-magnitude prompt pass.
        if rho < 0:
            raise ValueError(f"rho must be >= 0, got {rho!r}")
        self.rho = float(rho)
        # Direction of the prefill ramp.
        #   False  r_t = ((t+1)/T)^rho   rises to 1 at the last prompt position
        #   True   r_t = ((T-t)/T)^rho   starts at 1 and decays to 1/T at the last
        self.warmup_reverse = bool(warmup_reverse)

    def set_vec_names(self, names: list[str]) -> None:
        self.vec_names = names
    
    def extract(self, status: bool):
        self.is_extract = status

    def set_multiplier(self, mul: float) -> None:
        self.multiplier = mul

    @staticmethod
    def _prompt_valid_mask(kwargs, B: int, T: int):
        """(B, T) bool over real prompt positions, or None if no usable mask was passed.

        Generation pads LEFT, so a seed taken over the raw prefill tensor would average
        pad activations into the baseline, by an amount the longest prompt in the batch
        decides.
        """
        am = kwargs.get("attention_mask")
        if am is None:
            return None
        if am.dim() == 2 and am.shape[0] == B and am.shape[1] == T:
            return am.bool()
        # 4-D additive causal mask: the last query row says which keys are visible.
        if am.dim() == 4 and am.shape[0] == B and am.shape[-1] == T:
            return am[:, 0, -1, :] > torch.finfo(am.dtype).min / 2
        return None

    def _level_debias(self, lvl: torch.Tensor) -> torch.Tensor:
        """Remove each behaviour's own cosine offset from the levels.

        A tensor baseline is a fixed calibration. "running" estimates it online from
        every step seen so far, which needs no calibration file but starts cold, and on
        PERPROMPT_BASELINE_APPLY_TYPES is reset and reseeded at every prefill instead.
        """
        if self.level_baseline is None:
            return lvl
        if isinstance(self.level_baseline, str):
            if self.level_baseline != "running":
                raise ValueError(
                    f"level_baseline must be a tensor or 'running', got "
                    f"{self.level_baseline!r}")
            if self.apply_type in PERPROMPT_BASELINE_APPLY_TYPES:
                # One estimate per batch item, (B, N), seeded at prefill from that
                # item's own prompt. Shapes already align, so no unsqueeze.
                if self._lvl_sum is None:                 # a stray call before prefill
                    self._lvl_sum = torch.zeros_like(lvl.detach().float())
                    self._lvl_n = torch.zeros(lvl.shape[0], 1, device=lvl.device,
                                              dtype=torch.float32)
                self._lvl_sum = self._lvl_sum + lvl.detach().float()
                # (B, 1) here: masking leaves each item its own count.
                self._lvl_n = self._lvl_n + 1
                return lvl - (self._lvl_sum / self._lvl_n).to(lvl.dtype)
            batch = lvl.detach().float().mean(dim=0)              # (N,)
            if self._lvl_sum is None:
                self._lvl_sum = torch.zeros_like(batch)
            self._lvl_sum += batch
            self._lvl_n += 1
            return lvl - (self._lvl_sum / self._lvl_n).to(lvl.dtype).unsqueeze(0)
        return lvl - self.level_baseline.to(device=lvl.device, dtype=lvl.dtype).unsqueeze(0)

    def set_level_baseline(self, baseline: Optional[torch.Tensor]) -> None:
        """(N,) per-behaviour mean cosine for this layer; see LEVEL_APPLY_TYPES."""
        self.level_baseline = baseline

    def set_vec(self, vec) -> None:
        self.vec = vec

    def set_dir_idx(self, dir_idx: Optional[int]) -> None:
        self.dir_idx = dir_idx

    def set_steer(self, steer: bool) -> None:
        """Turn steering on or off for this layer, leaving the block itself wrapped.

        Every block is wrapped so extraction and the sweep can address any of them, but
        only the layers in the config's `layer:` list are handed an (N, D) vector stack.
        """
        self.steer = steer

    def record(self, activation):
        if self.is_extract:
            if self.multiplier > 0:
                self.caa_buffer['pos'].append(activation)
            elif self.multiplier < 0:
                self.caa_buffer['neg'].append(activation)

    def extract_vec(self, clear: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.caa_buffer['pos'] and not self.caa_buffer['neg']:
            raise ValueError("Buffer is empty, run extraction first.")
        
        pos_stacked = torch.stack(self.caa_buffer['pos'])
        neg_stacked = torch.stack(self.caa_buffer['neg'])

        if pos_stacked.dim() > 2:
            vec_pos = pos_stacked.mean(dim=(0, 1))  # [D]
            vec_neg = neg_stacked.mean(dim=(0, 1))  # [D]
        else:
            vec_pos = pos_stacked.mean(dim=0)       # [D]
            vec_neg = neg_stacked.mean(dim=0)       # [D]
        
        if clear:
            self.caa_buffer = {'pos': [], 'neg': []}
        
        sv_pos = vec_pos - vec_neg
        sv_neg = vec_neg - vec_pos
        
        return sv_pos, sv_neg
        
    def forward(self, hidden_states, *args, **kwargs):
        output = self.block(hidden_states, *args, **kwargs)
        out_tensor = output[0] if isinstance(output, tuple) else output
        avg_output = out_tensor.detach().mean(dim=1)  # [B, D]
        
        if self.is_extract:
            batch_avg = avg_output.mean(dim=0)
            self.record(batch_avg.cpu())
        else:
            # Wrapped but unsteered: self.vec is still the constructor's placeholder,
            # not an (N, D) stack, so every branch below would misread it.
            if not self.steer:
                return output

            # 'simultaneous' pins one persona per layer, so an unclaimed layer is idle.
            if self.apply_type == 'simultaneous' and self.dir_idx is None:
                return output

            # Same, but only during decode: prefill injects at every layer. Throughout
            # this file out_tensor.shape[1] > 1 is the prefill test.
            if (self.apply_type in BP_APPLY_TYPES and self.dir_idx is None
                    and out_tensor.shape[1] == 1):
                return output


            multiplier = torch.tensor(self.multiplier, dtype=out_tensor.dtype, device=out_tensor.device) 
            mask = torch.tensor(1, dtype=out_tensor.dtype, device=out_tensor.device)
            B = out_tensor.shape[0]
            N = self.vec.shape[0]

            # Exact membership, never a substring test: 'sync-static' must not reach the
            # level machinery.
            if self.apply_type in LEVEL_APPLY_TYPES:
                last_token = out_tensor.detach()[:, -1, :]  # (B, D)
                T = out_tensor.shape[1]
                is_prefill = T > 1

                def compute_cos_sims(tokens):
                    vec     = self.vec.to(out_tensor.device)                         # (N, D)
                    cv_norm = vec.norm(dim=-1).clamp(min=1e-8)                       # (N,)
                    t_norm  = tokens.norm(dim=-1).clamp(min=1e-8)                    # (B, S)
                    dot     = torch.einsum('bsd,nd->bns', tokens, vec)               # (B, N, S)
                    denom   = t_norm.unsqueeze(1) * cv_norm.view(1, N, 1)            # (B, N, S)
                    return (dot / denom).clamp(-1.0, 1.0)                            # (B, N, S)

                level_now = None                 # cos(h_t, v_p) at this step, for the trace

                if is_prefill:
                    # Clearing on the lowest steered layer fires once per prompt.
                    if (self.apply_type in BUDGET_APPLY_TYPES
                            and BUDGET_LAYERS and self.layer_idx == min(BUDGET_LAYERS)):
                        reset_budget_state()
                    prompt_tokens = out_tensor.detach()[:, -self.K:-1, :]          # (B, S, D)
                    all_tokens    = torch.cat([prompt_tokens, last_token.unsqueeze(1)], dim=1)  # (B, S+1, D)
                    cos_sims_init = compute_cos_sims(all_tokens)                             # (B, N, S+1)
                    level_now     = cos_sims_init[:, :, -1]                                  # (B, N)

                    if self.apply_type in PERPROMPT_BASELINE_APPLY_TYPES:
                        # Reseed from every prompt position, overwriting whatever the
                        # previous prompt left. Each position counts as one sample, so a
                        # decode step added later weighs 1/(T+k).
                        lvl_dev = out_tensor.device
                        full = compute_cos_sims(out_tensor.detach())          # (B, N, T)
                        vm = self._prompt_valid_mask(kwargs, B, full.shape[-1])
                        if vm is None:
                            # No usable mask: fall back to the window's own slice, which
                            # is the prompt tail and so real tokens under left padding.
                            if not getattr(type(self), '_pp_mask_warned', False):
                                print("[SYNC-20] no attention_mask on the prefill call; "
                                      "seeding the per-prompt baseline from the last "
                                      f"{self.K} prompt positions instead of "
                                      "position 0.")
                                type(self)._pp_mask_warned = True
                            seed = cos_sims_init                              # (B, N, S+1)
                            self._lvl_sum = seed.sum(dim=-1).detach().float()
                            self._lvl_n = torch.full((B, 1), float(seed.shape[-1]),
                                                     device=lvl_dev, dtype=torch.float32)
                        else:
                            m = vm.to(full.device).unsqueeze(1).to(full.dtype)  # (B, 1, T)
                            self._lvl_sum = (full * m).sum(dim=-1).detach().float()
                            self._lvl_n = m.sum(dim=-1).clamp(min=1.0).detach().float()

                    # one deque per batch item, each entry is (N,)
                    self._cos_history = [
                        deque(
                            [cos_sims_init[b, :, t] for t in range(cos_sims_init.shape[2])],
                            maxlen=self.K
                        )
                        for b in range(B)
                    ]

                    sensitivity_weight = torch.ones(B, N, 1, dtype=out_tensor.dtype, device=out_tensor.device)

                else:
                    # Decode phase
                    if getattr(self, '_cos_history', None) is None or len(self._cos_history) != B:
                        self._cos_history = [deque(maxlen=self.K) for _ in range(B)]

                    # compute cos sim only for the new token, append (N,) per batch item
                    new_cos = compute_cos_sims(last_token.unsqueeze(1))  # (B, N, 1)
                    level_now = new_cos[:, :, 0]                         # (B, N)
                    for b in range(B):
                        self._cos_history[b].append(new_cos[b, :, 0])   # (N,)

                    min_history = min(len(self._cos_history[b]) for b in range(B))

                    tau_tensor = None            # stays None through the warm-up

                    if min_history < self.K:
                        sensitivity_weight = torch.ones(B, N, 1, dtype=out_tensor.dtype, device=out_tensor.device)

                    else:
                        cos_sims = torch.stack([
                            torch.stack(list(self._cos_history[b])[-min_history:], dim=1)  # (N, H)
                            for b in range(B)
                        ], dim=0)  # (B, N, H)

                        if self.apply_type in LEVEL_APPLY_TYPES:
                            #   c_p = mean_t cos(h_t, v_p) - baseline_p
                            #   z   = (c - mean_q c) / (std_q c + eps)
                            lvl = cos_sims.mean(dim=-1)                                   # (B, N)
                            lvl = self._level_debias(lvl)
                            z = lvl.float()
                            z = (z - z.mean(dim=-1, keepdim=True)) / (
                                z.std(dim=-1, keepdim=True) + self.level_eps)
                            if self.apply_type in BUDGET_APPLY_TYPES:
                                # The board carries a MARGIN in cosine units, not z:
                                #
                                #   margin_p = lvl_p - mean_{q!=p} lvl_q
                                #
                                # lvl is already de-biased and so comparable across
                                # layers, which is what a cross-layer budget needs. z
                                # divides by the within-layer std, which would let the
                                # layer that discriminates least win the argmax.
                                margin = lvl.float()
                                margin = margin - (margin.sum(dim=-1, keepdim=True)
                                                   - margin) / max(N - 1, 1)
                                # The rebuild fires on the lowest steered layer, where the
                                # board is complete for the previous step.
                                if self.layer_idx == min(BUDGET_LAYERS):
                                    _rebuild_budget_assignment()
                                LEVEL_BOARD[self.layer_idx] = margin.detach().cpu()

                                who = BUDGET_ASSIGNMENT.get(self.layer_idx)
                                if who is None:
                                    # First gated step after prefill: the board is still
                                    # empty, so fall back to the uniform blend.
                                    gamma = torch.ones_like(z)
                                else:
                                    who = who.to(z.device)
                                    gamma = torch.zeros_like(z)
                                    hit = who >= 0
                                    if hit.any():
                                        gamma[hit] = gamma[hit].scatter(
                                            -1, who[hit].unsqueeze(-1), 1.0)
                                sensitivity_weight = gamma.to(out_tensor.dtype).unsqueeze(-1)
                            sensitivity_weight = gamma.to(out_tensor.dtype).unsqueeze(-1)  # (B, N, 1)
                            tau_tensor = lvl                    # what the trace records

                mask = mask * sensitivity_weight.to(dtype=out_tensor.dtype, device=out_tensor.device)  # (B, N, 1)

                if GATE_TRACE is not None:
                    GATE_TRACE.record(self.layer_idx, sensitivity_weight,
                                      None if is_prefill else tau_tensor,
                                      warmup=is_prefill or tau_tensor is None,
                                      level=level_now)

            # Baseline case
            if mask.dim() < 3:
                B = out_tensor.shape[0]
                N = self.vec.shape[0]
                mask = mask.view(1, 1, 1).expand(B, N, 1)

            weighted_sum = (mask.to(device=output[0].device, dtype=output[0].dtype) * self.vec.unsqueeze(0).to(device=output[0].device, dtype=output[0].dtype)).sum(dim=1)  # (B, D)
            mask_sum = mask.sum(dim=1)  # (B, 1)

            if self.apply_type in BUDGET_APPLY_TYPES:
                # host layer  one-hot mask, mask_sum = 1 -> v_p at its own norm
                # idle layer  zero mask, mask_sum = 0 -> nothing injects (clamp keeps
                #             the division finite)
                # prefill     all-ones mask, mask_sum = N -> mean_p v_p
                vec = (weighted_sum / mask_sum.clamp(min=1.0)) * multiplier          # (B, D)

            elif self.apply_type == 'billy':
                vec = (weighted_sum / mask_sum) * multiplier

            elif self.apply_type in BP_APPLY_TYPES:
                # prefill  all-ones mask, so weighted_sum / mask_sum is mean_p v_p --
                #          the 'billy' branch below, not norm-pinned
                # decode   the 'simultaneous' rule
                if out_tensor.shape[1] > 1:
                    vec = (weighted_sum / mask_sum) * multiplier                          # (B, D)
                else:
                    if not 0 <= (self.dir_idx if self.dir_idx is not None else -1) < N:
                        raise IndexError(
                            f"layer {self.layer_idx}: dir_idx {self.dir_idx} is out of range "
                            f"for {N} steering vector(s)")
                    chosen = self.vec[self.dir_idx].to(device=output[0].device,
                                                       dtype=output[0].dtype)             # (D,)
                    vec = (chosen * multiplier).unsqueeze(0).expand(out_tensor.shape[0], -1)

            elif self.apply_type == 'simultaneous':
                # dir_idx is the persona this layer was assigned by the offline sweep.
                if not 0 <= self.dir_idx < N:
                    raise IndexError(
                        f"layer {self.layer_idx}: dir_idx {self.dir_idx} is out of range for "
                        f"{N} steering vector(s)"
                    )
                chosen = self.vec[self.dir_idx].to(device=output[0].device, dtype=output[0].dtype)  # (D,)
                vec = (chosen * multiplier).unsqueeze(0).expand(out_tensor.shape[0], -1)            # (B, D)

            # Ramp the prompt pass:
            #
            #     r_t = ((t + 1) / T) ** rho,   t = 0 .. T-1
            #     h_t <- h_t + r_t * vec
            #
            # r_{T-1} = 1 exactly, so the last input token carries full magnitude and
            # decode continues from there. Decode is T = 1 and is never ramped. This
            # changes total prompt displacement as well as its shape: at rho = 1 the mean
            # ramp is (T+1)/2T.
            ramp = None
            if self.apply_type in WARM_PREFILL_APPLY_TYPES and out_tensor.shape[1] > 1:
                T_ramp = out_tensor.shape[1]
                steps = torch.arange(1, T_ramp + 1, device=out_tensor.device,
                                     dtype=torch.float32) / T_ramp
                if self.warmup_reverse:
                    steps = steps.flip(0)        # 1 at the first position, 1/T at the last
                ramp = steps.pow(self.rho).view(1, T_ramp, 1)

            if isinstance(output, tuple):
                delta = vec.unsqueeze(1).to(output[0].device)                        # (B, 1, D)
                if ramp is not None:
                    delta = delta * ramp.to(device=output[0].device, dtype=delta.dtype)
                modified_hidden = output[0] + delta
                output = (modified_hidden,) + output[1:]
            elif isinstance(output, torch.Tensor):
                delta = vec.unsqueeze(1).to(output.device)
                if ramp is not None:
                    delta = delta * ramp.to(device=output.device, dtype=delta.dtype)
                output = output + delta

        return output
