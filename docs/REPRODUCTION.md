# Reproduction contract: official SDAR, ALFWorld, four A100 GPUs

## Algorithm identity

Source: https://github.com/ZJU-REAL/SDAR at
`29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
Reference launcher: `examples/sdar_trainer/run_alfworld_3b.sh`.

This is **Self-Distilled Agentic Reinforcement Learning (SDAR)**, not the
ungated GRPO+OPSD comparison, RLSD advantage reweighting, RetireOPD, or a
RETHINK variant. Both sizes start from the pinned Qwen2.5 Instruct base model,
without SFT. A frozen base reference provides the separate reference KL.

The Student samples complete ALFWorld episodes. GRPO uses native environment
rewards and group size eight. The Teacher is a no-gradient scoring view of the
**current Actor**, conditioned on retrieved static ALFWorld skills. It scores
the same Student responses and prefixes, rather than generating replacement
actions. Skills are not given to the Student during training or evaluation.

\[
\Delta_t=\log\pi_\theta(y_t\mid H_t,\mathrm{skill})
 -\operatorname{sg}[\log\pi_\theta(y_t\mid H_t)],\qquad
g_t=\operatorname{sg}[\sigma(5\Delta_t)].
\]

\[
\mathcal L_{\rm SDAR}
=\operatorname{masked\_mean}_t
\left[g_t\big(\operatorname{sg}[\log\pi_T(y_t)]-\log\pi_\theta(y_t)\big)\right].
\]

The native Actor jointly optimizes the GRPO objective and 0.01 times this
auxiliary loss, plus its original reference KL (coefficient 0.01), entropy
term (0.001), and invalid-action penalty (0.1). The sigmoid gate is detached
and nonnegative. A negative Teacher gap attenuates the auxiliary gradient;
it does not reverse it. Native masks and aggregation are unchanged.

## Scientific settings

| Setting | Both model sizes |
| --- | --- |
| Backbone | Qwen2.5-1.5B-Instruct or Qwen2.5-3B-Instruct; fixed public revisions |
| Data | Native ALFWorld train pool; no private trajectories or SFT dataset |
| Updates | 150 per size |
| Per update | 16 task groups, eight episodes per group: 128 episodes globally |
| Learning rate | 1e-6 |
| PPO mini/micro batch | Global decision-row mini-batch 256, per-GPU micro-batch 32 |
| PPO epochs / aggregation | One / token-mean |
| SDAR | Coefficient 0.01, beta 5.0, skill_all=false |
| Interaction | 50 steps, history two, prompt 2048, response 512 |
| Training sampling | Temperature 1, top_p 1, top_k -1 |
| Validation sampling | Temperature 0.4, one rollout per task |
| Distribution | Four FSDP ranks; two rollout replicas with TP=2 |
| vLLM | Memory utilization 0.6, eager=false, free_cache_engine=false |

Increasing GPU count does not increase the episode batch. Native decision-row
batch adjustment is retained. Do not add an outer torchrun to this Ray launcher.

This standalone release runs only the backbone selected in its README/shell
entrypoints. Relative to the eight-GPU release e8d323f, the experiment JSON
changes only trainer.n_gpus_per_node from eight to four. Resource validation,
Ray GPU allocation, checkpoint rank-file checks and default GPU IDs also use
four; the default CPU budget is 32. With configured global mini-batch256 and
micro32/GPU, native gradient accumulation handles local64 rows per rank instead
of local32 on eight ranks. Keep the native decision-row adjustment and loss
aggregation; this is not a promise of bitwise eight-vs-four numerical identity.

## Necessary adaptations and what is not changed

1. Portable, pinned source/model downloads and project-relative resource paths.
2. A shared configuration for the additional public 1.5B backbone. No layers,
   tokenizer tokens, architecture, or loss functions are added.
3. Fixed seen140 validation at update 0 and every ten updates, rather than the
   official launcher's 128 validation entries and five-update frequency.
   Task paths, game-file hashes, and seeds are included in the repository.
   This is a controlled comparison protocol, not a claim of identical paper scores.
4. Every 16 independent native text environments share a Ray process. Each
   environment retains its own seed/state; order and transitions are tested.
   This is resource scheduling, not a change to observations/actions/rewards.
5. Local JSON metrics, task-level results, and checkpoint/export bookkeeping.
   Saves occur every 20 updates and at 150; native max_keep is set to two.
   This split does not replace upstream's checkpoint-rotation implementation.

The native trainer's `fit`, Actor optimization, GRPO advantages, SDAR loss,
Teacher construction, skill retrieval, observation prompts, action parser,
and environment rewards remain unchanged. Their hashes are locked in
`configs/upstream.json`. The full limited adapter is `patches/runtime.patch`.

## Clean-machine execution

Follow README section 2 in a new Linux directory. Do not copy any author's
environment, local cache, drive, login files, or existing experiment output.
The repository obtains all required source, model, skills and environment
resources through the documented public downloads. Downloading weights is
separate from CPU source/configuration verification.

Use the repository's constraints when installing upstream dependencies, and
run `python -m pip check`. FlashAttention requires a matching wheel or a
working CUDA Toolkit compiler. Before spending GPU time:

```bash
python src/setup.py --verify
python scripts/check_release.py
python -m pytest tests -q
bash scripts/train.sh --dry-run
```

Then check GPU ownership and available host RAM/disk. Use all four allocated
80GB GPUs via `bash scripts/train.sh` in a persistent session. The compatibility
queue `scripts/run_both.sh` also runs only this repository's one selected size.
Its first initial validation reports the untrained base performance.

## First real GPU iteration acceptance

- Confirm four FSDP ranks, TP=2 rollout, and this repository's intended backbone.
- Verify actual environment reset, generated action, execution and feedback.
- Confirm finite rewards/advantages, Teacher log probabilities, Teacher gap
  and gate metrics, finite Actor gradient norm, and an actual parameter update.
- At the first save, verify all four model/optimizer/extra-state rank files
  plus Actor config and dataloader state. Missing ranks are not a resumable save.
- At update150, export with the official model merger and preserve its hashes.

CPU CI verifies configuration/source contracts and a real tensor-level loss
gradient test. It does **not** certify NCCL, CUDA memory, vLLM initialization,
full-scale RL training, or any benchmark score. Those require the above GPU run.

## Evaluation and reporting

Use exactly `configs/seen140_tasks.json` for both sizes: 140 unique valid_seen
games, per-task seed 2026+i, unchanged native action execution and reward.
Responses preserve special tokens in the recorded text and also retain token
IDs. Extra strict-protocol statistics do not change which actions execute.
Do not evaluate the Student with privileged skills or a Teacher.

Report update150, not the checkpoint chosen for its highest test score.
Keep successes/140, success rate, reasoning completeness, native action-valid
rate, per-task outcomes, training/validation curves, source/model/data hashes,
and the actual launch configuration. `src/report.py` generates CSV/PNG from
real results only. Unrun experiments remain explicitly unrun.

## Resume and storage

Native checkpoints save model/optimizer/scheduler/rank RNG/dataloader state.
The launcher verifies four rank-file sets and the original binding before
resuming; it rejects an eight-GPU run or a different recipe. It does not save
the internal TextWorld environment iterator, so exact uninterrupted-versus-
resumed rollout identity is not guaranteed. Interrupted writes are not valid
checkpoints. A partial first save is not permission to restart automatically.

Each repository runs one size and requests native max_actor_ckpt_to_keep=2.
Native retention is not a disk-wide transactional cleanup guarantee after
failures or process restarts; inspect existing saves before recovery. This
split does not port the GRPO repository's separate checkpoint-storage fix.
Reserve at least 200GiB free, preferably 250GiB, for one size's public assets,
two full resumable saves, transient save/export peaks and run evidence. This
is an estimate, not a measured disk high-water mark; old accumulated saves
and failed writes require additional space and must not be silently deleted.
