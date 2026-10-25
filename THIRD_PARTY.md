# Sources and modifications

- **SDAR**, ZJU-REAL: https://github.com/ZJU-REAL/SDAR , Apache-2.0.
  Fixed commit: `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
  Based on veRL and verl-agent. Their notices remain in fetched source.
  Reproduction changes touch `agent_system/environments/env_manager.py`,
  `agent_system/environments/env_package/alfworld/envs.py`, and
  `verl/trainer/main_sdar.py`: fixed-task validation with native one-actor-per-environment execution,
  and a read-only evaluation/logging subclass. An additional helper selects
  existing games. Native loss, Actor/Teacher scoring, GRPO, skills and prompts
  are unmodified. See `patches/runtime.patch` and `configs/upstream.json`.
- **Qwen2.5-1.5B-Instruct**, Qwen: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct , Apache-2.0.
- **Qwen2.5-3B-Instruct**, Qwen: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct , Qwen Research License.
  Model files are downloaded from pinned official revisions, not redistributed here.
- **ALFWorld**: https://github.com/alfworld/alfworld , obtain assets with the
  official downloader and follow the source code/data licenses.
- **Fixed seen140 task manifest**: same public manifest as
  https://github.com/leiye302/rethink-alfworld-qwen25-1.5b ; contains only public
  game-relative identities, hashes and environment seeds, not private trajectories.

Please cite the original method when using this reproduction:

```bibtex
@article{lu2026sdar,
  title={Self-Distilled Agentic Reinforcement Learning},
  author={Lu, Zhengxi and Yao, Zhiyuan and Han, Zhuowen and Wang, Zi-Han and others},
  journal={arXiv preprint arXiv:2605.15155},
  year={2026}
}
```
