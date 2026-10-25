import ast
import copy
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import WORLD_SIZE, checkpoint_complete, devices, load, protocol, tasks
from fixed_validation import fixed_validation_envs
from run import config
from setup import verify


def test_native_source_and_task_locks():
    lock = verify()
    assert len(lock['unchanged_files']) >= 10
    assert 'verl/workers/actor/dp_actor.py' in lock['unchanged_files']
    assert len(tasks()) == 140


@pytest.mark.parametrize('size', ['1.5b', '3b'])
def test_composed_native_recipe(size, tmp_path):
    cfg = config(size, tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    assert (cfg.trainer.n_gpus_per_node, cfg.ray_init.num_gpus) == (4, 4)
    assert cfg.ray_init.num_cpus == 32
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.6
    assert not cfg.actor_rollout_ref.rollout.enforce_eager
    assert not cfg.actor_rollout_ref.rollout.free_cache_engine
    assert not cfg.actor_rollout_ref.model.use_fused_kernels
    assert cfg.algorithm.sdar.gate_beta == 5 and cfg.algorithm.sdar.sdar_coef == 0.01
    assert cfg.data.train_batch_size == 16 and cfg.env.rollout.n == 8
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 256
    assert cfg.actor_rollout_ref.actor.kl_loss_coef == 0.01
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0.001
    assert cfg.env.max_steps == 50 and cfg.env.history_length == 2
    assert cfg.data.max_response_length == 512
    assert cfg.data.val_batch_size == 140
    assert cfg.trainer.total_training_steps == 150
    assert cfg.trainer.test_freq == 10 and cfg.trainer.save_freq == 20
    assert cfg.trainer.max_actor_ckpt_to_keep == 2
    assert 'local_hints' not in cfg.algorithm
    assert cfg.actor_rollout_ref.model.get('lora_rank', 0) == 0


@pytest.mark.parametrize('spec', ['0,1', '0,1,2', '0,1,2,2', '0,1,2,3,4',
                                 '0,1,2,3,4,5,6,7', '', '0,1,2,GPU-xxx'])
def test_invalid_gpu_allocation(spec):
    with pytest.raises(ValueError):
        devices(spec)


def test_four_gpu_allocation():
    assert WORLD_SIZE == 4
    assert devices('0,1,2,3') == [str(i) for i in range(4)]
    assert devices('4,5,6,7') == [str(i) for i in range(4, 8)]


def test_partial_checkpoint_rejected(tmp_path):
    with pytest.raises(RuntimeError):
        checkpoint_complete(tmp_path)
    for rank in range(WORLD_SIZE):
        for kind in ('model', 'optim', 'extra_state'):
            path = tmp_path / 'actor' / f'{kind}_world_size_{WORLD_SIZE}_rank_{rank}.pt'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'unit-test')
    (tmp_path/'actor/config.json').write_text('{}')
    (tmp_path/'data.pt').write_bytes(b'unit-test')
    assert checkpoint_complete(tmp_path) == tmp_path


def test_telemetry_does_not_accept_duplicate_actions():
    assert protocol('<think>Plan.</think><action>look</action>')['strict_protocol']
    assert not protocol('<think></think><action>look</action>')['strict_protocol']
    assert not protocol('<think>Plan.</think><action>look</action><action>look</action>')['strict_protocol']


def test_fixed_task_uses_only_seen_and_preserves_base(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setenv('ALFWORLD_DATA', str(tmp_path))
    game = tmp_path/'json_2.1.1/valid_seen/task/trial/game.tw-pddl'
    game.parent.mkdir(parents=True)
    game.write_bytes(b'game')
    manifest = tmp_path/'manifest.json'
    row = {'task_id': game.relative_to(tmp_path).as_posix(), 'seed': 2026,
           'game_sha256': hashlib.sha256(b'game').hexdigest()}
    manifest.write_text(json.dumps([row]))
    base = SimpleNamespace(game_files=[str(game)], num_games=140)
    selected = fixed_validation_envs(base, manifest, 1, 1)
    assert selected[0].fixed_validation_seed == 2026
    assert base.num_games == 140 and selected[0] is not base
    row['game_sha256'] = '0'*64
    manifest.write_text(json.dumps([row]))
    with pytest.raises(ValueError):
        fixed_validation_envs(base, manifest, 1, 1)


def test_native_sdar_gradient_is_teacher_dependent_and_teacher_detached():
    torch = pytest.importorskip('torch')
    source = ROOT/'.deps/sdar/verl/trainer/ppo/sdar_utils.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_sdar_loss')
    namespace = {'torch': torch, 'agg_loss': lambda loss_mat, loss_mask, loss_agg_mode:
                 (loss_mat*loss_mask).sum()/loss_mask.sum()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    gradients = []
    for teacher_value in (-0.1, -3.0):
        student = torch.tensor([[-1.0, -1.0]], requires_grad=True)
        teacher = torch.full_like(student, teacher_value, requires_grad=True)
        mask = torch.tensor([[1.0, 0.0]])
        loss, _ = namespace['compute_sdar_loss'](student, teacher, mask, gate_beta=5.0)
        loss.backward()
        assert teacher.grad is None and student.grad[0, 1].item() == 0
        gradients.append(student.grad[0, 0].item())
    assert gradients[0] < -0.9 and abs(gradients[1]) < 0.001


def test_native_workers_preserve_seeds_order_and_results():
    source = ROOT/'.deps/sdar/agent_system/environments/env_package/alfworld/envs.py'
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in {'AlfworldWorker', 'AlfworldEnvs'}]

    class NativeEnvironment:
        def seed(self, seed): self.value = seed
        def reset(self): return [str(self.value)], {'admissible_commands': [['look']]}
        def step(self, actions):
            return [str(self.value)+actions[0]], [0], [False], {'admissible_commands': [['look']], 'won': [False]}

    class Base:
        def init_env(self, batch_size): return NativeEnvironment()

    class Method:
        def __init__(self, fn): self.remote = fn

    class Handle:
        def __init__(self, obj): self.obj = obj
        def __getattr__(self, name): return Method(getattr(self.obj, name))

    class Remote:
        def __init__(self, cls): self.cls = cls
        def remote(self, *args): return Handle(self.cls(*args))

    ray = SimpleNamespace(is_initialized=lambda: True, get=lambda x:x,
                          remote=lambda **kwargs: lambda cls:Remote(cls))
    namespace = {'gym': SimpleNamespace(Env=object), 'ray': ray, 'get_environment': lambda _: lambda *a,**k:Base(),
                 'load_config_file': lambda _: {'env': {'type': 'AlfredTWEnv'}},
                 'compute_reward': lambda info,multi_modal: 0}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), namespace)
    env = namespace['AlfworldEnvs']('unused', 17, 3, 2, {})
    assert len(env.workers) == 6
    assert not hasattr(env, 'pool_size')
    reset = copy.deepcopy(env.reset())
    step = copy.deepcopy(env.step(['look'] * 6))
    assert reset[0] == ['17', '17', '18', '18', '19', '19']
    assert step[0] == ['17look', '17look', '18look', '18look', '19look', '19look']


def test_dry_run_needs_no_model_or_gpu(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'src/run.py'), 'train', '--size', '1.5b',
                             '--run-dir', str(tmp_path/'unused'), '--dry-run'],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path/'unused').exists()
    assert 'gate_beta: 5.0' in result.stdout


def test_models_share_all_scientific_settings(tmp_path):
    from omegaconf import OmegaConf
    left = config('1.5b', tmp_path/'model15', tmp_path/'assets', tmp_path/'run15')
    right = config('3b', tmp_path/'model3', tmp_path/'assets', tmp_path/'run3')
    for cfg in (left, right):
        cfg.actor_rollout_ref.model.path = 'model'
        cfg.critic.model.tokenizer_path = 'model'
        cfg.reward_model.model.input_tokenizer = 'model'
        cfg.trainer.experiment_name = 'experiment'
        cfg.data.train_files = cfg.data.val_files = 'parquet'
        cfg.trainer.default_local_dir = cfg.trainer.validation_data_dir = cfg.trainer.rollout_data_dir = 'output'
        cfg.ray_init.runtime_env.env_vars.SDAR_RUN_DIR = 'output'
    assert OmegaConf.to_container(left, resolve=True) == OmegaConf.to_container(right, resolve=True)


def test_scientific_settings_match_official_3b_launcher_except_gpu_count(tmp_path):
    """Only GPU topology differs; native loss and global batches stay fixed."""
    from omegaconf import OmegaConf
    source = (ROOT/'.deps/sdar/examples/sdar_trainer/run_alfworld_3b.sh').read_text()
    assignments = dict(re.findall(r'(?m)^\s*\+?([\w.]+)=([^\s\\]+)', source))
    cfg = config('3b', tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    keys = [
        'algorithm.adv_estimator', 'data.train_batch_size', 'data.max_prompt_length',
        'data.max_response_length', 'data.filter_overlong_prompts', 'data.truncation',
        'data.return_raw_chat', 'actor_rollout_ref.actor.optim.lr',
        'actor_rollout_ref.model.use_remove_padding',
        'actor_rollout_ref.actor.ppo_mini_batch_size',
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu',
        'actor_rollout_ref.actor.use_kl_loss', 'actor_rollout_ref.actor.kl_loss_coef',
        'actor_rollout_ref.actor.kl_loss_type',
        'actor_rollout_ref.model.enable_gradient_checkpointing',
        'actor_rollout_ref.actor.fsdp_config.param_offload',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu',
        'actor_rollout_ref.rollout.tensor_model_parallel_size',
        'actor_rollout_ref.rollout.gpu_memory_utilization',
        'actor_rollout_ref.rollout.enable_chunked_prefill',
        'actor_rollout_ref.rollout.enforce_eager',
        'actor_rollout_ref.rollout.free_cache_engine',
        'actor_rollout_ref.rollout.val_kwargs.temperature',
        'actor_rollout_ref.rollout.val_kwargs.do_sample',
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu',
        'actor_rollout_ref.ref.fsdp_config.param_offload',
        'actor_rollout_ref.actor.use_invalid_action_penalty',
        'actor_rollout_ref.actor.invalid_action_penalty_coef',
        'algorithm.use_kl_in_reward', 'algorithm.sdar.sdar_coef',
        'algorithm.sdar.gate_beta', 'algorithm.sdar.skill_all',
        'env.env_name', 'env.seed', 'env.max_steps', 'env.rollout.n',
        'trainer.critic_warmup',
        'trainer.ray_wait_register_center_timeout', 'trainer.nnodes',
        'trainer.total_epochs', 'trainer.val_before_train',
    ]
    for key in keys:
        raw = assignments[key]
        if raw.startswith('$'):
            raw = assignments[raw[1:]]
        if raw.lower() in ('true', 'false'):
            expected = raw.lower() == 'true'
        elif raw.startswith("'"):
            expected = raw.strip("'")
        else:
            try:
                expected = float(raw)
            except ValueError:
                expected = raw
        assert OmegaConf.select(cfg, key) == expected, key


def test_eight_rank_checkpoint_is_not_a_four_rank_resume(tmp_path):
    for rank in range(8):
        for kind in ('model', 'optim', 'extra_state'):
            path = tmp_path/'actor'/f'{kind}_world_size_8_rank_{rank}.pt'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'unit-test')
    (tmp_path/'actor/config.json').write_text('{}')
    (tmp_path/'data.pt').write_bytes(b'unit-test')
    with pytest.raises(RuntimeError, match='Incomplete resumable checkpoint'):
        checkpoint_complete(tmp_path)


def test_validation_rollout_uses_native_padding_then_unpadding():
    source = (ROOT/'.deps/sdar/agent_system/multi_turn_rollout/rollout_loop.py').read_text()
    assert 'pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)' in source
    assert 'unpad_dataproto(batch_output_padded, pad_size=pad_size)' in source


def test_runtime_directory_config_can_be_set_before_ray_start(tmp_path):
    cfg = config('1.5b', tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    cfg.ray_init._temp_dir = str(tmp_path/'ray')
    assert cfg.ray_init._temp_dir == str(tmp_path/'ray')


def test_recipe_differs_from_eight_gpu_release_only_in_gpu_count():
    recipe = load('experiment.json')
    assert recipe.pop('trainer.n_gpus_per_node') == 4
    assert recipe['env.resources_per_worker.num_cpus'] == 0.1
    assert 'env.alfworld.worker_pool_size' not in recipe
    # Reconstruct only the removed runtime grouping to check the original scientific recipe.
    recipe['env.resources_per_worker.num_cpus'] = 1
    recipe['env.alfworld.worker_pool_size'] = 16
    normalized = json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()
    # Baseline e8d323f, excluding its one hardware-topology field.
    assert hashlib.sha256(normalized).hexdigest() == 'abd97ac43f6c98227d5af5922c9ef4db4416b6399e45a538695cb3b246689881'


def test_native_gradient_accumulation_preserves_global_batch_on_four_gpus(tmp_path):
    cfg = config('3b', tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    source = ROOT/'.deps/sdar/verl/workers/fsdp_workers.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    normalization = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
        and ast.unparse(node.test) == 'self._is_actor'
        and 'self.config.actor.ppo_mini_batch_size *= self.config.rollout.n' in ast.unparse(node))
    results = {}
    for ranks in (4, 8):
        worker = SimpleNamespace(_is_actor=True, config=copy.deepcopy(cfg.actor_rollout_ref),
            device_mesh=SimpleNamespace(size=lambda: ranks), ulysses_sequence_parallel_size=1)
        exec(compile(ast.Module(body=[normalization], type_ignores=[]), str(source), 'exec'), {'self': worker})
        local = worker.config.actor.ppo_mini_batch_size
        micro = worker.config.actor.ppo_micro_batch_size_per_gpu
        assert ranks * local == cfg.actor_rollout_ref.actor.ppo_mini_batch_size
        results[ranks] = (local, micro, local // micro)
    assert results == {4: (64, 32, 2), 8: (32, 32, 1)}


def test_release_allowlist_and_privacy():
    result = subprocess.run([sys.executable, str(ROOT/'scripts/check_release.py')],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_evaluation_hook_preserves_native_success_and_all_tasks(tmp_path, monkeypatch):
    import importlib.util
    import types
    torch = pytest.importorskip('torch')
    rows = tasks()
    gamefiles = [str(tmp_path/row['task_id']) for row in rows]
    batch = SimpleNamespace(
        batch={'responses': torch.ones((140, 2), dtype=torch.long),
               'attention_mask': torch.ones((140, 4), dtype=torch.long)},
        non_tensor_batch={'gamefile': gamefiles, 'traj_uid': [str(i) for i in range(140)],
                          'turn_step': [0]*140, 'episode_rewards': [10.0]*70+[0.0]*70,
                          'is_action_valid': [True]*140})

    class NativeTrainer:
        def _validate(self):
            self.traj_collector.multi_turn_loop()
            return {'val/success_rate': 0.5}

    native_module = types.ModuleType('verl.trainer.ppo.skillsd_ray_trainer')
    native_module.SkillSDRayTrainer = NativeTrainer
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.skillsd_ray_trainer', native_module)
    spec = importlib.util.spec_from_file_location('tested_sdar_eval', ROOT/'src/sdar_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('ALFWORLD_DATA', str(tmp_path))
    monkeypatch.setenv('SDAR_RUN_DIR', str(tmp_path/'run'))
    trainer = module.FixedSeenTrainer()
    trainer.global_steps = 10
    original = lambda:batch
    trainer.traj_collector = SimpleNamespace(multi_turn_loop=original)
    trainer.tokenizer = SimpleNamespace(decode=lambda ids,**kw:'<think>Plan.</think><action>look</action>')
    assert trainer._validate() == {'val/success_rate':0.5}
    assert trainer.traj_collector.multi_turn_loop is original
    summary = json.loads((tmp_path/'run/evaluation/update_010/summary.json').read_text())
    assert summary['successes'] == 70 and len(summary['task_results']) == 140
    assert summary['reasoning_complete_rate'] == 1
