# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation (from source)
```bash
pip install -e ".[dev,test]"
# For specific hardware/policy extras:
pip install -e ".[smolvla,aloha]"
```

### Linting and Formatting
```bash
pre-commit run --all-files        # Run all pre-commit hooks (ruff format + lint, typos)
pre-commit install                # Install hooks to run automatically before commits
```

### Tests
```bash
# Requires git-lfs artifacts:
git lfs install && git lfs pull

pytest -sv ./tests                              # Run full test suite
pytest -sv tests/test_specific_feature.py       # Run a specific test file
pytest -sv tests/datasets/test_lerobot_dataset.py  # Run a specific module
```

### CLI Entry Points
```bash
lerobot-train --policy=act --dataset.repo_id=lerobot/aloha_mobile_cabinet
lerobot-eval --policy.path=lerobot/pi0_libero_finetuned --env.type=libero
lerobot-record --robot.type=so100 --dataset.repo_id=<user>/<dataset>
lerobot-teleoperate --robot.type=so100
lerobot-calibrate --robot.type=so100
lerobot-dataset-viz --dataset.repo_id=lerobot/aloha_mobile_cabinet
lerobot-edit-dataset  # Delete episodes, split, merge datasets
lerobot-info          # Show installed components info
```

## Architecture

### Package Layout (`src/lerobot/`)

| Module | Purpose |
|--------|---------|
| `configs/` | Dataclass-based configuration system using `draccus` |
| `datasets/` | `LeRobotDataset` and streaming dataset; Parquet + MP4 format |
| `policies/` | Policy implementations (ACT, Diffusion, SmolVLA, Pi0, etc.) |
| `robots/` | Hardware abstraction (`Robot` base class) |
| `cameras/` | Camera interface abstraction |
| `motors/` | Motor driver abstraction (Feetech, Dynamixel, Damiao, Robstride) |
| `teleoperators/` | Teleoperation devices |
| `envs/` | Simulation environment wrappers (LIBERO, MetaWorld, gym-aloha) |
| `scripts/` | Entry point scripts corresponding to CLI commands |
| `processor/` | Observation/action processing pipeline |
| `rl/` | Reinforcement learning utilities (HIL-SERL, SAC) |
| `async_inference/` | gRPC-based async policy inference |
| `transport/` | gRPC transport layer |
| `optim/` | Optimizer and LR scheduler configs |
| `model/` | Shared model building blocks |
| `utils/` | General utilities |
| `tools/` | Standalone helper scripts |

### Configuration System

All configs are typed Python dataclasses using the `draccus` library. CLI arguments map directly to nested dataclass fields via dot notation:
```bash
lerobot-train --policy.type=act --policy.n_obs_steps=2 --dataset.repo_id=...
```

Key config classes:
- `TrainPipelineConfig` (`configs/train.py`) — top-level training config
- `PreTrainedConfig` (`configs/policies.py`) — base for all policy configs; uses `draccus.ChoiceRegistry` for `--policy.type=<name>` dispatch
- `DatasetConfig` (`configs/default.py`) — dataset loading options
- `RobotConfig` (`robots/config.py`) — base for robot hardware configs

The `parser.wrap()` decorator (`configs/parser.py`) enables loading configs from pretrained Hub checkpoints via `--policy.path=<hub_id_or_local_dir>`.

### Policy System

All policies extend `PreTrainedPolicy` (`policies/pretrained.py`), which subclasses `nn.Module` and `HubMixin`. Each policy must define:
- `config_class` — corresponding `PreTrainedConfig` subclass
- `name` — string identifier used for `--policy.type=<name>` dispatch
- `select_action(batch)` — inference method
- `forward(batch)` — training method returning loss dict

Policy configs register themselves with `draccus.ChoiceRegistry` so the factory (`policies/factory.py`) can instantiate any policy by type name.

### Dataset Format (v3.0)

`LeRobotDataset` stores data as:
- **Parquet files** for state/action/metadata (chunked by episode)
- **MP4 videos** for camera observations (one file per camera per episode or chunk)
- **`info.json`** for dataset metadata (features, fps, shapes, stats)

Datasets live in `$HF_LEROBOT_HOME/<repo_id>/` by default. The `streaming=True` option enables streaming from Hub without full download.

### Robot Interface

`Robot` (`robots/robot.py`) is an abstract base class requiring:
- `connect()` / `disconnect()`
- `get_observation() -> RobotObservation`
- `send_action(action: RobotAction)`
- `calibrate()`

The processor pipeline (`processor/`) handles converting between robot observations/actions and policy-ready tensor batches.

### Extending the Codebase

- **New policy:** Subclass `PreTrainedConfig` and `PreTrainedPolicy`, define `name`, decorate config with `@PreTrainedConfig.register_subclass("myname")`.
- **New robot:** Subclass `Robot` and `RobotConfig`, implement abstract methods.
- **New environment:** Subclass `EnvConfig` and register with draccus; implement a gym-compatible env.
- **External plugins** can be loaded at runtime via `--env.discover_packages_path=mypackage` or `--policy.discover_packages_path=mypackage`.

## Code Style

- Linter: `ruff` (line length 110, targeting Python 3.10+)
- Quote style: double quotes
- Type annotations enforced in `configs/`, `optim/`, `model/`, `cameras/`, `motors/`, `transport/`, `envs/` (mypy strict); lenient elsewhere
- No print statements in library code (use `logging`)
- Docstring convention: Google style
