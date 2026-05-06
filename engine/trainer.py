from __future__ import annotations

import time
from pathlib import Path

import torch

from env import G1MimicEnv, MimicEnvConfig
from net import FlowMatchingPolicy
from .checkpoint import CheckpointMixin
from .config import MixGRPOConfig
from .logging import LoggingMixin
from .sampling import flow_grpo_step
from .validation import ValidationMixin


class MixGRPOTrainer(ValidationMixin, CheckpointMixin, LoggingMixin):
    def __init__(self, simulation_app, cfg: MixGRPOConfig):
        self.simulation_app = simulation_app
        self.cfg = cfg
        self.start_update = 1
        self.chunk_dim = cfg.horizon * cfg.action_dim
        self.checkpoint_dir = Path(cfg.checkpoint_dir).expanduser().resolve() if cfg.checkpoint_dir else None

        self.env = G1MimicEnv(
            MimicEnvConfig(
                device=cfg.device,
                num_envs=cfg.num_envs,
                sim_dt=cfg.sim_dt,
                fix_root_link=cfg.fix_root_link,
                motion_start_phase=cfg.motion_start_phase,
                motion_end_phase=cfg.motion_end_phase,
                motion_file=cfg.motion_file,
                max_episode_steps=cfg.max_episode_steps,
            )
        )
        if self.env.action_dim != cfg.action_dim:
            raise ValueError(f"Expected env action_dim {self.env.action_dim}, got {cfg.action_dim}")
        self.num_grpo_groups = self.env.num_envs

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        policy_obs_dim = cfg.policy_obs_dim if cfg.policy_obs_dim > 0 else self.env.observation_dim
        self.policy = FlowMatchingPolicy(
            obs_dim=policy_obs_dim,
            action_dim=cfg.action_dim,
            horizon=cfg.horizon,
            hidden_dim=cfg.hidden_dim,
            time_embed_dim=cfg.time_embed_dim,
            depth=cfg.depth,
            action_limit=cfg.action_limit,
        ).to(self.env.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.current_observation = self._reset_training_envs()

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if cfg.resume:
            self._load_checkpoint(Path(cfg.resume).expanduser().resolve())

    # 拍照存档
    def _snapshot_env_state(self) -> dict[str, torch.Tensor]:
        robot = self.env.robot
        return {
            "root_state_w": robot.data.root_state_w.clone(),    # 机器人的世界坐标系根状态张量
            "joint_pos": robot.data.joint_pos.clone(),  # 记录所有关节的角度
            "joint_vel": robot.data.joint_vel.clone(),  # 记录关节的角速度
            "phase_steps": self.env.phase_steps.clone(),    # 记录每个环境当前处于运动的哪个帧
            "episode_steps": self.env.episode_steps.clone(),    # 每个环境的当前 episode 步骤计数器
            "last_action": self.env.last_action.clone(),    # 上一次执行的动作指令。
            "next_push_step": self.env.next_push_step.clone(),  # 下一次施加外部扰动的时间点。
            "bin_failed_count": self.env.bin_failed_count.clone(), # 特定任务（如投篮或放料）的失败累积计数。
        }
    # 读档回滚
    def _restore_env_state(self, snapshot: dict[str, torch.Tensor]) -> None:
        env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        root_state = snapshot["root_state_w"]
        # 将世界坐标（World Space）转回本地坐标（Local Space）。
        root_pos_local = root_state[:, :3] - self.env.scene.env_origins
        # 把快照里的位置、姿态、线速度、角速度、关节角度、关节速度直接写入物理引擎的缓存。
        self.env._write_robot_state(
            root_pos=root_pos_local,
            root_quat=root_state[:, 3:7],
            root_lin_vel=root_state[:, 7:10],
            root_ang_vel=root_state[:, 10:13],
            joint_pos=snapshot["joint_pos"][:, self.env.action_joint_ids],
            joint_vel=snapshot["joint_vel"][:, self.env.action_joint_ids],
            env_ids=env_ids,
        )
        # 把步态相位、回合步数、上一次动作等逻辑变量统统改回快照的值。
        self.env.phase_steps = snapshot["phase_steps"].clone()
        self.env.episode_steps = snapshot["episode_steps"].clone()
        self.env.last_action = snapshot["last_action"].clone()
        self.env.next_push_step = snapshot["next_push_step"].clone()
        self.env.bin_failed_count = snapshot["bin_failed_count"].clone()
        self.env._current_bin_failed.zero_()
        # 清除这些环境在物理引擎缓存中的旧状态（比如上一帧的加速度、接触力等），确保引擎开始从你读档的新位置重新计算。
        self.env.scene.reset(env_ids=env_ids)
        # 仿真器向前推进一步物理时间，需要“播放”一帧，画面才会更新
        self.env.scene.update(self.env.physics_dt)

    def _cps_sample_with_logprobs(self, obs: torch.Tensor, noise: torch.Tensor) -> dict[str, torch.Tensor]:
        # 检查输入的观测值（obs）和噪声（noise）维度是否正确，以及步数是否合法。
        self.policy._validate_inputs(obs, noise, self.cfg.flow_steps)
        # 把 obs 调整成策略网络固定的观测维度
        obs_prep = self.policy._prepare_observation(obs)
        batch_size = noise.shape[0]
        # 去噪步数
        steps = self.cfg.flow_steps
        # 构建时间调度表，1.0: 代表纯噪声状态（起始点），0.0: 代表纯动作状态（终点）。
        # 形状类似于[1.0,  0.75,  0.5,  0.25,  0.0]
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)
        # 当前的“中间态”动作。最开始它就是纯噪声。
        latent = noise
        # all_latents / all_log_probs: 列表容器，用来记录每一步演变的结果和对应的概率（Log-Prob）
        all_latents = [latent.detach()]
        all_log_probs = []

        for step_index in range(steps):
            sigma = sigma_schedule[step_index]
            # s_val = 1.0 - sigma: 将“噪声水平”转换为“时间进度 s”。
            s_val = 1.0 - sigma.item()
            # 把这个标量时间扩充成一个 Batch 维度的向量
            s_batch = torch.full((batch_size,), s_val, device=noise.device, dtype=noise.dtype)

            # velocity_field: 这是你训练的核心神经网络。
            # 它的输入是（环境观测、当前状态、当前时间），输出是一个“速度向量”。
            with torch.no_grad():
                velocity = self.policy.velocity_field(obs_prep, latent, s_batch)
            model_output = -velocity
            # deterministic: 如果是最后一步，通常会关掉随机干扰，确保输出稳定。
            deterministic = step_index == steps - 1
            # 它根据神经网络给出的“速度”，把 latent 往前推一小步。
            latent, log_prob = flow_grpo_step(
                model_output=model_output,
                latents=latent,
                sigmas=sigma_schedule,
                index=step_index,
                deterministic=deterministic,
                noise_level=self.cfg.cps_eta,
            )
            # 将这一步更新后的 latent 和产生的 log_prob 存起来。
            all_latents.append(latent.detach())
            all_log_probs.append(log_prob.detach())
        pre_tanh = torch.clamp(latent, -10.0, 10.0)
        # 由于action_limit=1: 把网络输出的动作值限制在[-1, 1]内，防止数值爆炸。
        actions = self.policy._action_transform(pre_tanh)
        return {
            "actions": actions.view(batch_size, self.policy.horizon, self.policy.action_dim),
            "all_latents": torch.stack(all_latents, dim=1),
            "all_log_probs": torch.stack(all_log_probs, dim=1),
            "sigma_schedule": sigma_schedule,
        }

    def _recompute_logprob_one_step(
        self,
        obs_prep: torch.Tensor,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        sigma_schedule: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        # 将 s_val = 1.0 - sigma 转换成一个 Batch 维度的向量，供网络输入使用。
        batch_size = latents.shape[0]
        sigma = sigma_schedule[step_index]
        s_val = 1.0 - sigma.item()
        s_batch = torch.full((batch_size,), s_val, device=latents.device, dtype=latents.dtype)
        # 用“最新”的神经网络重新思考旧的轨迹数据，计算出新的 velocity 和 log_prob。
        velocity = self.policy.velocity_field(obs_prep, latents, s_batch)
        _, log_prob = flow_grpo_step(
            model_output=-velocity,
            latents=latents,
            sigmas=sigma_schedule,
            index=step_index,
            prev_sample=next_latents,
            deterministic=False,
            noise_level=self.cfg.cps_eta,
        )
        return log_prob

    def _collect_groups(self, current_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        total_envs = self.env.num_envs
        group_size = self.cfg.group_size
        chunks_per_rollout = self.cfg.chunks_per_rollout
        # 在搜集数据开始前，先把当前所有平行环境（比如 4096 ）的物理状态统统“拍照”存下来。
        # 之后这 4096 个环境会同时尝试第 1 种路线，记录得分，然后同时尝试第 2 种路线，记录得分......
        snapshot = self._snapshot_env_state()
        sigma_schedule = None

        metric_chunk_return_first = None
        metric_chunk_return_last = None
        metric_actions_first = None
        metric_actions_last = None

        all_group_rollout_rewards = []
        all_group_chunk_rewards = []
        all_group_obs = []
        all_group_latents = []
        all_group_log_probs = []
        all_group_valid_masks = []

        metric_chunk_return: torch.Tensor | None = None
        metric_actions: torch.Tensor | None = None
        metric_infos_list: list[dict[str, torch.Tensor]] = []

        # 探索 group_size 条不同的路线。
        for group_index in range(group_size):
            # 所有机器人瞬间“传送”回刚才的存档点。
            self._restore_env_state(snapshot)
            # 所有机器人的状态集合，形状大致为[4096, 154]
            obs_t = current_obs
            # 用来标记哪些机器人在接下来的探索中“摔倒了”
            done_mask = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)

            group_chunk_rewards = []
            group_chunk_obs = []
            group_chunk_latents = []
            group_chunk_log_probs = []
            group_chunk_valid_masks = []

            # 机器人在当前路线上会连续往前走 chunks_per_rollout 个动作块（Chunk）。
            # 每次循环触发 1 次 CPS 采样，生成一段长度为 horizon 的动作，交由环境连续执行。
            # 期间会记录每个动作块（Chunk）的奖励、观测、潜在状态、概率和有效性（是否摔倒）。
            # 如果某个环境在中途摔倒了，那么它后续的奖励和数据都会被掩盖掉，不参与训练。
            for chunk_index in range(chunks_per_rollout):
                chunk_valid_mask = ~done_mask
                noise = torch.zeros(total_envs, self.chunk_dim, device=self.env.device)
                # 根据当前的观测 obs_t 和随机噪声 noise，调用 CPS 采样函数，
                # 生成一段长度为self.cfg.horizon的动作序列和对应的潜在状态、概率等信息。
                with torch.no_grad():
                    sample = self._cps_sample_with_logprobs(obs_t, noise)

                # 去噪的进度表，_recompute_logprob_one_step中，在重新打分的时候，
                # 网络必须知道“当前处于去噪的哪一步”（即方差有多大，该加多少噪声）。
                sigma_schedule = sample["sigma_schedule"]

                # 把刚才算出来的、长度为self.cfg.horizon的动作序列 sample["actions"] 丢给底层仿真器。
                # 仿真器会连续执行这几步，并返回：
                # 执行过程中的得分 (step_rewards)、是否摔倒 (terminations)、是否超时 (truncations) 等信息。
                # 注意这里设置了不自动重置 (auto_reset=False)，摔倒了就躺着。
                _, step_rewards, terminations, truncations, infos_list = self.env.chunk_step(
                    sample["actions"],
                    auto_reset=False,
                )
                # 计算这个动作块的总奖励（中途摔倒，之前的几步也是有奖励的）。
                chunk_reward = step_rewards.sum(dim=1).masked_fill(done_mask, 0.0)
                # 如果在刚才这几步里，机器人新发生了摔倒（terminations）或超时（truncations），
                # 就把它们加入到 done_mask 里（使用逻辑或 |）。
                done_mask = done_mask | (terminations | truncations).any(dim=1)
                # 动作执行完了，再次拍照，获取环境最新的观测画面。
                next_obs_t = self.env.get_observation()

                # 这段是为了写日志（Log / Tensorboard）。
                # 为了防止打印太多东西，只挑取“第 0 个分支”的“第 0 个动作块”的数据作为监控指标保存下来。
                if group_index == 0:
                    if chunk_index == 0:
                        metric_chunk_return_first = chunk_reward.detach()
                        metric_actions_first = sample["actions"].clone()
                        metric_infos_list = infos_list  # Info 通常存一份即可
                    
                    if chunk_index == chunks_per_rollout - 1:
                        metric_chunk_return_last = chunk_reward.detach()
                        metric_actions_last = sample["actions"].clone()

                # 把刚才这一轮（这个 Chunk）发生的所有事：
                # 你看到了什么（obs_t）、脑子里的中间想法（latents）、
                # 确信度（log_probs）、得了多少分（rewards）、是否有效（valid_masks），
                # 全部追加（append）进当前组（Group）的列表里。
                group_chunk_obs.append(obs_t)
                group_chunk_latents.append(sample["all_latents"])
                group_chunk_log_probs.append(sample["all_log_probs"][:, :-1])
                group_chunk_rewards.append(chunk_reward)
                group_chunk_valid_masks.append(chunk_valid_mask)
                # 时间向前推移。把刚刚获取的“新画面”变成了“当前画面”，准备进入下一次 chunk_index 循环。
                obs_t = next_obs_t
            # 形状 [4096, chunks_per_rollout]（假设 chunk 为 4）的矩阵。
            # 代表 4096 个机器人在当前路线 (group_index) 上，连续 4 个动作块（每个块包含 horizon 步）的各自得分。
            group_chunk_rewards_t = torch.stack(group_chunk_rewards, dim=1)
            # 把那 4 步的分数加起来，算出一个总分，append到all_group_rollout_rewards列表中
            all_group_rollout_rewards.append(group_chunk_rewards_t.sum(dim=1))
            # 把每一步的分数也保留下来，append到all_group_chunk_rewards列表中。
            all_group_chunk_rewards.append(group_chunk_rewards_t)
            # 把这个组（Group）里所有动作块（Chunk）的观测、潜在状态、概率和有效性数据，append到总列表里。
            all_group_obs.append(torch.stack(group_chunk_obs, dim=1))
            all_group_latents.append(torch.stack(group_chunk_latents, dim=1))
            all_group_log_probs.append(torch.stack(group_chunk_log_probs, dim=1))
            all_group_valid_masks.append(torch.stack(group_chunk_valid_masks, dim=1))
        # 此时外层循环（group_index）全部跑完，机器人全部结束（包括倒地或者是完成了整个动作）
        # rollout_rewards 的形状变成了 [4096, 4]（4096 个机器人，每个机器人 4 种路线的总分）。
        rollout_rewards = torch.stack(all_group_rollout_rewards, dim=1)
        # chunk_rewards 的形状变成了 [4096, 4, 4]（4096 个机器人，4 种路线，每条路线 4 步的详细得分）。
        # 这个张量之后会被直接送去计算 Advantage！
        chunk_rewards = torch.stack(all_group_chunk_rewards, dim=1)

        if metric_chunk_return is None:
            metric_chunk_return = torch.zeros(total_envs, device=self.env.device)
        if metric_actions is None:
            metric_actions = torch.zeros(total_envs, self.cfg.horizon, self.cfg.action_dim, device=self.env.device)
       
        # 最后打包成字典返回，这就是喂给强化学习算梯度的终极数据集。
        return {
            "obs": torch.stack(all_group_obs, dim=1),
            "rewards": rollout_rewards,
            "chunk_rewards": chunk_rewards,
            "latents": torch.stack(all_group_latents, dim=1),
            "log_probs": torch.stack(all_group_log_probs, dim=1),
            "valid_mask": torch.stack(all_group_valid_masks, dim=1),
            "sigma_schedule": sigma_schedule,
            "metric_chunk_return": metric_chunk_return,
            "metric_actions": metric_actions,
            "metric_infos_list": metric_infos_list,
            "metric_chunk_return_first": metric_chunk_return_first,
            "metric_chunk_return_last": metric_chunk_return_last,
            "metric_actions_first": metric_actions_first,
            "metric_actions_last": metric_actions_last,
        }

    # 输入形状 [total_envs, group_size, chunks_per_rollout] (例如：[4096, 4, 4])
    def _compute_grpo_advantages(self, chunk_rewards: torch.Tensor) -> torch.Tensor:
        returns = torch.zeros_like(chunk_rewards)
        running = torch.zeros_like(chunk_rewards[:, :, 0])
        for chunk_index in range(chunk_rewards.shape[-1] - 1, -1, -1):
            running = chunk_rewards[:, :, chunk_index] + self.cfg.discount_gamma * running
            returns[:, :, chunk_index] = running
        mean = returns.mean(dim=1, keepdim=True)
        std = returns.std(dim=1, keepdim=True) + 1e-8
        # 输出形状和输入形状一样，每个数字都是相对优势（chunk级别的）
        return (returns - mean) / std

    def _policy_update(
        self,
        obs: torch.Tensor,
        latents: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        sigma_schedule: torch.Tensor,
    ) -> dict[str, float]:
        # obs 的形状为 [N, obs_dim]，比如 N = 100000，代表这里有 10 万张有效状态的独立照片
        sample_count = obs.shape[0]
        # step_count，也就是Flow 去噪步数，比如 4 步。
        step_count = old_log_probs.shape[1]
        # 把 advantages 限制在一个范围内，防止某一次得分高得离谱，带偏整个网络。
        clipped_advantages = torch.clamp(advantages, -self.cfg.adv_clip_max, self.cfg.adv_clip_max)
        obs_prep = self.policy._prepare_observation(obs)
        probe_count = min(128, sample_count)
        probe_obs = obs[:probe_count]
        probe_noise = torch.zeros(probe_count, self.chunk_dim, device=obs.device, dtype=obs.dtype)
        with torch.no_grad():
            # 为了监控训练是否健康，代码挑选了前 128 个样本，
            # 记录下网络在更新前输出的动作（probe_action_before）和当前的权重（params_before）。
            # 等更新完了再比对一下，看看“脑子”变化了多少。
            probe_action_before = self.policy(probe_obs, probe_noise, steps=self.cfg.flow_steps)
            params_before = [param.detach().clone() for param in self.policy.parameters()]

        totals = {
            "loss": 0.0,               # 总损失
            "policy_loss": 0.0,        # PPO 策略损失（主角）
            "latent_reg_loss": 0.0,    # 惩罚网络输出过大的正则损失
            "action_sat_loss": 0.0,    # 惩罚动作超出物理限位的损失
            "clip_frac": 0.0,          # 触发了 Clip 截断的比例（反映网络是不是步子迈太大了）
            "ratio": 0.0,              # 新旧策略的平均概率比
            "ratio_min": float("inf"), # 概率比的最小值
            "ratio_max": 0.0,
            "logprob_delta_abs": 0.0,
        }
        update_count = 0
        grad_norm = 0.0

        # 拿同一批数据学习policy_epochs次。
        for _ in range(self.cfg.policy_epochs):
            # 打乱数据的顺序，让网络训练更稳健
            perm = torch.randperm(sample_count, device=obs.device)
            # 按照 mini_batch_size 的大小，把数据切成一块一块的，逐块送入网络进行训练。
            for start in range(0, sample_count, self.cfg.mini_batch_size):
                mb = perm[start : start + self.cfg.mini_batch_size]
                mb_obs = obs_prep[mb]            # 提取这 mini_batch_size 个样本的 观测值
                mb_adv = clipped_advantages[mb]  # 提取这 mini_batch_size 个样本的 优势得分
                mb_latents = latents[mb]         # 提取这 mini_batch_size 个样本的 动作潜变量
                mb_policy_loss = torch.tensor(0.0, device=obs.device)
                mb_clip_frac = 0.0
                mb_ratio_sum = 0.0

                for step_index in range(step_count):
                    # 在这一步，脑海里的中间状态（比如加了 50% 噪声的动作）。
                    step_latent = mb_latents[:, step_index, :]
                    # 根据旧策略，下一步变成了什么样（比如变成了 25% 噪声的动作）。
                    step_next = mb_latents[:, step_index + 1, :]
                    # 旧大脑当时决定这么做的概率（已经固定，是常数）。
                    step_old_lp = old_log_probs[mb, step_index]
                    # 让刚刚更新过的神经网络，看着同样的状态（step_latent），
                    # 算出如果现在让它选，走到 step_next 的概率是多少。
                    # 这个 new_log_prob 身上挂着完整的计算图和梯度。
                    new_log_prob = self._recompute_logprob_one_step(
                        mb_obs,
                        step_latent,
                        step_next,
                        sigma_schedule,
                        step_index,
                    )
                    # 计算重要性采样比率 (Ratio)
                    ratio = torch.exp(new_log_prob - step_old_lp)
                    logprob_delta = new_log_prob - step_old_lp
                    # ratio 太大（超过 1 + clip_range）或者太小（低于 1 - clip_range）就截断，防止网络更新过度。
                    unclipped_loss = -mb_adv * ratio
                    clipped_loss = -mb_adv * torch.clamp(
                        ratio,
                        1.0 - self.cfg.clip_range,
                        1.0 + self.cfg.clip_range,
                    )
                    mb_policy_loss = mb_policy_loss + torch.maximum(unclipped_loss, clipped_loss).mean()

                    # 监控指标
                    with torch.no_grad():
                        # mb_clip_frac 统计了当前这批数据里，有多少比例的 ratio 触发了截断。
                        mb_clip_frac += (torch.abs(ratio - 1.0) > self.cfg.clip_range).float().mean().item() / step_count
                        mb_ratio_sum += ratio.mean().item() / step_count
                        totals["ratio_min"] = min(totals["ratio_min"], float(ratio.min().item()))
                        totals["ratio_max"] = max(totals["ratio_max"], float(ratio.max().item()))
                        totals["logprob_delta_abs"] += float(logprob_delta.abs().mean().item()) / step_count
                # 把刚才在循环里累加的每一步的 PPO Loss 求个平均。
                mb_policy_loss = mb_policy_loss / max(step_count, 1)
                # 为了知道当前的策略最终会生成什么样的动作，
                # 代码调用 _integrate_flow，从头到尾把初始噪声（mb_latents[:, 0, :]）推演成最终的动作。
                # pred_pre_tanh 是没经过激活函数的原始输出，pred_action 是映射到机器人关节限位后的真实动作。
                pred_pre_tanh = self.policy._integrate_flow(
                    mb_obs,
                    mb_latents[:, 0, :],
                    steps=self.cfg.flow_steps,
                )
                pred_action = self.policy._action_transform(pred_pre_tanh)

                # 潜在空间正则（防数值爆炸）：如果网络输出的原始数字太大（超过了 latent_soft_limit），
                # 就会产生一个平方惩罚（Loss）。逼迫网络在安全的数值范围内工作。
                latent_excess = torch.relu(pred_pre_tanh.abs() - self.cfg.latent_soft_limit)
                mb_latent_reg_loss = torch.mean(latent_excess**2)

                # 如果机器人的动作总是顶到物理极限（比如死命往下压），会导致电机过载或仿真器崩溃。
                # 这里设定了一个 sat_threshold，动作超标了就扣分（产生 Loss）。
                sat_threshold = min(max(self.cfg.action_saturation_threshold, 0.0), self.cfg.action_limit)
                action_excess = torch.relu(pred_action.abs() - sat_threshold)
                mb_action_sat_loss = torch.mean(action_excess**2)

                loss = (
                    mb_policy_loss
                    + self.cfg.latent_reg_coeff * mb_latent_reg_loss
                    + self.cfg.action_saturation_coeff * mb_action_sat_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                totals["loss"] += loss.item()
                totals["policy_loss"] += mb_policy_loss.item()
                totals["latent_reg_loss"] += mb_latent_reg_loss.item()
                totals["action_sat_loss"] += mb_action_sat_loss.item()
                totals["clip_frac"] += mb_clip_frac
                totals["ratio"] += mb_ratio_sum
                update_count += 1

        denom = max(update_count, 1)
        with torch.no_grad():
            # probe_action_after 是用来监控训练健康度的固定探针动作输出
            # 现在网络更新完了，再次输入同样的测试题，看看新动作和旧动作差了多少（action_delta）。
            # 如果变化太大，说明训练不稳定；如果全是 0，说明网络学不动了。
            probe_action_after = self.policy(probe_obs, probe_noise, steps=self.cfg.flow_steps)
            action_delta = torch.mean(torch.abs(probe_action_after - probe_action_before))
            
            # 对比更新前后的网络权重参数矩阵，算出均方根误差（RMS）。
            # 这也是监控训练健康度的顶级指标。最后打包成 return 返回给日志系统。
            param_delta_sq = torch.tensor(0.0, device=obs.device)
            param_count = 0
            for param, before in zip(self.policy.parameters(), params_before, strict=True):
                delta = param.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(delta * delta)
                param_count += delta.numel()
            param_rms_delta = torch.sqrt(param_delta_sq / max(param_count, 1))

        return {
            "policy/loss": totals["loss"] / denom,
            "policy/policy_loss": totals["policy_loss"] / denom,
            "policy/latent_reg_loss": totals["latent_reg_loss"] / denom,
            "policy/action_sat_loss": totals["action_sat_loss"] / denom,
            "policy/clip_frac": totals["clip_frac"] / denom,
            "policy/ratio": totals["ratio"] / denom,
            "policy/ratio_min": totals["ratio_min"] if totals["ratio_min"] != float("inf") else 0.0,
            "policy/ratio_max": totals["ratio_max"],
            "policy/logprob_delta_abs": totals["logprob_delta_abs"] / denom,
            "policy/grad_norm": float(grad_norm),
            "policy/action_delta": float(action_delta.item()),
            "policy/param_rms_delta": float(param_rms_delta.item()),
        }
    
    # 在一次完整的数据搜集（Rollout）中，一共要往前走多少个物理步
    def _training_rollout_horizon(self) -> int:
        return max(1, self.cfg.horizon * self.cfg.chunks_per_rollout)


    def _reset_training_envs(self) -> torch.Tensor:
        # 随机从动作文件中抽取一帧，作为机器人这一次训练的初始位置和姿态。
        phase_indices = self.env.sample_phase_indices(
            self.num_grpo_groups,
            horizon=self._training_rollout_horizon(),
        )
        # 瞬间把所有的机器人按照指定的姿势摆好，然后获取它们的初始观测值，准备开始训练。
        return self.env.reset(phase_indices=phase_indices)

    def train(self) -> None:
        print("[INFO] Starting MixGRPO training", flush=True)
        print(f"[INFO] motion_file={self.cfg.motion_file}", flush=True)
        print(
            f"[INFO] obs_dim={self.env.observation_dim} "
            f"action_chunk=({self.cfg.horizon}, {self.cfg.action_dim}) "
            f"num_envs={self.cfg.num_envs} group_size={self.cfg.group_size} "
            f"chunks_per_rollout={self.cfg.chunks_per_rollout} "
            f"cps_eta={self.cfg.cps_eta} flow_steps={self.cfg.flow_steps} "
            f"action_limit={self.cfg.action_limit}",
            flush=True,
        )
        print(
            f"[INFO] policy_epochs={self.cfg.policy_epochs} "
            f"clip_range={self.cfg.clip_range} "
            f"latent_reg_coeff={self.cfg.latent_reg_coeff} "
            f"action_saturation_coeff={self.cfg.action_saturation_coeff} "
            f"mini_batch={self.cfg.mini_batch_size} lr={self.cfg.lr}",
            flush=True,
        )
        if self.checkpoint_dir is not None:
            print(f"[INFO] checkpoint_dir={self.checkpoint_dir}", flush=True)
        if self.cfg.resume:
            print(f"[INFO] resumed_from={self.cfg.resume}", flush=True)

        for update_idx in range(self.start_update, self.cfg.max_updates + 1):
            if not self.simulation_app.is_running():
                break

            t0 = time.perf_counter()
            # 给所有的机器人分配随机的初始状态。
            current_obs = self._reset_training_envs()
            # 让每个机器人搜集一次完整的 Rollout 数据，包含它在 group_size 条不同路线上的表现。
            group_data = self._collect_groups(current_obs)
            collect_time = time.perf_counter() - t0
            # 给刚才带回来的各种探索路线打上相对表现分。
            advantages = self._compute_grpo_advantages(group_data["chunk_rewards"])
            env_count = self.num_grpo_groups
            group_size = self.cfg.group_size
            chunks = self.cfg.chunks_per_rollout
            steps = self.cfg.flow_steps
            train_steps = max(steps - 1, 0)

            # 原始形状：[4096, 4, 4]。.reshape 后（最终形状）：[65536] （一维张量）
            # 含义：它被拍扁成了一本长达 65536 行的生死簿，里面装着 True（活着）和 False（死了）。
            valid_flat = group_data["valid_mask"].reshape(env_count * group_size * chunks)

            # 原始形状：[4096, 4, 4, 154] （假设 obs_dim 为 154）。
            # .reshape(..., -1) 后：变成了 [65536, 154] 的二维矩阵。
            #[valid_flat] 筛选后（最终形状）：[50000, 154]
            #含义：把原本按机器人、路线排布的照片墙打散，丢掉摔倒后的画面，剩下 50000 张纯净的有效观测照片。
            obs_flat = group_data["obs"].reshape(env_count * group_size * chunks, -1)[valid_flat]

            # 原始形状：[4096, 4, 4, 5, 116] （假设 Flow 去噪有 4 步，那就是 5 个中间状态；chunk_dim 为 116，也就是 action_dim*horizon = 29*4 = 116）。
            # .reshape 后：变成了 [65536, 5, 116] 的三维矩阵。
            # [valid_flat] 筛选后（最终形状）：[50000, 5, 116]
            # 含义：50000 个有效动作，以及生成每个动作时脑海里经历的 5 个中间去噪状态。
            latents_flat = group_data["latents"].reshape(env_count * group_size * chunks, steps + 1, -1)[valid_flat]

            # 原始形状：[4096, 4, 4, 4] （最后的 4 代表去噪 4 步产生的 4 个对数概率）。
            # .reshape 后：变成了 [65536, 4] 的二维矩阵。
            # [valid_flat] 筛选后（最终形状）：[50000, 4]
            # 含义：这 50000 个有效动作，在当时走那 4 步去噪流程时，旧大脑给出的信心概率。
            log_probs_flat = group_data["log_probs"].reshape(env_count * group_size * chunks, train_steps)[valid_flat]

            #原始形状：[4096, 4, 4] （GRPO 算出来的相对优势分数）。
            # .reshape 后：变成了 [65536] 的一维张量。
            # [valid_flat] 筛选后（最终形状）：[50000]
            # 含义：对应那 50000 个有效动作的最终“绩效考评得分”。
            adv_flat = advantages.reshape(env_count * group_size * chunks)[valid_flat]

            # 拿着刚才洗干净的 obs_flat、相对优势 adv_flat 等核心数据，送进 _policy_update 里。反向传播修改权重
            t1 = time.perf_counter()
            update_metrics = self._policy_update(
                obs_flat,
                latents_flat,
                log_probs_flat,
                adv_flat,
                group_data["sigma_schedule"],
            )
            update_time = time.perf_counter() - t1
            metrics = self._build_metrics(group_data, advantages, update_metrics, collect_time, update_time)

            # 每隔一定次数（validation_every），让机器人停止探索，关掉随机噪声（使用确定性策略），真刀真枪地跑一次测试。
            # fixed_seed=42 是为了保证每次考试的考卷（环境随机种子）是一样的，这样才能公平对比模型是不是真的变聪明了。
            if self.cfg.validation_every > 0 and update_idx % self.cfg.validation_every == 0:
                metrics.update(self.run_validation_rollout())
                fixed_metrics = self.run_validation_rollout(fixed_seed=42)
                for key, value in fixed_metrics.items():
                    metrics[key.replace("validation/", "val_fixed/")] = value

            if update_idx % self.cfg.log_every == 0:
                self._log_update(update_idx, metrics)

            if self.checkpoint_dir is not None and (
                update_idx == self.cfg.max_updates
                or (self.cfg.save_every > 0 and update_idx % self.cfg.save_every == 0)
            ):
                self._save_checkpoint(update_idx, metrics)

            # 如果在考试中，机器人的表现达到了你在配置文件里设定的目标，系统就会判断它“神功大成”。
            # 此时会立刻保存一个带有 success 名字的权重文件，并且直接跳出主循环（break），提前结束训练
            if self._target_validation_reached(metrics):
                if self.checkpoint_dir is not None:
                    self._save_checkpoint(update_idx, metrics, filename=self.cfg.success_checkpoint_name)
                print(
                    f"[SUCCESS] validation reached {self.cfg.target_validation_steps} steps; "
                    f"saved {self.cfg.success_checkpoint_name}",
                    flush=True,
                )
                break

        self.current_observation = self._reset_training_envs()
        print("[INFO] Training finished.", flush=True)

    def _build_metrics(
        self,
        group_data: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        update_metrics: dict[str, float],
        collect_time: float,
        update_time: float,
    ) -> dict[str, float]:
        group_rewards = group_data["rewards"]
        metric_chunk_return = group_data["metric_chunk_return"]
        metric_actions = group_data["metric_actions"]
        act_abs_tensor = metric_actions.abs()

        metric_actions_first = group_data.get("metric_actions_first")
        metric_actions_last = group_data.get("metric_actions_last")
        act_first_mean = metric_actions_first.abs().mean().item() if metric_actions_first is not None else 0.0
        act_last_mean = metric_actions_last.abs().mean().item() if metric_actions_last is not None else 0.0


        act_abs = act_abs_tensor.mean(dim=(0, 1))
        final_latents = group_data["latents"][..., -1, :]
        valid_mask = group_data["valid_mask"]
        metrics = {
            **update_metrics,
            "group/reward_mean": float(group_rewards.mean().item()),
            "group/reward_std": float(group_rewards.std().item()),
            "group/reward_min": float(group_rewards.min().item()),
            "group/reward_max": float(group_rewards.max().item()),
            "group/advantage_abs_mean": float(advantages.abs().mean().item()),
            "rollout/valid_frac": float(valid_mask.float().mean().item()),
            "rollout/chunk_return_mean": float(metric_chunk_return.mean().item()),
            "rollout/chunk_return_std": float(metric_chunk_return.std().item()),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
            "act/abs_mean": float(act_abs_tensor.mean().item()),
            "act/abs_max": float(act_abs_tensor.max().item()),
            "act/abs_p95": float(torch.quantile(act_abs_tensor.flatten(), 0.95).item()),
            "act/legs_abs": float(act_abs[[0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]].mean().item()),
            "act/waist_abs": float(act_abs[[2, 5, 8]].mean().item()),
            "act/arms_abs": float(act_abs[[11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]].mean().item()),
            "latent/final_abs_mean": float(final_latents.abs().mean().item()),
            "latent/final_abs_max": float(final_latents.abs().max().item()),
            "act/l_shoulder_pitch": float(act_abs[11].item()),
            "act/r_shoulder_pitch": float(act_abs[12].item()),
            "act/l_shoulder_roll": float(act_abs[15].item()),
            "act/r_shoulder_roll": float(act_abs[16].item()),
            "act/l_shoulder_yaw": float(act_abs[19].item()),
            "act/r_shoulder_yaw": float(act_abs[20].item()),
            "act/l_elbow": float(act_abs[21].item()),
            "act/r_elbow": float(act_abs[22].item()),
            "act/l_wrist_roll": float(act_abs[23].item()),
            "act/r_wrist_roll": float(act_abs[24].item()),
            "act/l_wrist_pitch": float(act_abs[25].item()),
            "act/r_wrist_pitch": float(act_abs[26].item()),
            "act/l_wrist_yaw": float(act_abs[27].item()),
            "act/r_wrist_yaw": float(act_abs[28].item()),
            "act/first_abs_mean": act_first_mean,
            "act/last_abs_mean": act_last_mean,
        }
        done_union: dict[str, torch.Tensor] = {}
        for step_info in group_data["metric_infos_list"]:
            for key, value in step_info["reward_terms"].items():
                metric_key = f"reward/{key}_mean"
                metrics[metric_key] = metrics.get(metric_key, 0.0) + float(value.mean().item()) / self.cfg.horizon
            for key, value in step_info["done_terms"].items():
                if key not in done_union:
                    done_union[key] = value.bool().clone()
                else:
                    done_union[key] |= value.bool()
        for key, union_mask in done_union.items():
            metrics[f"done/{key}_frac"] = float(union_mask.float().mean().item())
        reward_weights = {
            "joint_acc": -2.5e-7,
            "joint_torque": -1.0e-5,
            "action_rate": -1.0e-1,
            "joint_limit": -10.0,
            "anchor_pos_reward": 0.5,
            "anchor_ori_reward": 0.5,
            "body_pos_reward": 1.0,
            "body_ori_reward": 1.0,
            "body_lin_vel_reward": 1.0,
            "body_ang_vel_reward": 1.0,
            "undesired_contacts": -0.1,
        }
        weighted_positive = 0.0
        weighted_penalty = 0.0
        for reward_name, weight in reward_weights.items():
            raw_key = f"reward/{reward_name}_mean"
            if raw_key not in metrics:
                continue
            contribution = weight * metrics[raw_key] * self.env.dt
            metrics[f"reward_weighted/{reward_name}"] = contribution
            if contribution >= 0.0:
                weighted_positive += contribution
            else:
                weighted_penalty += contribution
        metrics["reward_weighted/positive"] = weighted_positive
        metrics["reward_weighted/penalty"] = weighted_penalty
        metrics["reward_weighted/total"] = weighted_positive + weighted_penalty
        return metrics
