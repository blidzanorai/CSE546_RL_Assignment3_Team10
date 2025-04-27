import os
from cv2 import log
import gymnasium as gym
from gymnasium.spaces import Box
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import math
import gymnasium.wrappers as wrappers

ENV_ID = "Ant-v5"

NUM_WORKERS  = 6 
MAX_EPISODES  = 3000 
GAMMA = 0.99
GLOBAL_SMOOTH = 50
EVAL_EPISODES = 10
N_STEPS = 64 
USE_GAE = True
LR = 1e-5 
LAMBDA_GAE = 0.95
CLIP_GRAD_NORM = 0.5
ENTROPY_COEF_START = 0.001 
NORMALIZE_OBS = True # NOT USED HERE
NORMALIZE_ADV = True
USE_NORMALIZE_REWARD_WRAPPER = False 
USE_NORMALIZE_OBS_WRAPPER = False 

def normalize_observation(obs: np.ndarray, env: gym.Env) -> np.ndarray:
    obs_space = env.observation_space
    if not isinstance(obs_space, Box):
        return obs.astype(np.float32)

    low = obs_space.low
    high = obs_space.high

    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        return obs.astype(np.float32)
    if np.any((high - low) <= 1e-8):
        return obs.astype(np.float32)

    scaled = 2.0 * (obs.astype(np.float32) - low) / (high - low) - 1.0
    return np.clip(scaled, -1.0, 1.0)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space):
        super().__init__()
        self.action_space = action_space
        self.is_continuous = isinstance(action_space, Box)

        self.fc_base = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.v = nn.Linear(256, 1)

       
        if self.is_continuous:
            action_dim = action_space.shape[0]
            self.mu = nn.Linear(256, action_dim)  
            self.log_std = nn.Parameter(torch.ones(action_dim) * -2.0) 
            
            nn.init.orthogonal_(self.mu.weight, 0.01)
            nn.init.constant_(self.mu.bias, 0.0)
        else:
            action_dim = action_space.n
            self.pi = nn.Linear(256, action_dim)      
            nn.init.orthogonal_(self.pi.weight, 0.01)
            nn.init.constant_(self.pi.bias, 0.0)

        
        for layer in self.fc_base:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.constant_(self.v.bias, 0.0)

    def forward(self, x):
        
        features = self.fc_base(x)
        value = self.v(features)
        if self.is_continuous:
            mu = self.mu(features)
            log_std_clamped = torch.clamp(self.log_std, -20, 2)
            std = torch.exp(log_std_clamped)
            dist = torch.distributions.Normal(mu, std) 
        else:
            logits = self.pi(features)
            dist = torch.distributions.Categorical(logits=logits) 

        return dist, value

class A3CWorker(mp.Process):
    def __init__(self, wid, env_id, model, optimizer, counter, queue, num_steps, max_episodes, gamma, clip_grad_norm, use_gae, lambda_gae, entropy_coef_start, normalize_obs, normalize_adv, initial_lr):
        super().__init__()
        self.wid = wid
        self.env_id = env_id
        self.model = model
        self.optimizer = optimizer
        self.counter = counter
        self.queue = queue

        self.num_steps = num_steps
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.clip_grad_norm = clip_grad_norm
        self.use_gae = use_gae
        self.lambda_gae = lambda_gae
        self.entropy_coef_start = entropy_coef_start
        self.normalize_obs = normalize_obs
        self.normalize_adv = normalize_adv
        self.initial_lr = initial_lr
        self.local_rewards_deque = deque(maxlen=GLOBAL_SMOOTH // NUM_WORKERS + 1)
        print(f"---- [Worker {self.wid}] Initialized (PID: {os.getpid()}) ----")

    def run(self):
        print(f"---- [Worker {self.wid}] Starting Environment: {self.env_id} ----")
        if self.env_id == "Ant-v5":
             try:
                 env = gym.make(self.env_id, render_mode='rgb_array')
             except Exception:
                 print(f"---- [Worker {self.wid}] Failed to create Ant-v5 with rgb_array, trying without render_mode ----")
                 env = gym.make(self.env_id)
        else:
            env = gym.make(self.env_id)
        if USE_NORMALIZE_OBS_WRAPPER:
             print(f"---- [Worker {self.wid}] Applying NormalizeObservation wrapper ----")
             env = wrappers.NormalizeObservation(env)
        if USE_NORMALIZE_REWARD_WRAPPER:
             print(f"---- [Worker {self.wid}] Applying NormalizeReward wrapper ----")
             
             env = wrappers.NormalizeReward(env, gamma=self.gamma)
        self.is_continuous = isinstance(env.action_space, Box)
        if self.is_continuous:
             self.action_low = env.action_space.low
             self.action_high = env.action_space.high


        state, _ = env.reset()
        # if self.normalize_obs:
        #     state = normalize_observation(state, env)

        ep_reward = 0.0
        steps_in_ep = 0
        local_ep_count = 0

        while True:
            with self.counter.get_lock():
                if self.counter.value >= self.max_episodes:
                    print(f"---- [Worker {self.wid}] Max episodes reached ({self.counter.value}). Terminating. ----")
                    break

            traj = self._collect_trajectory(env, state)

            loss, next_state, ep_reward_update, done = self._compute_loss(traj)
            state = next_state
            ep_reward += ep_reward_update
            steps_in_ep += len(traj["rewards"])
            with self.counter.get_lock(): 
                current_episode = self.counter.value

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            if done:
                local_ep_count += 1
                self.local_rewards_deque.append(ep_reward)
        
                if local_ep_count % (10 if ENV_ID == "LunarLander-v3" else 50) == 0:
                    avg_reward_last = np.mean(self.local_rewards_deque)
                    print(f"---- [Worker {self.wid}] Local Ep Cnt: {local_ep_count} | Avg Reward (Last 10): {avg_reward_last:.2f} ----")

                with self.counter.get_lock():
                    global_ep_count = self.counter.value
                    self.counter.value += 1
                self.queue.put((self.wid, ep_reward, global_ep_count + 1))
                ep_reward = 0.0
                steps_in_ep = 0
                state, _ = env.reset()
                # if self.normalize_obs:
                #     state = normalize_observation(state, env)

        print(f"---- [Worker {self.wid}] Closing environment and exiting run loop. ----")
        env.close()

    def _collect_trajectory(self, env, current_state):
        log_probs, values, rewards, entropies, dones = [], [], [], [], []
        state = current_state
        done = False
        terminated = False
        truncated = False

        for step in range(self.num_steps):
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0) 

            dist, value = self.model(s_tensor)
            entropy = dist.entropy().mean() 

            if self.is_continuous:
                 action_raw = dist.sample()
                 log_prob = dist.log_prob(action_raw).sum(axis=-1) 
                 
                 action_clipped = torch.clamp(action_raw,
                                             torch.as_tensor(self.action_low, dtype=torch.float32),
                                             torch.as_tensor(self.action_high, dtype=torch.float32))
                 action_env = action_clipped.squeeze(0).numpy()
            else:
                 action = dist.sample()
                 log_prob = dist.log_prob(action)
                 action_env = action.item()


            next_state, r, terminated, truncated, _ = env.step(action_env)
            done = terminated or truncated

            # if self.normalize_obs:
            #     next_state = normalize_observation(next_state, env)

            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(r)
            entropies.append(entropy)
            dones.append(done)

            state = next_state
            if done:
                break

        return {"log_probs": log_probs, "values": values, "rewards": rewards,
                "entropies": entropies, "dones": dones, "next_state": state}


    def _compute_loss(self, traj):
        rewards = traj["rewards"]
        values = traj["values"]
        log_probs = traj["log_probs"]
        entropies = traj["entropies"]
        dones = traj["dones"]
        next_state = traj["next_state"]
        is_final_done = dones[-1]


        R = 0.0
        if not is_final_done:
            s_tensor = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, next_val = self.model(s_tensor)
            R = next_val.squeeze(0).item()


        returns = []
        advantages = []
        gae = 0.0
        values_np = torch.stack(values).detach().numpy().flatten()
        rewards_np = np.array(rewards, dtype=np.float32)

        for i in reversed(range(len(rewards))):
            
            R = rewards_np[i] + self.gamma * R * (1 - dones[i]) # If done, next R is 0
            returns.insert(0, R)

            # THIS code is to Calculate GAE advantage, only used to improve Ant Performance
            if self.use_gae:
                 next_v = values_np[i+1] if i < len(rewards) - 1 else R 
                 delta = rewards_np[i] + self.gamma * next_v * (1 - dones[i]) - values_np[i]
                 gae = delta + self.gamma * self.lambda_gae * gae * (1 - dones[i])
                 advantages.insert(0, gae)

        returns = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(values).squeeze(1) # Ensure [T] shape
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        if not self.use_gae:
            advantages = returns - values_t
        else:
            advantages = torch.tensor(advantages, dtype=torch.float32)

        if self.normalize_adv:
             if advantages.numel() > 1:
                 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
             elif advantages.numel() == 1: 
                 advantages = advantages - advantages.mean()
             else: 
                  advantages = torch.zeros_like(returns)


        policy_loss = -(log_probs_t * advantages.detach()).mean()
        value_loss = 0.5 * F.mse_loss(values_t, returns)

        
        
        current_entropy_coef = self.entropy_coef_start # Linear decay, no floor
        entropy_loss = entropies_t.mean()
        total_loss = policy_loss + value_loss - current_entropy_coef * entropy_loss

        ep_reward_update = sum(rewards)

        return total_loss, next_state, ep_reward_update, is_final_done


class A3CTrainer:
    def __init__(self, env_id, num_workers, num_steps, max_episodes, gamma, lr, clip_grad_norm, use_gae, lambda_gae, entropy_coef_start, normalize_obs, normalize_adv):
        print(f"====== Initializing A3CTrainer for {env_id} ======")
        self.env_id = env_id
        self.num_workers = num_workers
        self.num_steps = num_steps
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm
        self.use_gae = use_gae
        self.lambda_gae = lambda_gae
        self.entropy_coef_start = entropy_coef_start
        self.normalize_obs = normalize_obs
        self.normalize_adv = normalize_adv
        # self.initial_lr = initial_lr

        print(f"---- Creating temporary environment to get specs ----")
        if self.env_id == "Ant-v5":
             try:
                 temp_env = gym.make(self.env_id, render_mode='rgb_array')
             except Exception:
                 temp_env = gym.make(self.env_id)
        else:
             temp_env = gym.make(self.env_id)

        obs_dim = temp_env.observation_space.shape[0]
        self.action_space = temp_env.action_space
        self.is_continuous = isinstance(self.action_space, Box)
        if self.is_continuous:
            act_dim_str = f"Box{self.action_space.shape}"
        else:
            act_dim_str = f"Discrete({self.action_space.n})"

        self.reward_threshold = temp_env.spec.reward_threshold if temp_env.spec else None # Check if spec exists
        print(f"---- Env Specs: Obs Dim={obs_dim}, Act Space={act_dim_str}, Reward Threshold={self.reward_threshold} ----")
        print(f"---- Using GAE: {self.use_gae}, Normalize Obs: {self.normalize_obs}, Normalize Adv: {self.normalize_adv} ----")
        temp_env.close()


        print(f"---- Creating Global Actor-Critic Model ----")
        self.model = ActorCritic(obs_dim, self.action_space)
        self.model.share_memory()

        print(f"---- Creating Shared Adam Optimizer ----")
        self.optimizer = SharedAdam(self.model.parameters(), lr=self.lr)

        self.counter = mp.Value('i', 0)
        self.queue = mp.Queue()

        print(f"====== A3CTrainer Initialized ======")


    def train(self):
        start_time = time.time()
        print(f"\n====== Starting Training with {self.num_workers} Workers ======")
        workers = [A3CWorker(i, self.env_id, self.model, self.optimizer,
                             self.counter, self.queue, self.num_steps,
                             self.max_episodes, self.gamma, self.clip_grad_norm,
                             self.use_gae, self.lambda_gae, self.entropy_coef_start,
                             self.normalize_obs, self.normalize_adv, self.lr)
                   for i in range(self.num_workers)]

        for w in workers:
            w.start()
            print(f"---- Worker {w.wid} started ----")

        rewards = []
        worker_rewards = {i: [] for i in range(self.num_workers)}
        last_print_time = time.time()

        while True:
            with self.counter.get_lock():
                current_global_ep = self.counter.value
            if current_global_ep >= self.max_episodes:
                print(f"\n====== Max episodes ({self.max_episodes}) reached based on counter. Stopping training. ======")
                break

            wid, ep_r, global_ep_num = self.queue.get()
            rewards.append(ep_r)
            worker_rewards[wid].append(ep_r)
            num_completed_eps = len(rewards)

            current_time = time.time()
           
            if current_time - last_print_time > 30.0:
                 avg_reward_last_100 = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards) if rewards else 0.0
                 elapsed_time = current_time - start_time
                 eps_per_sec = num_completed_eps / elapsed_time if elapsed_time > 0 else 0
                 print(f"---- Progress: Ep {global_ep_num}/{self.max_episodes} | "
                       f"Avg R (L100): {avg_reward_last_100:.2f} | "
                       f"Total Eps: {num_completed_eps} | "
                       f"Eps/Sec: {eps_per_sec:.2f} | "
                       f"Time: {elapsed_time:.1f}s ----")
                 last_print_time = current_time

            
            if self.reward_threshold is not None and len(rewards) >= 100 and num_completed_eps % 50 == 0 :
                 avg_r_100 = np.mean(rewards[-100:])
                 if ENV_ID == "Ant-v5": self.reward_threshold = 1000
                 if avg_r_100 >= self.reward_threshold:
                     with self.counter.get_lock():
                         
                         self.counter.value = self.max_episodes
                     print(f"\n====== Environment Solved! Avg Reward (Last 100) = {avg_r_100:.2f} >= Threshold ({self.reward_threshold}) ======")
                     print(f"====== Initiating Early Stopping... ======")
                     
                     break

        print(f"\n====== Training loop finished. Waiting for workers to terminate... ======")
        
        while not self.queue.empty():
            self.queue.get()
            time.sleep(0.1)

        for w in workers:
            w.join(timeout=10) 
            if w.is_alive():
                print(f"---- Worker {w.wid} did not join cleanly, attempting terminate ----")
                w.terminate()
            print(f"---- Worker {w.wid} joined ----")

        total_training_time = time.time() - start_time
        print(f"====== All workers finished. Total Training Time: {total_training_time:.2f} seconds ======")

        safe_env_id = self.env_id.replace("/", "_") 
        model_filename = f"a3c_{safe_env_id}.pth"
        print(f"---- Saving global model weights to {model_filename} ----")
        torch.save(self.model.state_dict(), model_filename)

        plot_filename_global = f"a3c_{safe_env_id}_global_rewards.png"
        print(f"---- Plotting global rewards to {plot_filename_global} ----")
        self._plot_and_save(rewards, plot_filename_global,
                            title=f"A3C {self.env_id} Global Training Rewards", smooth=GLOBAL_SMOOTH)

        plot_filename_workers = f"a3c_{safe_env_id}_workers_rewards.png"
        print(f"---- Plotting individual worker rewards to {plot_filename_workers} ----")
        self._plot_worker_rewards(worker_rewards, plot_filename_workers)

        return rewards, self.model


    def evaluate(self, n_episodes=EVAL_EPISODES):
        print(f"\n====== Starting Evaluation Phase ({n_episodes} episodes) ======")
        safe_env_id = self.env_id.replace("/", "_")
        eval_render_mode = "human" if self.env_id == "LunarLander-v3" else "rgb_array" # Avoid human render for Mujoco headless
        try:
             eval_env = gym.make(self.env_id, render_mode=eval_render_mode)
        except Exception:
             print(f"---- Could not create eval env with render_mode='{eval_render_mode}', trying without. ----")
             eval_env = gym.make(self.env_id)
        if USE_NORMALIZE_OBS_WRAPPER:
            print(f"---- [Evaluate] Applying NormalizeObservation wrapper ----")
            
            eval_env = wrappers.NormalizeObservation(eval_env)


        self.model.eval() 
        eval_rewards = []

        if self.is_continuous:
            action_low = eval_env.action_space.low
            action_high = eval_env.action_space.high


        for i in range(n_episodes):
            s, _ = eval_env.reset()
            # if self.normalize_obs:
            #     s = normalize_observation(s, eval_env)
            done = False
            terminated = False
            truncated = False
            total_reward = 0
            step_count = 0
            print(f"---- Eval Episode {i+1}/{n_episodes} Starting ----")
            while not done:
                step_count += 1
                with torch.no_grad():
                    s_tensor = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
                    dist, _ = self.model(s_tensor)

                    if self.is_continuous:
                       
                        action_raw = dist.mean
                        
                        action_clipped = torch.clamp(action_raw,
                                                    torch.as_tensor(action_low, dtype=torch.float32),
                                                    torch.as_tensor(action_high, dtype=torch.float32))
                        action_env = action_clipped.squeeze(0).numpy()
                    else:
                        
                        action = torch.argmax(dist.logits).item()
                        action_env = action


                s, r, terminated, truncated, _ = eval_env.step(action_env)
                # if self.normalize_obs:
                #     s = normalize_observation(s, eval_env)

                done = terminated or truncated
                total_reward += r
                if eval_render_mode == "human":
                     eval_env.render()


            eval_rewards.append(total_reward)
            print(f"---- Eval Episode {i+1}/{n_episodes} Finished. Reward: {total_reward:.2f}, Steps: {step_count} ----")

        eval_env.close()
        print(f"---- Evaluation Environment Closed ----")

        avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0.0
        std_eval_reward = np.std(eval_rewards) if eval_rewards else 0.0
        print(f"====== Evaluation Finished. Average Reward: {avg_eval_reward:.2f} +/- {std_eval_reward:.2f} ======")

        plot_filename_eval = f"a3c_{safe_env_id}_eval_rewards.png"
        print(f"---- Plotting evaluation rewards to {plot_filename_eval} ----")
        self._plot_and_save(eval_rewards, plot_filename_eval, title=f"A3C {self.env_id} Greedy Evaluation", smooth=1)

        return avg_eval_reward


    @staticmethod
    def _plot_and_save(rewards, filename, title="", smooth=1):
        if not rewards:
            print(f"---- No rewards data to plot for {filename} ----")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, alpha=0.6, label="Raw Reward per Episode")
        if smooth > 1 and len(rewards) >= smooth:
            ma = np.convolve(rewards, np.ones(smooth)/smooth, mode='valid')
            x_ma = range(smooth - 1, len(rewards))
            plt.plot(x_ma, ma, lw=2, label=f"{smooth}-Episode Moving Average")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        try:
            plt.savefig(filename)
            print(f"---- Plot saved successfully: {filename} ----")
        except Exception as e:
            print(f"---- Error saving plot {filename}: {e} ----")
        plt.close()

    def _plot_worker_rewards(self, worker_rewards, filename):
        num_workers = len(worker_rewards)
        if num_workers == 0: return

        cols = int(np.ceil(np.sqrt(num_workers)))
        rows = int(np.ceil(num_workers / cols))

        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        axs = axs.flatten()

        max_eps_len = 0
        for wid in range(num_workers):
             rw = worker_rewards.get(wid, [])
             if len(rw) > max_eps_len:
                 max_eps_len = len(rw)


        worker_smooth = max(1, GLOBAL_SMOOTH // num_workers)

        for wid in range(num_workers):
            if wid < len(axs):
                ax = axs[wid]
                rw = worker_rewards.get(wid, [])
                if not rw: continue
                ax.plot(rw, alpha=0.7, label=f'Worker {wid} Raw')
                if len(rw) >= worker_smooth:
                    ma = np.convolve(rw, np.ones(worker_smooth)/worker_smooth, mode='valid')
                    ax.plot(range(worker_smooth-1, len(rw)), ma, lw=2, label=f'{worker_smooth}-ep MA')
                ax.set_title(f'Worker {wid}')
                ax.set_xlabel('Episode (Worker Specific)')
                ax.set_ylabel('Reward')
                ax.grid(True)
                ax.legend(fontsize='small')
                ax.set_xlim(0, max_eps_len) 

        for i in range(num_workers, len(axs)):
             axs[i].set_visible(False)

        safe_env_id = self.env_id.replace("/", "_")
        fig.suptitle(f"A3C {safe_env_id} Individual Worker Training Rewards", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        try:
            fig.savefig(filename)
            print(f"---- Worker plots saved successfully: {filename} ----")
        except Exception as e:
            print(f"---- Error saving worker plots {filename}: {e} ----")
        plt.close(fig)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    print("---- Set multiprocessing start method to 'spawn' ----")

    print(f"\n====== Main Process Started (PID: {os.getpid()}) for Env: {ENV_ID} ======")

    trainer = A3CTrainer(
        env_id=ENV_ID,
        num_workers=NUM_WORKERS,
        num_steps=N_STEPS,
        max_episodes=MAX_EPISODES,
        gamma=GAMMA,
        lr=LR,
        clip_grad_norm=CLIP_GRAD_NORM,
        use_gae=USE_GAE,
        lambda_gae=LAMBDA_GAE,
        entropy_coef_start=ENTROPY_COEF_START,
        normalize_obs=NORMALIZE_OBS,
        normalize_adv=NORMALIZE_ADV
    )

    training_rewards, model = trainer.train()

    trainer.evaluate()

    print(f"\n====== Script Finished ======")
