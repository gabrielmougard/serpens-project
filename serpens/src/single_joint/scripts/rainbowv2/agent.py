from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import rospy

from rainbowv2.network import Network
#from rainbow.tensorboard import RainbowTensorBoard
from cpprb import ReplayBuffer, PrioritizedReplayBuffer
import time
from torch.utils.tensorboard import SummaryWriter


class RainbowAgent:
    """Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
        # Convergence parameters
        convergence_window: int = 100,
        convergence_window_epsilon_p: int = 10, 
        convergence_avg_score: float = 195.0,
        convergence_avg_epsilon: float = 0.0524, # 3 degs converted to rads
        convergence_avg_epsilon_p: float = 0.0174, # 1 deg/s converted to rad/s
        # Tensorboard parameters
        model_name: str = "snake_joint",

    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed

        #produces a unique timestamp for each run 
        run_timestamp=str(
        #returns number of day and number of month
        str(time.localtime(time.time())[2]) + "_" +
        str(time.localtime(time.time())[1]) + "_" +
        #returns hour, minute and second
        str(time.localtime(time.time())[3]) + "_" +
        str(time.localtime(time.time())[4]) + "_" +
        str(time.localtime(time.time())[5])
        )

        #Will write scalars that can be visualized using tensorboard in the directory "runLogs/timestamp"
        self.writer = SummaryWriter("runLogs/" + run_timestamp)


        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            memory_size,
            {
                "obs": {"shape": (obs_dim,)},
                "act": {"shape": (1,)},
                "rew": {},
                "next_obs": {"shape": (obs_dim,)},
                "done": {}
            },
            alpha=alpha    
        )
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                memory_size,
                {
                    "obs": {"shape": (obs_dim,)},
                    "act": {"shape": (1,)},
                    "rew": {},
                    "next_obs": {"shape": (obs_dim,)},
                    "done": {}
                },
                Nstep={
                    "size": n_step,
                    "gamma": gamma,
                    "rew": "rew",
                    "next": "next_obs"
                }
            )
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(),0.0001)

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

        # Custom tensorboard object
        # self.tensorboard = RainbowTensorBoard(
        #     log_dir="single_joint_logs/{}-{}".format(
        #         model_name,
        #         datetime.now().strftime("%m-%d-%Y-%H_%M_%S")
        #     )
        # )
        # Convergence criterion
        self.convergence_window = convergence_window
        self.convergence_window_epsilon_p = convergence_window_epsilon_p
        self.convergence_avg_score = convergence_avg_score 
        self.convergence_avg_epsilon = convergence_avg_epsilon
        self.convergence_avg_epsilon_p = convergence_avg_epsilon_p


    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:

            self.transition = [state, selected_action]
        

        return selected_action


    def step(self, action: np.ndarray, score:int) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action,score)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                idx = self.memory_n.add(
                    **dict(
                        zip(["obs", "act", "rew", "next_obs", "done"], self.transition)
                    )
                )
                one_step_transition = [ v[idx] for _,v in self.memory_n.get_all_transitions().items()] if idx else None

            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.add(
                    **dict(
                        zip(["obs", "act", "rew", "next_obs", "done"], one_step_transition)
                    )
                )
    
        return next_state, reward, done


    def update_model(self,frame_idx:int) -> torch.Tensor:
        """Update the model by gradient descent.
        shape of elementwise_loss = [128,51]
        shape of loss = ([])
        shape of weights ([128,1)]
        """
        # PER needs beta to calculate weights
        samples = self.memory.sample(self.batch_size, beta=self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indexes"]
        #rospy.loginfo(samples.keys())
        #rospy.loginfo(weights.shape)
        #rospy.loginfo(indices.shape())

        #torch.save(self.dqn.state_dict(),str("checkpoint_"+str(time.time())))
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        self.writer.add_scalar('update_model/Lossv0', loss.detach().item(),frame_idx )
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = {k: [v[i] for i in indices] for k,v in self.memory_n.get_all_transitions().items()}
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            #rospy.loginfo(elementwise_loss_n_loss.shape)
            #rospy.loginfo(elementwise_loss.shape)

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)


        self.optimizer.zero_grad()
        self.writer.add_scalar('update_model/Lossv1', loss.detach().item(),frame_idx )
        #From pytorch doc: backward() Computes the gradient of current tensor w.r.t. graph leaves.
        #self.writer.add_image("loss gradient before", loss, frame_idx)
        loss.backward()
        #self.writer.add_image("loss gradient after", loss, frame_idx)
        self.writer.add_scalar('update_model/Lossv2', loss.detach().item(),frame_idx )
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()
        
        #rospy.loginfo("second")
        #rospy.loginfo(loss.shape)

        #rospy.loginfo("loss dimension = " + loss.ndim()  )   
        #rospy.loginfo("loss = " + str(loss.detach().item()) + "type = " + str(type(loss.detach().item())  )   )   
        self.writer.add_scalar('update_model/Loss', loss.detach().item(),frame_idx )
        return loss.detach().item()


    def train(self, num_frames: int):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        score = 0

        for frame_idx in tqdm(range(1, num_frames + 1)):

            action = self.select_action(state)
            next_state, reward, done = self.step(action,score)

            state = next_state
            score += reward
            
            # NoisyNet: removed decrease of epsilon
            
            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                #rospy.loginfo("logging for done")
                self.writer.add_scalar('train/score', score, frame_idx)
                self.writer.add_scalar('train/final_epsilon', state[6], frame_idx)
                self.writer.add_scalar('train/epsilon_p', state[7], frame_idx)
                state = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if self.memory.get_stored_size() >= self.batch_size:
                #frame_id given as argument for logging by self.writer. 
                #rospy.loginfo("frame_idx= " + str(frame_idx) + "type = " + str(type(frame_idx)))
                loss = self.update_model(frame_idx)

                losses.append(loss)
                update_cnt += 1
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update(loss)

        self.env.close()


    def test(self) -> List[np.ndarray]:
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        return frames


    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["act"]).to(device)
        reward = torch.FloatTensor(np.array(samples["rew"]).reshape(-1, 1)).to(device)
        done = torch.FloatTensor(np.array(samples["done"]).reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)


        return elementwise_loss


    def _target_hard_update(self,loss):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        #torch.save(self.dqn.state_dict(),str("checkpoint_"+str(time.time())))

        torch.save({
            'model_state_dict': self.dqn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, str("checkpoints/checkpoint_"+str(time.time())))

