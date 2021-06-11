import time

import torch
from torch.utils.tensorboard import SummaryWriter
import gym
import torch.optim as optim
import torch.nn as nn
import rospy
from tqdm import tqdm

from ppo.config import AgentConfig
from ppo.network import MlpPolicy


class PPOAgent:
    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        gamma: float,
        plot_every: int,
        update_freq: int,
        k_epoch: int,
        learning_rate: float,
        lmbda: float,
        eps_clip: float,
        v_coef: int,
        entropy_coef: float,
        checkpoint_interval: int
    ):
        self.env = env
        self.k_epoch = k_epoch
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.v_coef = v_coef
        self.entropy_coef = entropy_coef
        self.plot_every = plot_every
        self.update_freq = update_freq
        self.gamma = gamma
        self.memory_size = memory_size
        self.checkpoint_interval = checkpoint_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = self.env.action_space.n
        self.policy_network = MlpPolicy(action_size=self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.k_epoch, gamma=0.999)
        self.loss = 0
        self.criterion = nn.MSELoss()
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': [], 'count': 0,
            'advantage': [], 'td_target': torch.FloatTensor([]).to(self.device)
        }
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
        self.inference_writer = SummaryWriter("inferenceLogs/" + run_timestamp)
        self.CHECKPOINT_PATH = "saved_models"


    def new_random_game(self):
        self.env.reset()
        action = self.env.action_space.sample()
        screen, reward, terminal, info, stability_iterator = self.env.step(action)
        return screen, reward, action, terminal


    def train(self):
        episode = 0
        step = 0
        reward_history = []
        avg_reward = []
        solved = False

        # A new episode
        with tqdm(total=50000) as pbar:
            while not solved:
                start_step = step
                episode += 1
                episode_length = 0


                # Get initial state
                state, reward, action, terminal = self.new_random_game()
                current_state = state
                total_episode_reward = 1
            
                # A step in an episode
                while not solved:
                    step += 1
                    episode_length += 1
                    stability_iterator=0

                    # Choose action
                    prob_a = self.policy_network.pi(torch.FloatTensor(current_state).to(self.device))
                    # print(prob_a)
                    action = torch.distributions.Categorical(prob_a).sample().item()

                    # Act
                    state, reward, terminal, _, stability_iterator = self.env.step(action)
                    new_state = state
                    reward = -1 if terminal else reward
                    self.writer.add_scalar('epsilon', state[6], step)
                    self.add_memory(current_state, action, reward/10.0, new_state, terminal, prob_a[action].item())

                    current_state = new_state
                    total_episode_reward += reward
                    """
                    if terminal or total_episode_reward > 100:
                    """
                    #V2 similar to mountaincar
                    #If 
                    #A: The joint diverged (remained immobile far from the target for too long)
                    #B: Took too long 
                    #C: Remained close enough to the target for an extended period of time 
                    #Then the episode is over. 
                    if terminal or total_episode_reward < -200.0 or stability_iterator>=50:
                        episode_length = step - start_step
                        reward_history.append(total_episode_reward)
                        avg_reward.append(sum(reward_history[-10:])/10.0)
                        rospy.loginfo(f"epsilon : {state[6]} \n epsilon_p : {state[7]}")
                        self.finish_path(episode_length)
                        if len(reward_history) > 100:
                            self.writer.add_scalar('global_avg_score', sum(reward_history[-100:-1]) / 100, episode)
                        if len(reward_history) > 100 and sum(reward_history[-100:-1]) / 100 >= 95:
                            solved = True

                        rospy.loginfo('episode: %.2f, total step: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, '
                            'loss: %.4f, lr: %.4f, stability: %.4f' % (episode, step, episode_length, total_episode_reward, self.loss,
                                                        self.scheduler.get_lr()[0],stability_iterator))
                        rospy.loginfo(terminal)
                        time.sleep(2)
                        self.env.reset()

                        if step > 50000:
                            solved = True

                        break

                    pbar.update(1)

                if episode % self.update_freq == 0:
                    for _ in range(self.k_epoch):
                        self.update_network()

                if episode % self.plot_every == 0:
                    self.plot_graph(reward_history, avg_reward, episode)

                if episode % self.checkpoint_interval == 0:
                    # checkpoint state dict of model
                    torch.save(self.policy_network.state_dict(), self.CHECKPOINT_PATH)
                    self.previous_checkpointed_loss = self.loss

        self.env.close()


    def predict(self, model, order, step, state=None):
        if state is None:
            # Get initial state
            state, reward, action, terminal = self.new_random_game()
        # Choose action
        prob_a = model.pi(torch.FloatTensor(state).to(self.device))
        # print(prob_a)
        action = torch.distributions.Categorical(prob_a).sample().item()
        new_state, new_reward, new_action, _, _ = self.env.step(action)

        # plotting data
        self.inference_writer.add_scalar('epsilon', new_state[6], step)

        return (new_state, new_reward, new_action, step+1)


    def update_network(self):
        # get ratio
        pi = self.policy_network.pi(torch.FloatTensor(self.memory['state']).to(self.device))
        new_probs_a = torch.gather(pi, 1, torch.tensor(self.memory['action']).to(self.device)).to(self.device)
        old_probs_a = torch.FloatTensor(self.memory['action_prob']).to(self.device)
        ratio = (torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))).to(self.device)

        # surrogate loss
        surr1 = ratio * torch.FloatTensor(self.memory['advantage']).to(self.device)
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip).to(self.device) * torch.FloatTensor(self.memory['advantage']).to(self.device)
        pred_v = self.policy_network.v(torch.FloatTensor(self.memory['state']).to(self.device))
        v_loss = 0.5 * (pred_v - self.memory['td_target']).pow(2).to(self.device)  # Huber loss
        entropy = torch.distributions.Categorical(pi).entropy().to(self.device)
        entropy = torch.tensor([[e] for e in entropy]).to(self.device)
        self.loss = (-torch.min(surr1, surr2) + self.v_coef * v_loss - self.entropy_coef * entropy).mean()

        self.optimizer.zero_grad()
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 10.0)
        
        self.optimizer.step()
        self.scheduler.step()


    def add_memory(self, s, a, r, next_s, t, prob):
        if self.memory['count'] < self.memory_size:
            self.memory['count'] += 1
        else:
            self.memory['state'] = self.memory['state'][1:]
            self.memory['action'] = self.memory['action'][1:]
            self.memory['reward'] = self.memory['reward'][1:]
            self.memory['next_state'] = self.memory['next_state'][1:]
            self.memory['terminal'] = self.memory['terminal'][1:]
            self.memory['action_prob'] = self.memory['action_prob'][1:]
            self.memory['advantage'] = self.memory['advantage'][1:]
            self.memory['td_target'] = self.memory['td_target'][1:]

        self.memory['state'].append(s)
        self.memory['action'].append([a])
        self.memory['reward'].append([r])
        self.memory['next_state'].append(next_s)
        self.memory['terminal'].append([1 - t])
        self.memory['action_prob'].append(prob)


    def finish_path(self, length):
        state = self.memory['state'][-length:]
        reward = self.memory['reward'][-length:]
        next_state = self.memory['next_state'][-length:]
        terminal = self.memory['terminal'][-length:]

        td_target = torch.FloatTensor(reward).to(self.device) + self.gamma * self.policy_network.v((torch.FloatTensor(next_state) * torch.FloatTensor(terminal)).to(self.device))
        delta = td_target - self.policy_network.v(torch.FloatTensor(state).to(self.device))
        delta = delta.detach().cpu().numpy()

        # get advantage
        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        advantages.reverse()

        if self.memory['td_target'].shape == torch.Size([1, 0]):
            self.memory['td_target'] = td_target.data
        else:
            self.memory['td_target'] = torch.cat((self.memory['td_target'], td_target.data), dim=0).to(self.device)
        self.memory['advantage'] += advantages


    def plot_graph(self, reward_history, avg_reward, episode):
        # df = pd.DataFrame({'x': range(len(reward_history)), 'Reward': reward_history, 'Average': avg_reward})
        # plt.style.use('seaborn-darkgrid')
        # palette = plt.get_cmap('Set1')

        # plt.plot(df['x'], df['Reward'], marker='', color=palette(1), linewidth=0.8, alpha=0.9, label='Reward')
        # # plt.plot(df['x'], df['Average'], marker='', color='tomato', linewidth=1, alpha=0.9, label='Average')

        # # plt.legend(loc='upper left')
        # plt.title("CartPole", fontsize=14)
        # plt.xlabel("episode", fontsize=12)
        # plt.ylabel("score", fontsize=12)

        # plt.savefig('score.png')
        self.writer.add_scalar('loss', self.loss.detach().cpu().item(), episode)

        
