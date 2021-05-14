import sys
from sys import getsizeof
import gym
from gym import spaces
from gym.utils import seeding
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
import time
from std_srvs.srv import Empty


class SnakeJoint(gym.Env):
    def __init__(self):
        self._observation_msg = None
        self.max_episode_steps = 200
        self.iterator = 0
        self.reset_jnts = True
        self._collision_msg = None
        self.current_torque = 0.0

        # Get configuration parameters
        self.n_actions = rospy.get_param('/rainbow/n_actions')
        self.theta_ld_max = rospy.get_param("/rainbow/theta_ld_max")
        self.theta_l_max = rospy.get_param("/rainbow/theta_l_max")
        self.theta_m_max = rospy.get_param("/rainbow/theta_m_max")
        self.theta_m_p_max = rospy.get_param("/rainbow/theta_m_p_max")
        self.torque_step = rospy.get_param('/rainbow/torque_step')
        self.tau_ext_max = rospy.get_param('/rainbow/tau_ext_max')
        # Variables divergence/convergence conditions
        self.max_allowed_epsilon =  rospy.get_param('/rainbow/max_allowed_epsilon')
        self.max_ep_length =  rospy.get_param('/rainbow/max_ep_length')
        self.min_allowed_epsilon_p =  rospy.get_param('/rainbow/min_allowed_epsilon_p')
        self.max_time_episode = rospy.get_param('/rainbow/max_time_episode')
        self.angle_stability_criterion = rospy.get_param('/rainbow/angle_stability_criterion')
        self.load_speed_criterion = rospy.get_param('/rainbow/load_speed_criterion')

        # publishers and subscribers
        self.torque_pub = rospy.Publisher('/single_joint/link_motor_effort/command', Float64, queue_size=10)
        #For rqt plotting
        self.error_pub = rospy.Publisher('/single_joint/rqt/error', Float64, queue_size=10)
        self.theta_ld_pub = rospy.Publisher('/single_joint/rqt/thetald', Float64, queue_size=10)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        rospy.Subscriber('/single_joint/joint_states', JointState, self.observation_callback)

        self.action_space = spaces.Discrete(self.n_actions)

        boundaries = np.array([
            self.theta_ld_max,
            self.theta_l_max,
            np.finfo(np.float32).max,
            self.theta_m_max,
            self.theta_m_p_max,
            self.tau_ext_max,
            self.theta_ld_max + self.theta_l_max,
            np.finfo(np.float32).max
        ])

        self.observation_space = spaces.Box(
            -boundaries,
            boundaries,
            dtype=np.float32
        )

        self.seed()
        # start the environment server at a refreshing rate of 10Hz
        self.rate = rospy.Rate(10)


    def seed(self, seed=5048795115606990371):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointStates
        """
        self._observation_msg = message


    def take_observation(self):
        """
        Take observation from the environment and return it.
        :return: state.
        """
        obs_message = self._observation_msg
        # Check that the observation is not prior to the action
        # obs_message = self._observation_msg
        
        if obs_message is None: 
            while obs_message is None:
                try:
                    obs_message=rospy.wait_for_message('/single_joint/joint_states', JointState, timeout=5)
                except rospy.ROSInterruptException:
                    rospy.loginfo("jointStates failure")
                    sys.exit(1)


        epsilon = abs(self.episode_theta_ld - obs_message.position[1])
        obs = [
            self.episode_theta_ld,
            obs_message.position[1], # theta_l
            obs_message.velocity[1], # theta_l_p
            obs_message.position[0], # theta_m
            obs_message.velocity[0], # theta_m_p
            self.episode_external_torque,
            epsilon, # epsilon
            (epsilon - self.previous_epsilon) if self.previous_epsilon else np.finfo(np.float32).max # epsilon_p
        ]
        #Set observation to None after it has been read.
        # TODO : CHANGELOG : setting self._observation_msg to None create an infinite loop line 83... How to fix it ?
        #self._observation_msg = None
        # update self.previous_epsilon for the next times
        self.previous_epsilon = epsilon

        #TODO publish epsilon to a new channel/publish the data in obs to new channel(s)

        return np.array(obs)


    def _is_done(self, observation,score):
        done = bool(abs(observation[7]) < self.min_allowed_epsilon_p) or bool(score > 200)
        return done


    def _compute_reward(self, obs, done):
        """
        Gives more points for staying upright, gets data from given observations to avoid
        having different data than other previous functions
        :return:reward
        """
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Joint just diverged
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0
        return reward


    def step(self, action,score):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """

        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            rospy.loginfo("rosunpause failed!")

        self.iterator+=1
        # Execute "action"
        # Take action
        if action == 0: # decrease torque with very large step
            self.current_torque -= self.torque_step * 50
        elif action == 1: # decrease torque with large step
            self.current_torque -= self.torque_step * 10
        elif action == 2: # decrease torque with medium step
            self.current_torque -= self.torque_step * 5
        elif action == 3: # decrease torque with small step
            self.current_torque -= self.torque_step
        elif action == 4: # increase torque with small step
            self.current_torque += self.torque_step
        elif action == 5: # increase torque with medium step
            self.current_torque += self.torque_step * 5
        elif action == 6: # increase torque with large step
            self.current_torque += self.torque_step * 10
        elif action == 7: # increase torque with very large step
            self.current_torque += self.torque_step * 50
        

        joint_value = Float64()
        joint_value.data = self.current_torque + self.episode_external_torque
        self.torque_pub.publish(joint_value) 

        self.ros_clock = rospy.get_rostime().nsecs

        obs = self.take_observation()
        """
        t_start_episode=time.time()
        t_end_episode=t_start_episode+self.max_time_episode
        
        #rospy.loginfo(str("start" + str(t_start_episode)))
        """
        """
        
        #We give a maximum time for the joint to reach the desired position
        iteration=0
        #rospy.loginfo(str("start = " + str(t_start_episode) + " end = " + str(t_end_episode)))
        while  time.time() < t_end_episode:
            obs = self.take_observation()
            #rospy.loginfo(str(rospy.get_rostime().nsecs) + " " + str(t_end_episode))
            self.rqt_publishing(obs)
            iteration=iteration+1
            #V1 : instantaneous speed
            #rospy.loginfo(str(obs[2]))
            if abs(obs[2])<self.load_speed_criterion:
                #If the speed is low, we check if it stays low for a while. 
                is_stable=True
                #rospy.loginfo("checking stability...")
                check_rate = rospy.Rate(50)
                for i in range (0,100):
                    obs = self.take_observation()
                    self.rqt_publishing(obs)
                    if abs(obs[2]) > self.load_speed_criterion:
                        is_stable=False
                        break
                    check_rate.sleep()
                #If the speed has been bellow criterion for a while, we consider equilibrium.         
                if is_stable:
                    #rospy.loginfo(str("criterion met, speed = " + str(obs[2])))
                    break

                        
        #rospy.loginfo("t= " + str(rospy.get_rostime().nsecs) + " exited while after " + str(iteration) + " iterations, theta_ld= " + str(obs[6]) + " final angle = " + str(obs[2]) + " action = " + str(action))

          
        #While with 2 conditions: not moving or took too long. 
        #While temps< CRITERE TEMPS
            #X éléments (10) [] remplie de vitesses instantannées
            #Moyenne
            #
            #If moyenne<Critère de vitesse 
                #
        
            #V2 
            #X éléments remplie de positions 
            #Diff entre plus grand et petis<critère
            #   On est bon return  


        # Take an observation
        """
        done = self._is_done(obs,score)
        reward = self._compute_reward(obs, done)
        info = {}

        self.rqt_publishing(obs)
        
        self.rate.sleep()
        
        if done:
            rospy.loginfo(str("episode done, score = " + str(score)))

        return obs, reward, done, info



    # def step(self, action):
    #     return list(np.random.uniform(-1, 1, 8)), 1.0, False, {}


    def reset(self):
        """     
        Reset the agent for a particular experiment condition.
        """
        self.iterator = 0

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            rospy.loginfo("rospause failed!")

        self.episode_external_torque = self.np_random.uniform(-self.tau_ext_max, self.tau_ext_max)
        self.current_torque = self.np_random.uniform(-self.tau_ext_max, self.tau_ext_max)
        self.episode_theta_ld = self.np_random.uniform(-self.theta_ld_max, self.theta_ld_max)
        self.previous_epsilon = None
        self.steps_beyond_done = None
        joint_value = Float64()
        joint_value.data = self.current_torque + self.episode_external_torque
        self.torque_pub.publish(joint_value) 
        self.ros_clock = rospy.get_rostime().nsecs
        obs = self.take_observation()

        

        return obs

    def rqt_publishing(self,obs):
        """
        Will publish epsilon ant thetald to relevant channels for plotting
        """
        
        #For epsilon
        error = Float64()
        #Take epsilon from obs WARNING: refers to a static index in obs
        error.data = obs[6]
        self.error_pub.publish(error) 

        #For theta_ld
        theta_ld = Float64()
        #Take theta_ld from obs WARNING: refers to a static index in obs
        theta_ld.data = obs[0]
        self.theta_ld_pub.publish(theta_ld) 


    def close(self):
        rospy.loginfo("Closing SnakeJoint environment")
        rospy.signal_shutdown("Closing SnakeJoint environment")
