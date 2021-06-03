import math
from collections import deque

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
from gazebo_msgs.srv import SetModelConfiguration

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
        self.eps_buffer = deque(maxlen=15)

        # publishers and subscribers
        self.torque_pub = rospy.Publisher('/single_joint/link_motor_effort/command', Float64, queue_size=10)
        #For rqt plotting
        self.error_pub = rospy.Publisher('/single_joint/rqt/error', Float64, queue_size=10)
        self.theta_ld_pub = rospy.Publisher('/single_joint/rqt/thetald', Float64, queue_size=10)
        #Allows us to pause or unpause the physics engine, stopping or resuming movement instantly 
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        #Resets model poses
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_joints = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)

        rospy.Subscriber('/single_joint/joint_states', JointState, self.observation_callback)

        self.action_space = spaces.Discrete(self.n_actions)
        self.stability_iterator=0

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
        #rospy.loginfo("epsilon= "+ str(epsilon) + " previous_epsilon = "+ (str(self.previous_epsilon) if self.previous_epsilon else "not yet") )
        
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


        return np.array(obs)


    def _is_done(self, observation):
        eps_buffer_diverged = False
        large_eps_count = 0
        for eps in self.eps_buffer:
            if abs(eps) > self.max_allowed_epsilon:
                large_eps_count += 1
        if large_eps_count / len(self.eps_buffer) > 0.5: # If more than 50% of the entry in the buffer are greater than the max epsilon threshold 
            eps_buffer_diverged = True
        """
        if eps_buffer_diverged :
            rospy.loginfo("buffer diverged")
        
        if abs(observation[7]) < self.min_allowed_epsilon_p:
            rospy.loginfo("stability")
            
        if abs(observation[1]) >= self.theta_l_max:
            rospy.loginfo("angle too great")
        """
        done = bool(
            (
                eps_buffer_diverged
                and abs(observation[7]) < self.min_allowed_epsilon_p # If the system is not moving anymore
            ) 
            or abs(observation[1]) >= self.theta_l_max
        )
        #V2 similar to mountaincar

        
        return done


    def _compute_reward(self, obs, done):
        """
        Gives more points for staying upright, gets data from given observations to avoid
        having different data than other previous functions
        :return:reward
        """
        """
        if not done: 
            reward = 1/math.exp(obs[6]) 
        elif self.steps_beyond_done is None:
            # Joint just diverged
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0
        return reward
        """
        #V2 similar to mountaincar
        reward=0.0
        if not done: 
            reward = reward-math.exp(obs[6])
            if obs[6]<self.min_allowed_epsilon_p:
                reward=reward+0.5 
        elif self.steps_beyond_done is None:
            # Joint just diverged
            self.steps_beyond_done = 0
            reward = reward+1.0
        else:
            self.steps_beyond_done += 1
        
        return reward


    def step(self, action):
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
        self.eps_buffer.append(obs[6])
        done = self._is_done(obs)
        reward = self._compute_reward(obs, done)
        info = {}

        self.rqt_publishing(obs)
        
        self.rate.sleep()
        
        return obs, reward, done, info



    # def step(self, action):
    #     return list(np.random.uniform(-1, 1, 8)), 1.0, False, {}


    def reset(self):
        """     
        Reset the agent for a particular experiment condition.
        """
        self.iterator = 0

        #shoud reset the joint to original position
        rospy.wait_for_service('/gazebo/reset_world')

        try:
            rospy.loginfo("world reset called")
            self.reset_world()
        except (rospy.ServiceException) as e:
            rospy.loginfo("reset_world failed!")


        rospy.wait_for_service('/gazebo/set_model_configuration')

        try:
            #in the <robot>(xacro) named snake_joint of the param <robot_description>(launch)
            #give joints named fixated_to_pivot and pivot_to_moving_link(xacro) the values 0.0 and 0.0
            self.reset_joints("snake_joint", "robot_description", ["fixated_to_pivot", "pivot_to_moving_link"], [0.0, 0.0])


        except (rospy.ServiceException) as e:
            rospy.loginfo("/gazebo/reset_joints service call failed")
    

        rospy.wait_for_service('/gazebo/pause_physics')

        try:
            self.pause()
        except (rospy.ServiceException) as e:
            rospy.loginfo("rospause failed!")

        self.episode_external_torque=0
        #self.episode_external_torque = self.np_random.uniform(-self.tau_ext_max, self.tau_ext_max)
        #self.current_torque = self.np_random.uniform(-self.tau_ext_max, self.tau_ext_max)
        #added for test
        self.current_torque = 0
        self.eps_buffer = deque(maxlen=15)
        self.episode_theta_ld = self.np_random.uniform(-self.theta_ld_max, self.theta_ld_max)
        self.previous_epsilon = None
        self.steps_beyond_done = None
        joint_value = Float64()
        joint_value.data = self.current_torque + self.episode_external_torque

        #setting the phantom joint to the target position
        rospy.wait_for_service('/gazebo/set_model_configuration')

        try:
            #in the <robot>(xacro) named snake_joint of the param <robot_description>(launch)
            #give joints named fixated_to_pivot and pivot_to_moving_link(xacro) the values 0.0 and 0.0
            self.reset_joints("snake_joint", "robot_description", ["pivot_to_moving_link_phantom","spring_phantom"], [self.episode_theta_ld, 0.0])


        except (rospy.ServiceException) as e:
            rospy.loginfo("/gazebo/reset_joints service call failed")

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            rospy.loginfo("rosunpause failed!")

        self.torque_pub.publish(joint_value) 

        #Torque publish rate = 5hz, joint states publish rate = 160-200hz
        #Theory: joint states is updated more than twice before torque is recieved and applied, meaning epsilon-previous_epsilon=0: the joint does not move yet. 
        #This is an attempt at synchronization
        #Failed, problem solved another way
        """
        try:
            synchro=rospy.wait_for_message('/single_joint/link_motor_effort/command', Float64, timeout=5)
        except rospy.ROSInterruptException:
            rospy.loginfo("no torque detected or timeout")
        """
        
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
