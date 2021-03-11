import sys
import random

import rospy

from single_joint.srv import PositionOrder, PositionFeedback

class OrderGenerator:
    
    def __init__(self):
        self.theta_ld_min = rospy.get_param("/generator/theta_ld_min")
        self.theta_ld_max = rospy.get_param("/generator/theta_ld_max")
        self.theta_ld_resolution = rospy.get_param("/generator/theta_ld_resolution")
        self.theta_l_min = rospy.get_param("/generator/theta_l_min")
        self.theta_l_max = rospy.get_param("/generator/theta_l_max")
        self.theta_l_resolution = rospy.get_param("/generator/theta_l_resolution")
        self.theta_m_min = rospy.get_param("/generator/theta_m_min")
        self.theta_m_max = rospy.get_param("/generator/theta_m_max")
        self.theta_m_resolution = rospy.get_param("/generator/theta_m_resolution")
        self.theta_m_p_min = rospy.get_param("/generator/theta_m_p_min")
        self.theta_m_p_max = rospy.get_param("/generator/theta_m_p_max")
        self.theta_m_p_resolution = rospy.get_param("/generator/theta_m_p_resolution")

        rospy.wait_for_service("position_order")
        self.position_order_client = rospy.ServiceProxy("position_order", PositionOrder)
        
        self.position_feedback_server = rospy.Service("position_feedback", PositionFeedback, self._feedback_handler)

        self.position_order = dict()
        self.position_order_count = 0
        self.position_feedback = dict()
        self.position_feedback_count = 0

        rospy.loginfo("Starting the generator node...")


    def run(self):
        if len(self.position_order) == 0 and len(self.position_feedback) == 0:
            self._generate()
            self._send_order()


    def _send_order(self):
        try:
            res = self.position_order_client(**self.position_order)
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)
        rospy.loginfo(f"Order sent: {self.position_order}")


    def _feedback_handler(self, req):
        self.position_feedback["stamp"] = req.stamp
        self.position_feedback["msg_id"] = req.msg_id
        self.position_feedback["theta_m_update"] = req.theta_m_update
        self.position_feedback["theta_l_update"] = req.theta_l_update
        self.position_feedback["should_reset"] = req.should_reset

        rospy.loginfo(
            (
                "Packet received:\n" +
                "\tstamp : %i\n" +
                "\tmsg_id : %d\n" +
                "\ttheta_m_update : %f rad\n" +
                "\ttheta_l_update : %f rad\n" +
                "\tshould_reset : %i\n"
            ),
            req.stamp.nsecs,
            req.msg_id,
            req.theta_m_update,
            req.theta_l_update,
            req.should_reset
        )

        if req.should_reset:
            self.position_feedback = dict()
            self._generate()
        else:
            self._next()
        self._send_order()


    def _generate(self):
        """
        Random initialization of the input parameters.
        - 'theta_ld' : will be fixed during the whole episode ('epsilon' is computed using 'theta_ld' and the moving value of 'theta_l').
        - 'theta_l' : will be randomly initialized (a joint is not necessarily straight at the beginning) but will vary through time.
        - 'theta_m' : same as 'theta_l'.
        - 'theta_m_p' ; we randomly decide of which rotationnal speed we want at the beginning. It may vary through time as well.
        """
        rand_theta_ld = self._random_sample(self.theta_ld_min, self.theta_ld_max, self.theta_ld_resolution)
        rand_theta_l = self._random_sample(self.theta_l_min, self.theta_l_max, self.theta_l_resolution)
        rand_theta_m = self._random_sample(self.theta_m_min, self.theta_m_max, self.theta_m_resolution)
        rand_theta_m_p = self._random_sample(self.theta_m_p_min, self.theta_m_p_max, self.theta_m_p_resolution)

        # set the values of the differential parameters with very big values at first (theoretically infinity)
        theta_l_p = 1e20 
        epsilon = 1e20
        epsilon_p = 1e20

        # return the Service to send
        self.position_order["stamp"] = rospy.get_rostime()
        self.position_order["msg_id"] = self.position_order_count
        self.position_order["theta_ld"] = rand_theta_ld
        self.position_order["theta_l"] = rand_theta_l
        self.position_order["theta_l_p"] = theta_l_p
        self.position_order["theta_m"] = rand_theta_m
        self.position_order["theta_m_p"] = rand_theta_m_p
        self.position_order["epsilon"] = epsilon
        self.position_order["epsilon_p"] = epsilon_p

        rospy.loginfo(f"Order generated : {self.position_order}")


    def _next(self):
        next_position = dict()
        delta = self.position_feedback["stamp"] - self.position_order["stamp"]
        next_position["stamp"] = rospy.get_rostime()
        next_position["msg_id"] = self.position_order_count
        next_position["theta_ld"] = self.position_order["theta_ld"]
        next_position["theta_l"] = self.position_feedback["theta_l_update"]
        next_position["theta_l_p"] = (self.position_feedback["theta_l_update"] - self.position_order["theta_l"]) / delta
        next_position["theta_m"] = self.position_feedback["theta_m_update"]
        next_position["theta_m_p"] = (self.position_feedback["theta_m_update"] - self.position_order["theta_m"]) / delta
        next_position["epsilon"] = abs(self.position_feedback["theta_l_update"] - self.position_order["theta_ld"])
        next_position["epsilon_p"] = (
            abs(self.position_feedback["theta_l_update"] - self.position_order["theta_ld"]) -
            self.position_order["epsilon"]
        ) / delta
        
        # update position_order
        self.position_order = next_position


    def _random_sample(self, min_val, max_val, resolution):
        l = int((max_val - min_val) // resolution)
        dist = []
        for i in range(l):
            dist.append(min_val + i * resolution)
        rand_idx = random.randint(0, l)
        return dist[rand_idx]
