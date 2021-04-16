import rospy
from sensor_msgs.msg import JointState
import std_msgs.msg import Float64

def joint_states_callback(data):
    rospy.loginfo("joint states : {}".format(data))
    
def talk_and_listen():
    pub = rospy.Publisher('/single_joint/joint1_effort_controller/command', Float64, queue_size=10)
    rospy.Subscriber('/single_joint/joint_states', JointState, joint_states_callback)

    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10Hz
    while not rospy.is_shutdown():
        torque_value = random.uniform(0.0, 2.5) # between 0.0 N.m and 2.5 N.m
        rospy.loginfo("The torque value is : {}".format(torque_value))
        pub.publish(torque_value)
        rate.sleep()

if __name__ == '__main__':
    try:
        talk_and_listen()
    except rospy.ROSInterruptException:
        pass
