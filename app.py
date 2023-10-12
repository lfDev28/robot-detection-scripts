import cv2
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String  # Assuming signs can be represented as strings for simplicity
from cv_bridge import CvBridge

from EdgeDetector import EdgeDetector
from Feature import Feature
# Include your Sign Detector import here, e.g., from SignDetector import SignDetector

class DetectionNode:
    def __init__(self):
        rospy.init_node('detection_node')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sign_pub = rospy.Publisher('/detected_sign', String, queue_size=10)
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.bridge = CvBridge()

        self.edge_detector = EdgeDetector()
        self.feature_detector = Feature()
        # Initialize your sign detector here, e.g., self.sign_detector = SignDetector()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Edge Detection and decision making
            edges = self.edge_detector.process_frame(frame)
            cmd_msg = self.make_decision(edges)
            self.cmd_vel_pub.publish(cmd_msg)

            # Feature Detection (assuming it returns detected features as list for now)
            features = self.feature_detector.process_frame(frame)
            for feature in features:
                # You can publish or act on the detected features here

            # Sign Detection (assuming it returns detected signs as string for now)
            # detected_sign = self.sign_detector.process_frame(frame)
            # sign_msg = String()
            # sign_msg.data = detected_sign
            # self.sign_pub.publish(sign_msg)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def make_decision(self, edges):
        # Decision-making logic based on edges as previously described
        height, width = edges.shape
        middle = edges[height//3: 2*height//3, width//3: 2*width//3]
        white_pixels = cv2.countNonZero(middle)

        cmd_msg = Twist()
        if white_pixels > (width * height * 0.3):
            cmd_msg.linear.x = 0
        else:
            cmd_msg.linear.x = 0.5
        return cmd_msg

if __name__ == '__main__':
    dn = DetectionNode()
    rospy.spin()
