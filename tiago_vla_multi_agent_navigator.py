import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import base64
import time
import re
from huggingface_hub import InferenceClient

class TiagoFullFOVAgentNavigator(Node):
    def __init__(self):
        super().__init__('tiago_full_fov_agent_navigator')
        self.bridge = CvBridge()
        
        # Initialize Hugging Face Client for VLM Inference
        self.client = InferenceClient(api_key="HF_API_KEY")
        
        # --- Control Parameters ---
        self.max_linear_speed = 0.22  # Maximum forward velocity (m/s)
        self.kp_angular = 0.85        # Proportional Gain for steering response
        self.last_request_time = 0    # Timer to manage API request frequency
        self.request_interval = 6.0   # Seconds between AI reasoning cycles
        
        # ROS2 Subscription to the camera feed
        self.subscription = self.create_subscription(
            Image, '/Tiago_Lite/Astra_rgb/image_color', self.image_callback, 10
        )
        
        # ROS2 Publisher for movement commands
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info('--- TIAGo Full FOV Multi-Agent Navigator Started ---')
        self.get_logger().info('Architecture: Peripheral Perception + Semantic Attention')

    def get_perception_report(self, b64_img):
        """ 
        AGENT 1: The Observer 
        Responsible for semantic scene analysis and object localization.
        Uses full FOV to ensure the target door is detected even at periphery.
        """
        prompt = (
            "You are the eyes of a TIAGo robot. Analyze the FULL camera feed:\n"
            "1. Search the ENTIRE frame for a white door. If seen, report its position (Far Left, Center-Left, Center, Center-Right, Far Right).\n"
            "2. Identify obstacles ONLY if they are directly in the lower-central path (potential collision).\n"
            "3. Disregard objects on the far edges unless they are the door target."
        )
        response = self.client.chat_completion(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]}],
            max_tokens=100
        )
        return response.choices[0].message.content

    def get_pilot_decision(self, perception_report):
        """ 
        AGENT 2: The Pilot 
        Translates semantic reports into numerical control setpoints (Speed/Steer).
        """
        prompt = (
            f"Context: {perception_report}\n\n"
            "You are the Pilot. Execute based on these rules:\n"
            "1. If the door is visible anywhere, set STATUS:TRACKING and align STEER. "
            "(-1.0 for Far Left, 0.0 for Center, 1.0 for Far Right).\n"
            "2. If NO door is visible, set STATUS:SEARCHING, SPEED:0.0, STEER:0.0.\n"
            "3. If an obstacle is directly in the center, set SPEED:0.0 and STEER away.\n\n"
            "FORMAT: SPEED:[0.0-1.0], STEER:[-1.0 to 1.0], STATUS:[SEARCHING/TRACKING/GOAL]"
        )
        response = self.client.chat_completion(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        return response.choices[0].message.content

    def image_callback(self, msg):
        """ Main ROS2 Callback: Handles the perception-action loop. """
        current_time = time.time()
        # Rate limiting to avoid API overloading
        if current_time - self.last_request_time < self.request_interval:
            return
        self.last_request_time = current_time
        
        try:
            # Convert ROS2 Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Encode image to Base64 for API transmission
            _, buffer = cv2.imencode('.jpg', cv_image)
            b64_img = base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            self.get_logger().error(f'CV Conversion Error: {e}')
            return

        try:
            # Perception (Observer Agent) 
            report = self.get_perception_report(b64_img)
            self.get_logger().info(f'[Observer]: {report}')

            # Reasoning (Pilot Agent) 
            control_data = self.get_pilot_decision(report)
            self.get_logger().info(f'[Pilot Data]: {control_data}')

            # Control Execution (P-Controller):
            # Extract numerical values from AI response using Regex
            speed_match = re.search(r"SPEED:([\d\.]+)", control_data)
            steer_match = re.search(r"STEER:([-\d\.]+)", control_data)
            
            move_msg = Twist()
            report_upper = report.upper()
            # Failsafe: check both Agent 1 report and Agent 2 status for door visibility
            door_found = "YES" in report_upper or "DOOR" in report_upper or "VISIBLE" in report_upper

            if door_found:
                # TRACKING MODE: Goal is visible 
                if speed_match and steer_match:
                    speed_factor = float(speed_match.group(1))
                    steer_error = float(steer_match.group(1))
                    
                    if "GOAL" in control_data.upper():
                        move_msg.linear.x = 0.0
                        move_msg.angular.z = 0.0
                        self.get_logger().info('>>> GOAL REACHED')
                    else:
                        # P-Controller: Angular velocity proportional to steering error
                        move_msg.angular.z = -1.0 * self.kp_angular * steer_error
                        
                        # Velocity Profiling: Reduce linear speed during sharp turns for stability
                        stability = 1.0 - (abs(steer_error) * 0.6)
                        move_msg.linear.x = self.max_linear_speed * speed_factor * stability
                        self.get_logger().info(f'>>> TRACKING: V={move_msg.linear.x:.2f}, W={move_msg.angular.z:.2f}')
            else:
                # SPIN SEARCH MODE: No door detected 
                # Robot performs a constant right-side rotation to scan the room
                move_msg.linear.x = 0.0
                move_msg.angular.z = -0.75 
                self.get_logger().warn('>>> SEARCHING: Fixed Right-Side Spinning...')

            # Publish the calculated Twist command to the robot
            self.publisher.publish(move_msg)

        except Exception as e:
            self.get_logger().error(f'Multi-Agent Pipeline Error: {e}')

def main():
    rclpy.init()
    node = TiagoFullFOVAgentNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()