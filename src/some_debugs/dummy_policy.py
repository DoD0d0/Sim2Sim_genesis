import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

class DummyPolicy(Node):
    def __init__(self):
        super().__init__('dummy_policy')
        self.pub = self.create_publisher(Float32MultiArray, '/dodo/effort_cmd', 10)
        self.sub = self.create_subscription(JointState, '/dodo/joint_states', self.on_js, 10)
        self.n = None
        self.count = 0

    def on_js(self, msg: JointState):
        # 第一次收到 joint_states 时，确定 DOF 数
        if self.n is None:
            self.n = len(msg.name)
            self.get_logger().info(f"Got joint_states with n_dofs={self.n}")

        # 发全 0 力矩（长度必须匹配）
        cmd = Float32MultiArray()
        cmd.data = [0.0] * self.n

        self.pub.publish(cmd)

        self.count += 1
        if self.count % 120 == 0:
            self.get_logger().info("Publishing torques (current demo mode).")

def main():
    rclpy.init()
    node = DummyPolicy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
