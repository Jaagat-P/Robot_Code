#!/usr/bin/env python3

# experimenting with PID controllers, which we explored during lecture. 
import rclpy
from rclpy.node import Node
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from geometry_msgs.msg import Twist
import numpy as np
from collections import deque


class PIDVelocityController(Node):
    
    def __init__(self):
        super().__init__("pid_velocity_controller")
        self.declare_parameter("kp_linear", 1.0)    
        self.declare_parameter("ki_linear", 0.1)  
        self.declare_parameter("kd_linear", 0.05)  
        self.declare_parameter("kp_angular", 1.0)  
        self.declare_parameter("ki_angular", 0.1)   
        self.declare_parameter("kd_angular", 0.05)  
        self.declare_parameter("max_integral", 1.0)
        self.declare_parameter("control_rate", 50.0)  # Hz
        self.desired_v = 0.0
        self.desired_omega = 0.0
        self.actual_v = 0.0
        self.actual_omega = 0.0
        self.error_v = 0.0
        self.error_omega = 0.0
        self.integral_v = 0.0
        self.integral_omega = 0.0
        self.prev_error_v = 0.0
        self.prev_error_omega = 0.0
        self.prev_time = self.get_clock().now()
        self.error_history_v = deque(maxlen=100)
        self.error_history_omega = deque(maxlen=100)
        self.create_subscription(
            TurtleBotControl, 
            "/cmd_vel_desired",  
            self.desired_velocity_callback, 
            10
        )
        self.create_subscription(
            TurtleBotState,
            "/state",  
            self.state_callback,
            10
        )
        self.control_pub = self.create_publisher(
            TurtleBotControl, 
            "/cmd_vel",  
            10
        )
        rate = self.get_parameter("control_rate").value
        self.create_timer(1.0 / rate, self.control_loop)
        
        self.get_logger().info(f"PID velocity controller is initialized at this {rate} Hz")
    
    def desired_velocity_callback(self, msg: TurtleBotControl):
        self.desired_v = msg.v
        self.desired_omega = msg.omega
    
    def state_callback(self, msg: TurtleBotState):
        self.actual_v = np.sqrt(msg.x**2 + msg.y**2)  
        self.actual_omega = msg.theta  
       
    
    def control_loop(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = current_time
        if dt <= 0:
            return
        kp_v = self.get_parameter("kp_linear").value
        ki_v = self.get_parameter("ki_linear").value
        kd_v = self.get_parameter("kd_linear").value
        kp_omega = self.get_parameter("kp_angular").value
        ki_omega = self.get_parameter("ki_angular").value
        kd_omega = self.get_parameter("kd_angular").value
        max_integral = self.get_parameter("max_integral").value
        self.error_v = self.desired_v - self.actual_v
        self.integral_v += self.error_v * dt
        self.integral_v = np.clip(self.integral_v, -max_integral, max_integral)
        derivative_v = (self.error_v - self.prev_error_v) / dt
        control_v = (kp_v * self.error_v + 
                    ki_v * self.integral_v + 
                    kd_v * derivative_v)
        output_v = self.desired_v + control_v
        self.error_omega = self.desired_omega - self.actual_omega
        self.integral_omega += self.error_omega * dt
        self.integral_omega = np.clip(self.integral_omega, -max_integral, max_integral)
        derivative_omega = (self.error_omega - self.prev_error_omega) / dt
        control_omega = (kp_omega * self.error_omega + 
                        ki_omega * self.integral_omega + 
                        kd_omega * derivative_omega)
        
        output_omega = self.desired_omega + control_omega
        self.prev_error_v = self.error_v
        self.prev_error_omega = self.error_omega
        self.error_history_v.append(self.error_v)
        self.error_history_omega.append(self.error_omega)
        msg = TurtleBotControl()
        msg.v = float(output_v)
        msg.omega = float(output_omega)
        self.control_pub.publish(msg)
        if len(self.error_history_v) % 50 == 0:
            avg_error_v = np.mean(list(self.error_history_v)[-50:])
            avg_error_omega = np.mean(list(self.error_history_omega)[-50:])
            self.get_logger().info(
                f"PID Status - "
                f"v: desired={self.desired_v:.3f}, actual={self.actual_v:.3f}, "
                f"error={self.error_v:.3f} (avg={avg_error_v:.3f}) | "
                f"ω: desired={self.desired_omega:.3f}, actual={self.actual_omega:.3f}, "
                f"error={self.error_omega:.3f} (avg={avg_error_omega:.3f})"
            )
    def reset_integral(self):
        self.integral_v = 0.0
        self.integral_omega = 0.0
        self.get_logger().info("reset the PID integrals we established")

def main(args=None):
    rclpy.init(args=args)
    node = PIDVelocityController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == "__main__":
    main()
