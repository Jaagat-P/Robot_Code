#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotState
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import typing as T


class RRTNode:
    def __init__(self, position: np.ndarray, parent: T.Optional['RRTNode'] = None):
        self.position = position
        self.parent = parent
        self.cost = 0.0 if parent is None else parent.cost + np.linalg.norm(position - parent.position)


class RRTFrontierExplorer(Node):
  # RRT.
    def __init__(self):
        super().__init__("rrt_frontier_explorer")
        #params; can change this as we see fit, should be fine.
        self.declare_parameter("max_iterations", 500)
        self.declare_parameter("step_size", 0.5) 
        self.declare_parameter("goal_sample_rate", 0.2) 
        self.declare_parameter("goal_tolerance", 0.3)  
        self.declare_parameter("min_frontier_cluster_size", 5)
        self.declare_parameter("exploration_gain_weight", 1.0)
        self.declare_parameter("distance_weight", 0.5)
        
        self.pose: T.Optional[TurtleBotState] = None
        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.exploring = False
        self.rrt_tree: T.List[RRTNode] = []
        
        self.create_subscription(Bool, "/nav_success", self.nav_callback, 10)
        self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        
        self.goal_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        
        self.create_timer(1.0, self.explore)
        
        self.get_logger().info("RRT intiialized")
    
    def nav_callback(self, msg: Bool):
        self.exploring = False
        if msg.data:
            self.get_logger().info("goal reached")
        else:
            self.get_logger().warn("try again")
    
    def state_callback(self, msg: TurtleBotState):
        self.pose = msg
    
    def map_callback(self, msg: OccupancyGrid):
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )
    
    def explore(self):
        if self.exploring or self.pose is None or self.occupancy is None:
            return
        
        frontiers = self.find_frontiers()
        
        if len(frontiers) == 0:
            self.get_logger().info("Exploration complete and we did not find any frontiers...")
            return
        frontier_clusters = self.cluster_frontiers(frontiers)
        
        if len(frontier_clusters) == 0:
            self.get_logger().info("No valid frontier clusters found")
            return
        best_frontier = self.select_frontier_with_rrt(frontier_clusters)
        
        if best_frontier is not None:
            self.send_goal(best_frontier)
            self.exploring = True
    
    def find_frontiers(self) -> np.ndarray:
        probs = self.occupancy.probs
        h, w = probs.shape
        frontiers = []
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if probs[i, j] < 0.2:  
                    # check neighbors
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0.4 < probs[ni, nj] < 0.6:  
                            x, y = self.occupancy.grid2state(np.array([[j, i]]))[0]
                            frontiers.append([x, y])
                            break
        
        return np.array(frontiers) if frontiers else np.array([]).reshape(0, 2)
    
    def cluster_frontiers(self, frontiers: np.ndarray) -> T.List[np.ndarray]:
        if len(frontiers) < self.get_parameter("min_frontier_cluster_size").value:
            return []
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(frontiers)
        labels = clustering.labels_
        
        # Compute cluster centroids
        clusters = []
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
            
            cluster_points = frontiers[labels == label]
            if len(cluster_points) >= self.get_parameter("min_frontier_cluster_size").value:
                centroid = np.mean(cluster_points, axis=0)
                clusters.append(centroid)
        
        return clusters
    
    def select_frontier_with_rrt(self, frontier_clusters: T.List[np.ndarray]) -> T.Optional[np.ndarray]:
        start_pos = np.array([self.pose.x, self.pose.y])
        
        best_frontier = None
        best_score = -float('inf')
        
        for frontier in frontier_clusters:
            path = self.build_rrt(start_pos, frontier)
            if path is not None:
                distance = np.linalg.norm(frontier - start_pos)
                exploration_gain = self.compute_exploration_gain(frontier)
                
                gain_weight = self.get_parameter("exploration_gain_weight").value
                dist_weight = self.get_parameter("distance_weight").value
                
                score = gain_weight * exploration_gain - dist_weight * distance
                
                if score > best_score:
                    best_score = score
                    best_frontier = frontier
        
        return best_frontier
    
    def build_rrt(self, start: np.ndarray, goal: np.ndarray) -> T.Optional[T.List[np.ndarray]]:
        max_iter = self.get_parameter("max_iterations").value
        step_size = self.get_parameter("step_size").value
        goal_sample_rate = self.get_parameter("goal_sample_rate").value
        goal_tolerance = self.get_parameter("goal_tolerance").value
        self.rrt_tree = [RRTNode(start)]
        x_min, y_min = self.occupancy.origin
        x_max = x_min + self.occupancy.size[0] * self.occupancy.resolution
        y_max = y_min + self.occupancy.size[1] * self.occupancy.resolution
        for i in range(max_iter):
            if np.random.random() < goal_sample_rate:
                rand_point = goal
            else:
                rand_point = np.array([
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max)
                ])
            
            nearest_node = self.get_nearest_node(rand_point)
            new_pos = self.steer(nearest_node.position, rand_point, step_size)
            
            if self.is_collision_free(nearest_node.position, new_pos):
                new_node = RRTNode(new_pos, nearest_node)
                self.rrt_tree.append(new_node)
                
                if np.linalg.norm(new_pos - goal) < goal_tolerance:
                    return self.extract_path(new_node)
        
        return None
    
    def get_nearest_node(self, point: np.ndarray) -> RRTNode:
        positions = np.array([node.position for node in self.rrt_tree])
        distances = np.linalg.norm(positions - point, axis=1)
        nearest_idx = np.argmin(distances)
        return self.rrt_tree[nearest_idx]
    
    def steer(self, from_pos: np.ndarray, to_pos: np.ndarray, step_size: float) -> np.ndarray:
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        
        if distance < step_size:
            return to_pos
        
        return from_pos + (direction / distance) * step_size
    
    def is_collision_free(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        num_samples = int(np.linalg.norm(pos2 - pos1) / (self.occupancy.resolution * 0.5))
        num_samples = max(num_samples, 2)
        
        for alpha in np.linspace(0, 1, num_samples):
            pos = (1 - alpha) * pos1 + alpha * pos2
            
            if not self.occupancy.is_free(pos):
                return False
        
        return True
    
    def extract_path(self, goal_node: RRTNode) -> T.List[np.ndarray]:
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        return list(reversed(path))
    
    def compute_exploration_gain(self, position: np.ndarray) -> float:
        sensor_range = 2.0 # can try this and just see what works best
        grid_pos = self.occupancy.state2grid(position.reshape(1, -1))[0]
        
        range_cells = int(sensor_range / self.occupancy.resolution)
        
        unknown_count = 0
        total_count = 0
        
        for di in range(-range_cells, range_cells + 1):
            for dj in range(-range_cells, range_cells + 1):
                gi, gj = int(grid_pos[1] + di), int(grid_pos[0] + dj)
                
                if (0 <= gi < self.occupancy.size[1] and 
                    0 <= gj < self.occupancy.size[0]):
                    total_count += 1
                    if 0.4 < self.occupancy.probs[gi, gj] < 0.6:
                        unknown_count += 1
        
        return unknown_count / max(total_count, 1)
    
    def send_goal(self, goal: np.ndarray):
        msg = TurtleBotState()
        msg.x = float(goal[0])
        msg.y = float(goal[1])
        msg.theta = 0.0  
        
        self.goal_pub.publish(msg)
        self.get_logger().info(f"Sent frontier goal: ({goal[0]:.2f}, {goal[1]:.2f})")

def main(args=None):
    rclpy.init(args=args)
    node = RRTFrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
