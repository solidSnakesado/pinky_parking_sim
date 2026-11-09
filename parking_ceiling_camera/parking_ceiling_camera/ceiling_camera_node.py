"""Ceiling (overhead) camera for the MuJoCo Pinky sim.

pinky_pro is not modified: this node loads the scene the sim generated
(~/.cache/pinky_mujoco/scene.xml), adds a fixed world camera, mirrors the robot
pose (/ground_truth) and wheel joints (/joint_states), and publishes the render.
"""
import math
import os

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState

# Matte colours sampled from a top-down photo of the real Pinky (render-only, physics unchanged).
REAL_COLORS = {'wheel': (0.05, 0.42, 0.72), 'body': (0.06, 0.06, 0.07), 'lidar': (0.04, 0.04, 0.05),
               'rear': (0.62, 0.62, 0.66), 'front': (0.45, 0.45, 0.45)}
# First matching geom-name prefix wins.
COLOR_RULES = [('l_wheel_visual_0_4', 'wheel'), ('r_wheel_visual_0_4', 'wheel'),
               ('l_wheel_visual', 'body'), ('r_wheel_visual', 'body'),
               ('robot_lamp_visual', 'rear'), ('screen_mount_visual', 'front'),
               ('rplidar_mount_visual', 'lidar'), ('base_link_visual', 'body'),
               ('front_camera_mount_visual', 'body'), ('caster_', 'body')]


def apply_real_colors(spec):
    for key, rgb in REAL_COLORS.items():
        spec.add_material(name=f'real_{key}', rgba=[*rgb, 1.0], specular=0.05, shininess=0.1, reflectance=0.0)
    for geom in spec.geoms:
        if 'visual' not in geom.name:
            continue
        for prefix, key in COLOR_RULES:
            if geom.name.startswith(prefix):
                geom.material = f'real_{key}'
                geom.rgba = [*REAL_COLORS[key], 1.0]
                break


class CeilingCamera(Node):
    def __init__(self):
        super().__init__('ceiling_camera_node')
        p = self.declare_parameter
        model_path = os.path.expanduser(p('model_path', '~/.cache/pinky_mujoco/scene.xml').value)
        pos = list(p('position', [0.8, 1.05, 0.98]).value)
        quat = list(p('quat', [0.7071068, 0.0, 0.0, 0.7071068]).value)
        hfov = math.radians(p('horizontal_fov_deg', 60.0).value)
        self.width = int(p('width', 640).value)
        self.height = int(p('height', 480).value)
        fps = float(p('fps', 10.0).value)
        self.frame_id = p('frame_id', 'ceiling_camera').value
        real_colors = bool(p('real_colors', True).value)
        fovy = math.degrees(2 * math.atan(math.tan(hfov / 2) * self.height / self.width))

        spec = mujoco.MjSpec.from_file(model_path)
        if real_colors:
            apply_real_colors(spec)
        spec.worldbody.add_camera(name='ceiling_camera', pos=pos, quat=quat, fovy=fovy)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.base_z = float(self.data.qpos[2])
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)

        fy = self.height / 2 / math.tan(math.radians(fovy) / 2)
        cx, cy = self.width / 2, self.height / 2
        self.info = CameraInfo(width=self.width, height=self.height, distortion_model='plumb_bob',
                               d=[0.0] * 5, k=[fy, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
                               r=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
                               p=[fy, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.info.header.frame_id = self.frame_id

        self.create_subscription(PoseStamped, '/ground_truth', self.on_pose, 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.image_pub = self.create_publisher(Image, 'ceiling_camera/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, 'ceiling_camera/camera_info', 10)
        self.create_timer(1.0 / fps, self.publish)
        self.get_logger().info(f'ceiling camera {self.width}x{self.height} @ {fps:g} Hz, '
                               f'pos={pos}, hfov={math.degrees(hfov):g} deg, real_colors={real_colors}')

    def on_pose(self, msg):
        q = msg.pose.orientation
        self.data.qpos[0:3] = [msg.pose.position.x, msg.pose.position.y, self.base_z]
        self.data.qpos[3:7] = [q.w, q.x, q.y, q.z]

    def on_joints(self, msg):
        for name, value in zip(msg.name, msg.position):
            try:
                self.data.qpos[self.model.joint(name).qposadr[0]] = value
            except KeyError:
                pass

    def publish(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self.renderer.update_scene(self.data, camera='ceiling_camera')
        rgb = self.renderer.render()
        stamp = self.get_clock().now().to_msg()
        image = Image(height=self.height, width=self.width, encoding='rgb8', is_bigendian=0,
                      step=self.width * 3, data=np.ascontiguousarray(rgb).tobytes())
        image.header.stamp, image.header.frame_id = stamp, self.frame_id
        self.info.header.stamp = stamp
        self.image_pub.publish(image)
        self.info_pub.publish(self.info)


def main(args=None):
    rclpy.init(args=args)
    node = CeilingCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
