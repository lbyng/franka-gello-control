# ruff: noqa
import time
from typing import List, Optional
import socket
import json

import mujoco
import mujoco.viewer
import numpy as np

from STservo_sdk import *


DEFAULT_NORMALIZATION_LIMITS = [
    [481, 3696],
    [1151, 2706],
    [2330, 812],
    [767, 3295],
    [1533, 3623],
    [919, 3528],
    [2125, 1881],
]

DEFAULT_JOINT_LIMITS = [
    [-235, 35],
    [0, 135],
    [-135, 0],
    [-202.5, 22.5],
    [-90, 90],
    [-202.5, 22.5],
    [180, -180],
]

def euler_to_quat(roll, pitch, yaw):

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

def rmat_to_quat(rmat):

    trace = rmat[0, 0] + rmat[1, 1] + rmat[2, 2]
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (rmat[2, 1] - rmat[1, 2]) / S
        y = (rmat[0, 2] - rmat[2, 0]) / S
        z = (rmat[1, 0] - rmat[0, 1]) / S
    elif rmat[0, 0] > rmat[1, 1] and rmat[0, 0] > rmat[2, 2]:
        S = np.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2]) * 2
        w = (rmat[2, 1] - rmat[1, 2]) / S
        x = 0.25 * S
        y = (rmat[0, 1] + rmat[1, 0]) / S
        z = (rmat[0, 2] + rmat[2, 0]) / S
    elif rmat[1, 1] > rmat[2, 2]:
        S = np.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2]) * 2
        w = (rmat[0, 2] - rmat[2, 0]) / S
        x = (rmat[0, 1] + rmat[1, 0]) / S
        y = 0.25 * S
        z = (rmat[1, 2] + rmat[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1]) * 2
        w = (rmat[1, 0] - rmat[0, 1]) / S
        x = (rmat[0, 2] + rmat[2, 0]) / S
        y = (rmat[1, 2] + rmat[2, 1]) / S
        z = 0.25 * S
    
    return np.array([w, x, y, z])

class GelloController:
    def __init__(
        self,
        device_name: str,
        arm_id: str = "right",
        baud_rate: int = 1000000,
    ):
        self.baud_rate = baud_rate
        self.device_name = device_name
        self.arm_id = arm_id
        self.portOpen = False
        self.num_motors = 7
        self.motor_pos = np.zeros(self.num_motors, dtype=np.float32)
        self.motor_speed = np.zeros(self.num_motors, dtype=np.float32)
        
        try:
            self.portHandler = PortHandler(self.device_name)
            self.packetHandler = sts(self.portHandler)

            if self.portHandler.openPort():
                print(f"Successfully opened {arm_id} arm port: {device_name}")
                self.portOpen = True
            else:
                print(f"Failed to open {arm_id} arm port, using simulated data")
                return

            if self.portHandler.setBaudRate(self.baud_rate):
                print(f"{arm_id} arm baud rate set successfully")
            else:
                print(f"{arm_id} arm baud rate setting failed, using simulated data")
                self.portHandler.closePort()
                self.portOpen = False
                return

            self.groupSyncRead = GroupSyncRead(self.packetHandler, STS_PRESENT_POSITION_L, 4)
        except Exception as e:
            print(f"Failed to initialize {arm_id} arm controller: {e}")
            print("Will use simulated data")

    def read_joints(self):
        if not self.portOpen:
            t = time.time()
            for i in range(self.num_motors):
                self.motor_pos[i] = 2000 + 500 * np.sin(t * 0.5 + i * 0.5)
            return self.motor_pos, self.motor_speed
        
        try:
            for sts_id in range(1, self.num_motors + 1):
                sts_addparam_result = self.groupSyncRead.addParam(sts_id)
                if not sts_addparam_result:
                    print(f"[{self.arm_id}][ID:{sts_id:03d}] groupSyncRead addparam Failed")
                    return None, None

            sts_comm_result = self.groupSyncRead.txRxPacket()
            if sts_comm_result != COMM_SUCCESS:
                print(f"[{self.arm_id}]{self.packetHandler.getTxRxResult(sts_comm_result)}")

            for sts_id in range(1, 8):
                sts_data_result, sts_error = self.groupSyncRead.isAvailable(sts_id, STS_PRESENT_POSITION_L, 4)
                if sts_data_result:
                    sts_present_position = self.groupSyncRead.getData(sts_id, STS_PRESENT_POSITION_L, 2)
                    sts_present_speed = self.groupSyncRead.getData(sts_id, STS_PRESENT_SPEED_L, 2)
                    self.motor_pos[sts_id - 1] = sts_present_position
                    self.motor_speed[sts_id - 1] = self.packetHandler.sts_tohost(sts_present_speed, 15)
                else:
                    print(f"[{self.arm_id}][ID:{sts_id:03d}] groupSyncRead getdata Failed")
                    return None, None
                if sts_error != 0:
                    print(f"[{self.arm_id}]{self.packetHandler.getRxPacketError(sts_error)}")
            self.groupSyncRead.clearParam()
        except Exception as e:
            print(f"[{self.arm_id}]Read Joint Data Error: {e}")
            t = time.time()
            for i in range(self.num_motors):
                self.motor_pos[i] = 2000 + 500 * np.sin(t * 0.5 + i * 0.5)

        return self.motor_pos, self.motor_speed

    def close(self):
        if self.portOpen:
            print(f"Close {self.arm_id} Arm Port")
            self.portHandler.closePort()
            self.portOpen = False

    def __del__(self):
        self.close()


class GelloControllerWrapper:
    def __init__(
        self,
        controller,
        motor_limits=None,
        joint_limits=None,
        use_mujoco=True,
        xml_path="gamify/controllers/gello_arm.xml",
    ):
        super().__init__()

        self.controller = controller
        self.motor_limits = motor_limits
        self.joint_limits = joint_limits
        if self.motor_limits is None:
            self.motor_limits = DEFAULT_NORMALIZATION_LIMITS
        if self.joint_limits is None:
            self.joint_limits = DEFAULT_JOINT_LIMITS

        self.motor_limits = np.array(self.motor_limits)
        self.joint_limits = np.radians(self.joint_limits)

        if use_mujoco:
            try:
                self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
                self.mujoco_data = mujoco.MjData(self.mujoco_model)
            except Exception as e:
                print(f"Failed to load MuJoCo model: {e}")
                print("Using simple default model")
                default_xml = """
                <mujoco>
                  <worldbody>
                    <body name="link6">
                      <geom type="sphere" size="0.05"/>
                    </body>
                  </worldbody>
                </mujoco>
                """
                self.mujoco_model = mujoco.MjModel.from_xml_string(default_xml)
                self.mujoco_data = mujoco.MjData(self.mujoco_model)

    def get_joint_pos(self):
        pos, _ = self.controller.read_joints()
        if pos is None:
            return np.zeros(7)
        pos = (pos - self.motor_limits[:, 1]) / (self.motor_limits[:, 0] - self.motor_limits[:, 1])
        pos = self.joint_limits[:, 0] + pos * (self.joint_limits[:, 1] - self.joint_limits[:, 0])
        return pos

    def close(self):
        self.controller.close()

    def get_ee_pos_rmat_gripper(self):
        qpos = self.get_joint_pos()
        try:
            self.mujoco_data.qpos[:] = qpos[:-1]
            mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
            link6_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "link6")
            pos = np.array(self.mujoco_data.xpos[link6_id])
            rmat = np.array(self.mujoco_data.xmat[link6_id]).reshape(3, 3)

            pos = np.array([pos[1], pos[0], pos[2]])
            rmat = np.array([rmat[1], -rmat[0], rmat[2]])
            gripper = qpos[-1]
        except Exception as e:
            print(f"[{self.controller.arm_id}]Forward kinematics calculation error: {e}")
            if self.controller.arm_id == "left":
                pos = np.array([0.6, -0.2, 1.0])
            else:
                pos = np.array([0.6, 0.2, 1.0])
            rmat = np.eye(3)
            gripper = 0.02

        return pos, rmat, gripper


class BimanualArmClient:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to simulator: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connect Failed: {e}")
            return False
    
    def send_data(self, left_pos=None, left_quat=None, left_grip=None, right_pos=None, right_quat=None, right_grip=None):
        if not self.connected:
            if not self.connect():
                return False
                
        try:
            data = {}
            
            if left_pos is not None and left_grip is not None:
                data['left'] = {
                    'pos': left_pos.tolist() if isinstance(left_pos, np.ndarray) else left_pos,
                    'grip': float(left_grip)
                }
                if left_quat is not None:
                    data['left']['quat'] = left_quat.tolist() if isinstance(left_quat, np.ndarray) else left_quat
            
            if right_pos is not None and right_grip is not None:
                data['right'] = {
                    'pos': right_pos.tolist() if isinstance(right_pos, np.ndarray) else right_pos,
                    'grip': float(right_grip)
                }
                if right_quat is not None:
                    data['right']['quat'] = right_quat.tolist() if isinstance(right_quat, np.ndarray) else right_quat
            
            if not data:
                return False
                
            json_data = json.dumps(data)
            self.socket.sendall(json_data.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Failed to send data: {e}")
            self.connected = False
            return False
            
    def close(self):
        if self.connected and self.socket:
            try:
                self.socket.close()
                print("Client connection closed")
            except Exception as e:
                print(f"Connection closing error: {e}")
            finally:
                self.connected = False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual Arm Gello Controller Client")
    parser.add_argument('--left_port', type=str, default="/dev/cu.usbmodem58FA0957621")
    parser.add_argument('--right_port', type=str, default="/dev/cu.usbmodem58FA0820271")
    parser.add_argument('--host', type=str, default="localhost", help="Simulator Host")
    parser.add_argument('--socket_port', type=int, default=12345, help="Simulator")
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--test_rotation', action='store_true', help="Test rotation by changing RPY angles over time")
    
    args = parser.parse_args()
    
    left_controller = None
    right_controller = None
    left_wrapper = None
    right_wrapper = None
    
    try:
        left_controller = GelloController(device_name=args.left_port, arm_id="left", baud_rate=1000000)
        left_wrapper = GelloControllerWrapper(controller=left_controller)
        print("Left arm controller initialization complete")

        right_controller = GelloController(device_name=args.right_port, arm_id="right", baud_rate=1000000)
        right_wrapper = GelloControllerWrapper(controller=right_controller)
        print("Right arm controller initialization complete")
            
    except Exception as e:
        print(f"Controller creation failed: {e}")
    
    client = BimanualArmClient(host=args.host, port=args.socket_port)
    
    try:
        if args.view:
            view_model = None
            view_data = None
            if right_wrapper is not None:
                view_model = right_wrapper.mujoco_model
                view_data = right_wrapper.mujoco_data
            elif left_wrapper is not None:
                view_model = left_wrapper.mujoco_model
                view_data = left_wrapper.mujoco_data
            
            if view_model is not None:
                with mujoco.viewer.launch_passive(view_model, view_data) as viewer:
                    print("Starting to send controller data to simulator...")
                    
                    start_time = time.time()
                    
                    while True:
                        left_pos = None
                        left_rmat = None
                        left_gripper = None
                        left_quat = None
                        
                        if left_wrapper is not None:
                            left_pos, left_rmat, left_gripper = left_wrapper.get_ee_pos_rmat_gripper()
                            
                            left_quat = rmat_to_quat(left_rmat.reshape(3, 3))
                            
                            if args.test_rotation:
                                elapsed = time.time() - start_time
                                roll = 0.0
                                pitch = 0.2 * np.sin(elapsed * 0.5)
                                yaw = 0.2 * np.sin(elapsed * 0.3)
                                left_quat = euler_to_quat(roll, pitch, yaw)
                        
                        right_pos = None
                        right_rmat = None
                        right_gripper = None
                        right_quat = None
                        
                        if right_wrapper is not None:
                            right_pos, right_rmat, right_gripper = right_wrapper.get_ee_pos_rmat_gripper()
                            
                            right_quat = rmat_to_quat(right_rmat.reshape(3, 3))
                            
                            if args.test_rotation:
                                elapsed = time.time() - start_time
                                roll = 0.1 * np.sin(elapsed * 0.4)
                                pitch = 0.0
                                yaw = 0.3 * np.sin(elapsed * 0.2)
                                right_quat = euler_to_quat(roll, pitch, yaw)
                        
                        client.send_data(left_pos, left_quat, left_gripper, right_pos, right_quat, right_gripper)
                        
                        if left_pos is not None:
                            print(f"Left Arm: Pose {left_pos}, Quat {left_quat if left_quat is not None else 'None'}, Gripper {left_gripper:.4f}")
                        if right_pos is not None:
                            print(f"Right Arm: Pose {right_pos}, Quat {right_quat if right_quat is not None else 'None'}, Gripper {right_gripper:.4f}")
                        
                        time.sleep(0.02)
                        viewer.sync()
            else:
                print("Cannot start viewer, no model available")
        else:
            print("Starting to send controller data to simulator...")
            
            last_print_time = 0
            start_time = time.time()
            
            while True:
                left_pos = None
                left_rmat = None
                left_gripper = None
                left_quat = None
                
                if left_wrapper is not None:
                    left_pos, left_rmat, left_gripper = left_wrapper.get_ee_pos_rmat_gripper()
                    
                    left_quat = rmat_to_quat(left_rmat.reshape(3, 3))
                    
                    if args.test_rotation:
                        elapsed = time.time() - start_time
                        roll = 0.0
                        pitch = 0.2 * np.sin(elapsed * 0.5)
                        yaw = 0.2 * np.sin(elapsed * 0.3)
                        left_quat = euler_to_quat(roll, pitch, yaw)
                
                right_pos = None
                right_rmat = None
                right_gripper = None
                right_quat = None
                
                if right_wrapper is not None:
                    right_pos, right_rmat, right_gripper = right_wrapper.get_ee_pos_rmat_gripper()
                    
                    right_quat = rmat_to_quat(right_rmat.reshape(3, 3))
                    
                    if args.test_rotation:
                        elapsed = time.time() - start_time
                        roll = 0.1 * np.sin(elapsed * 0.4)
                        pitch = 0.0
                        yaw = 0.3 * np.sin(elapsed * 0.2)
                        right_quat = euler_to_quat(roll, pitch, yaw)
                
                client.send_data(left_pos, left_quat, left_gripper, right_pos, right_quat, right_gripper)
                
                current_time = time.time()
                if current_time - last_print_time >= 2.0:
                    print("-" * 50)
                    if left_pos is not None:
                        print(f"Left Arm: Pose {left_pos}, Quat {left_quat if left_quat is not None else 'None'}, Gripper {left_gripper:.4f}")
                    if right_pos is not None:
                        print(f"Right Arm: Pose {right_pos}, Quat {right_quat if right_quat is not None else 'None'}, Gripper {right_gripper:.4f}")
                    last_print_time = current_time
                
                time.sleep(0.02)
                
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        client.close()
        if left_wrapper is not None:
            left_wrapper.close()
        if right_wrapper is not None:
            right_wrapper.close()