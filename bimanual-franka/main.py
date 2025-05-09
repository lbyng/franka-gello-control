import mujoco
import mujoco.viewer
import numpy as np
import time
import socket
import threading
import json

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785

# Arm Mapping Helper
def map_axis_value(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        normalized = 0.5
    else:
        normalized = (value - src_min) / (src_max - src_min)
    
    mapped_value = dst_min + normalized * (dst_max - dst_min)
    mapped_value = max(dst_min, min(dst_max, mapped_value))
    
    return mapped_value

# Grip Mapping Helper
def map_grip_value(controller_min, controller_max, gripper_min, gripper_max, controller_value):
    if controller_max == controller_min:
        normalized = 0.5
    else:
        normalized = (controller_value - controller_min) / (controller_max - controller_min)
    
    mapped_value = gripper_min + (1 - normalized) * (gripper_max - gripper_min)
    mapped_value = max(gripper_min, min(gripper_max, mapped_value))
    
    return mapped_value

# Left Arm Mapping (tuning needed)
def map_position_left(controller_pos):
    x, y, z = controller_pos
    
    x_sim = map_axis_value(x, -0.45, -0.41, 0.3, 0.8)
    y_sim = map_axis_value(y, 0.3, -0.1, -0.5, -0.1)
    z_sim = map_axis_value(z, -0.1, 0.2, 0.98, 1.8)
    
    return [x_sim, y_sim, z_sim]

# Right Arm Mapping (tuning needed)
def map_position_right(controller_pos):
    x, y, z = controller_pos
    
    x_sim = map_axis_value(x, 0.2, 0.38, 0.3, 0.8)
    y_sim = map_axis_value(y, 0.20, -0.10, 0.1, 0.5)
    z_sim = map_axis_value(z, 0.18, 0.4, 0.98, 1.8)
    
    return [x_sim, y_sim, z_sim]

# Left Grip Mapping (tuning needed)
def map_grip_value_left(controller_value):
    controller_min = 6
    controller_max = -6
    
    gripper_min = 0
    gripper_max = 255
    
    return map_grip_value(controller_min, controller_max, gripper_min, gripper_max, controller_value)

# Right Grip Mapping (tuning needed)
def map_grip_value_right(controller_value):
    controller_min = -15
    controller_max = 1.75
    
    gripper_min = 0
    gripper_max = 255
    
    return map_grip_value(controller_min, controller_max, gripper_min, gripper_max, controller_value)

def map_rotation_quaternion(quat, arm_id="right"):
    """
    Maps input quaternion to a new coordinate frame for intuitive arm control.

    Args:
        quat (list): Original quaternion [w, x, y, z].
        arm_id (str): Arm identifier, either "left" or "right". Defaults to "right".

    Returns:
        list: Mapped quaternion [w, x, y, z].
    """

    # Convert quaternion to Euler angles (ZYX order)
    w, x, y, z = quat
    
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    # Coordinate frame adjustment
    original_roll = roll
    original_pitch = pitch
    original_yaw = yaw

    roll = -original_pitch # align x-axis forward
    pitch = original_roll # align y-axis right
    yaw = original_yaw + np.pi/2 # align z-axis down
    
    # Apply forward/backward tilt amplification
    if pitch > 0:
        pitch = pitch * 1.0
    else:
        pitch = pitch * 1.5

    if roll > 0:
        roll = roll * 1.2
    else:
        roll = roll * 1.0
    
    # Arm-specific remapping
    if arm_id == "left":
        mapped_roll = roll * 1.2
        mapped_pitch = -pitch
        mapped_yaw = yaw * 1.2
    else:
        mapped_roll = roll
        mapped_pitch = -pitch
        mapped_yaw = yaw
    
    # Convert adjusted Euler angles back to quaternion (ZYX order)
    cy = np.cos(mapped_yaw * 0.5)
    sy = np.sin(mapped_yaw * 0.5)
    cp = np.cos(mapped_pitch * 0.5)
    sp = np.sin(mapped_pitch * 0.5)
    cr = np.cos(mapped_roll * 0.5)
    sr = np.sin(mapped_roll * 0.5)
    
    mapped_w = cr * cp * cy + sr * sp * sy
    mapped_x = sr * cp * cy - cr * sp * sy
    mapped_y = cr * sp * cy + sr * cp * sy
    mapped_z = cr * cp * sy - sr * sp * cy
    
    return [mapped_w, mapped_x, mapped_y, mapped_z]


class BimanualArmServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.client_thread = None
        
        self.latest_data = {
            'left': {
                'pos': [0.6, -0.2, 1.0], 
                'grip': 0.02, 
                'raw_grip': 0.0, 
                'raw_pos': [0.5, 0.0, 0.2],
                'quat': [1.0, 0.0, 0.0, 0.0],
                'raw_quat': [1.0, 0.0, 0.0, 0.0]
            },
            'right': {
                'pos': [0.6, 0.2, 1.0], 
                'grip': 0.02, 
                'raw_grip': 0.0, 
                'raw_pos': [0.5, 0.0, 0.2],
                'quat': [1.0, 0.0, 0.0, 0.0],
                'raw_quat': [1.0, 0.0, 0.0, 0.0]
            }
        }
        self.lock = threading.Lock()
        
    def start(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            
            self.running = True
            print(f"Server Started: {self.host}:{self.port}")
            
            self.accept_thread = threading.Thread(target=self._accept_connections)
            self.accept_thread.daemon = True
            self.accept_thread.start()
            
            return True
        except Exception as e:
            print(f"Server Start Error: {e}")
            return False
    
    def _accept_connections(self):
        while self.running:
            try:
                print("Waiting for controller connect...")
                client_socket, client_addr = self.server_socket.accept()
                print(f"Controller Connected: {client_addr}")
                
                if self.client_thread and self.client_thread.is_alive():
                    self.client_socket.close()
                
                self.client_socket = client_socket
                self.client_thread = threading.Thread(target=self._handle_client, args=(client_socket,))
                self.client_thread.daemon = True
                self.client_thread.start()
            except Exception as e:
                if self.running:
                    print(f"Receive Connect Error: {e}")
                    time.sleep(1)
    
    def _handle_client(self, client_socket):
        try:
            while self.running:
                data = client_socket.recv(1024)
                if not data:
                    print("Client Disconnected")
                    break
                
                try:
                    controller_data = json.loads(data.decode('utf-8'))
                    
                    with self.lock:
                        if 'right' in controller_data:
                            if 'pos' in controller_data['right']:
                                self.latest_data['right']['raw_pos'] = controller_data['right']['pos']
                                self.latest_data['right']['pos'] = map_position_right(controller_data['right']['pos'])
                            
                            if 'grip' in controller_data['right']:
                                self.latest_data['right']['raw_grip'] = controller_data['right']['grip']
                                self.latest_data['right']['grip'] = map_grip_value_right(controller_data['right']['grip'])
                                
                            if 'quat' in controller_data['right']:
                                self.latest_data['right']['raw_quat'] = controller_data['right']['quat']
                                self.latest_data['right']['quat'] = map_rotation_quaternion(
                                    controller_data['right']['quat'], "right")
                        
                        if 'left' in controller_data:
                            if 'pos' in controller_data['left']:
                                self.latest_data['left']['raw_pos'] = controller_data['left']['pos']
                                self.latest_data['left']['pos'] = map_position_left(controller_data['left']['pos'])
                            
                            if 'grip' in controller_data['left']:
                                self.latest_data['left']['raw_grip'] = controller_data['left']['grip']
                                self.latest_data['left']['grip'] = map_grip_value_left(controller_data['left']['grip'])
                                
                            if 'quat' in controller_data['left']:
                                self.latest_data['left']['raw_quat'] = controller_data['left']['quat']
                                self.latest_data['left']['quat'] = map_rotation_quaternion(
                                    controller_data['left']['quat'], "left")
                                
                except json.JSONDecodeError:
                    print("invalid JSON data")
                except Exception as e:
                    print(f"Data Process Error: {e}")
        except Exception as e:
            print(f"Client Process Error: {e}")
        finally:
            client_socket.close()
    
    def get_arm_data(self, arm_id):
        with self.lock:
            return self.latest_data.get(arm_id, {}).copy()
    
    def stop(self):
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        print("Server Stopped")

def main(host='localhost', port=12345, model_path=None):
    if model_path is None:
        model_path = "/Users/jiafeid/Downloads/bimanual-franka-cubepickup_task_submission/resources/bi-franka/scene_with_line_and_2_cubes.xml"
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"Model Load Error: {e}")
        print("Please check model path")
        return
        
    data = mujoco.MjData(model)

    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    server = BimanualArmServer(host, port)
    if not server.start():
        print("Server Starts Error")
        return

    try:
        left_joint_names = [f"fa-joint{i}" for i in range(1, 8)]
        left_actuator_names = [f"fa-actuator{i}" for i in range(1, 8)]

        right_joint_names = [f"fb-joint{i}" for i in range(1, 8)]
        right_actuator_names = [f"fb-actuator{i}" for i in range(1, 8)]

        dof_ids_left = np.array([model.joint(name).qposadr[0] for name in left_joint_names])
        dof_ids_right = np.array([model.joint(name).qposadr[0] for name in right_joint_names])

        actuator_ids_left = np.array([model.actuator(name).id for name in left_actuator_names])
        actuator_ids_right = np.array([model.actuator(name).id for name in right_actuator_names])

        key_id = model.key("home").id
        q0_left = model.key("home").qpos[dof_ids_left]
        q0_right = model.key("home").qpos[dof_ids_right]

        mocap_id_left = model.body("target_left").mocapid[0]
        mocap_id_right = model.body("target_right").mocapid[0]

        jac_left = np.zeros((6, model.nv))
        jac_right = np.zeros((6, model.nv))
        diag = damping * np.eye(6)
        twist_left = np.zeros(6)
        twist_right = np.zeros(6)
        body_quat_left = np.zeros(4)
        body_quat_right = np.zeros(4)
        body_quat_conj_left = np.zeros(4)
        body_quat_conj_right = np.zeros(4)
        error_quat_left = np.zeros(4)
        error_quat_right = np.zeros(4)
        dq = np.zeros(model.nv)

        left_gripper_actuator = model.actuator("fa-actuator8").id
        right_gripper_actuator = model.actuator("fb-actuator8").id

        initial_left_pos = np.array([0.6, -0.2, 1.0])
        initial_right_pos = np.array([0.6, 0.2, 1.0])
        initial_quat = np.array([1.0, 0.0, 0.0, 0.0])

        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            
            data.mocap_pos[mocap_id_left] = initial_left_pos
            data.mocap_pos[mocap_id_right] = initial_right_pos
            data.mocap_quat[mocap_id_left] = initial_quat
            data.mocap_quat[mocap_id_right] = initial_quat
            
            mujoco.mj_forward(model, data)
            
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            
            body_id_left = model.body("fa-hand").id
            body_id_right = model.body("fb-hand").id
            
            print("Start to receive bimanual input...")
  
            start_time = time.time()
            last_print_time = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                left_arm_data = server.get_arm_data('left')
                right_arm_data = server.get_arm_data('right')
                
                data.mocap_pos[mocap_id_left] = left_arm_data['pos']
                data.mocap_pos[mocap_id_right] = right_arm_data['pos']
                
                data.mocap_quat[mocap_id_left] = left_arm_data['quat']
                data.mocap_quat[mocap_id_right] = right_arm_data['quat']
                
                data.ctrl[left_gripper_actuator] = left_arm_data['grip']
                data.ctrl[right_gripper_actuator] = right_arm_data['grip']
                
                # --- Left Hand ---
                dx_left = data.mocap_pos[mocap_id_left] - data.body("fa-hand").xpos
                twist_left[:3] = Kpos * dx_left / integration_dt
                body_quat_left[:] = data.body("fa-hand").xquat
                mujoco.mju_negQuat(body_quat_conj_left, body_quat_left)
                mujoco.mju_mulQuat(error_quat_left, data.mocap_quat[mocap_id_left], body_quat_conj_left)
                mujoco.mju_quat2Vel(twist_left[3:], error_quat_left, 1.0)
                twist_left[3:] *= Kori / integration_dt

                mujoco.mj_jacBody(model, data, jac_left[:3], jac_left[3:], body_id_left)
                dq[dof_ids_left] = jac_left[:, dof_ids_left].T @ np.linalg.solve(
                    jac_left[:, dof_ids_left] @ jac_left[:, dof_ids_left].T + diag, twist_left
                )

                dq[dof_ids_left] += (np.eye(7) - np.linalg.pinv(jac_left[:, dof_ids_left]) @ jac_left[:, dof_ids_left]) @ (
                    Kn * (q0_left - data.qpos[dof_ids_left])
                )
                
                # --- Right Hand ---
                dx_right = data.mocap_pos[mocap_id_right] - data.body("fb-hand").xpos
                twist_right[:3] = Kpos * dx_right / integration_dt
                body_quat_right[:] = data.body("fb-hand").xquat
                mujoco.mju_negQuat(body_quat_conj_right, body_quat_right)
                mujoco.mju_mulQuat(error_quat_right, data.mocap_quat[mocap_id_right], body_quat_conj_right)
                mujoco.mju_quat2Vel(twist_right[3:], error_quat_right, 1.0)
                twist_right[3:] *= Kori / integration_dt

                mujoco.mj_jacBody(model, data, jac_right[:3], jac_right[3:], body_id_right)
                dq[dof_ids_right] = jac_right[:, dof_ids_right].T @ np.linalg.solve(
                    jac_right[:, dof_ids_right] @ jac_right[:, dof_ids_right].T + diag, twist_right
                )

                dq[dof_ids_right] += (np.eye(7) - np.linalg.pinv(jac_right[:, dof_ids_right]) @ jac_right[:, dof_ids_right]) @ (
                    Kn * (q0_right - data.qpos[dof_ids_right])
                )

                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

                q = data.qpos.copy()
                mujoco.mj_integratePos(model, q, dq, integration_dt)

                data.ctrl[actuator_ids_left] = q[dof_ids_left]
                data.ctrl[actuator_ids_right] = q[dof_ids_right]

                mujoco.mj_step(model, data)
                viewer.sync()

                current_time = time.time()
                if current_time - last_print_time >= 2.0:
                    elapsed_time = current_time - start_time

                    left_raw_pos = left_arm_data.get('raw_pos', ['N/A']*3)
                    left_mapped_pos = left_arm_data['pos']
                    left_raw_grip = left_arm_data.get('raw_grip', 'N/A')
                    left_mapped_grip = left_arm_data['grip']
                    left_raw_quat = left_arm_data.get('raw_quat', ['N/A']*4)
                    left_mapped_quat = left_arm_data.get('quat', ['N/A']*4)
                    
                    right_raw_pos = right_arm_data.get('raw_pos', ['N/A']*3)
                    right_mapped_pos = right_arm_data['pos']
                    right_raw_grip = right_arm_data.get('raw_grip', 'N/A')
                    right_mapped_grip = right_arm_data['grip']
                    right_raw_quat = right_arm_data.get('raw_quat', ['N/A']*4)
                    right_mapped_quat = right_arm_data.get('quat', ['N/A']*4)
                    
                    print(f"Time: {elapsed_time:.1f}s")
                    print("Left Arm:")
                    print(f"  Pose: Raw {left_raw_pos} -> Mapping {left_mapped_pos}")
                    print(f"  Quat: Raw {left_raw_quat} -> Mapping {left_mapped_quat}")
                    print(f"  Grip: Raw {left_raw_grip} -> Mapping {left_mapped_grip:.4f}")
                    print("Right Arm:")
                    print(f"  Pose: Raw {right_raw_pos} -> Mapping {right_mapped_pos}")
                    print(f"  Quat: Raw {right_raw_quat} -> Mapping {right_mapped_quat}")
                    print(f"  Grip: Raw {right_raw_grip} -> Mapping {right_mapped_grip:.4f}")
                    print("-------------------------------------------------")

                    last_print_time = current_time

                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except Exception as e:
        print(f"Simulator Running Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        server.stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Franka dual arms simulator server")
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--model_path', type=str, default='/Users/jiafeid/Downloads/bimanual-franka-cubepickup_task_submission/resources/bi-franka/scene_with_line_and_2_cubes.xml')
    
    args = parser.parse_args()
    
    main(args.host, args.port, args.model_path)