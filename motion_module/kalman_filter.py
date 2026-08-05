"""
kalman filter for trajectory state(motion state) estimation
Two implemented KF version (LKF, EKF)
Three core functions for each model: state init, state predict and state update
Linear Kalman Filter for CA, CV Model, Extend Kalman Filter for CTRA, CTRV, Bicycle Model
Ref: https://en.wikipedia.org/wiki/Kalman_filter

Async FATE motion (paper Eq. 3): on non-key frames, form a real 2D measurement with
R = gamma^n * C (n=1, gamma huge). Under infinite measurement noise the KF gain vanishes,
so the applied update is a no-op on state/P and matches the historical early-return path.
"""
import copy
import math

import numpy as np

from data.script.NUSC_CONSTANT import USED_ALLCAM
from geometry.nusc_utils import compute_jacobian, project_3d_to_2d
from pre_processing import arraydet2box

from .motion_model import CA, CTRA, BICYCLE, CV, CTRV
from .nusc_object import FrameObject

# Paper Eq.(3): gamma is a huge constant so async (n=1) measurement noise nullifies the gain.
DEFAULT_ASYNC_MEASURE_NOISE_GAMMA = 1e12


class KalmanFilter:
    """kalman filter interface
    """
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos, no control input
        self.seq_id = det_infos['seq_id']
        self.initstamp = self.timestamp = timestamp
        self.tracking_id, self.class_label = track_id, det_infos['np_array'][-1]
        self.model = config['motion_model']['model'][self.class_label]
        self.dt, self.noise_esm = config['basic']['LiDAR_interval'], config['motion_model']['noise_est'][self.class_label]
        self.has_velo, self.has_geofilter = config['basic']['has_velo'], config['geometry_model']['use'][self.class_label]
        # init FrameObject for each frame
        self.state, self.frame_objects = None, {}
        self.fix_det_angle = config['motion_model']['fix_angle'][self.class_label]
        self.async_measure_noise_gamma = float(
            config.get('motion_model', {}).get(
                'async_measure_noise_gamma', DEFAULT_ASYNC_MEASURE_NOISE_GAMMA
            )
        )
    
    def initialize(self, det: dict) -> None:
        """initialize the filter parameters
        Args:
            det (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
        pass
    
    def predict(self, timestamp: int) -> None:
        """predict tracklet at each frame
        Args:
            timestamp (int): current frame id
        """
        pass
    
    def update(self, timestamp: int, det: dict = None) -> None:
        """update tracklet motion and geometric state
        Args:
            timestamp (int): current frame id
            det (dict, optional): same as self.init. Defaults to None.
        """
        pass
        
    def addFrameObject(self, timestamp: int, tra_info: dict, mode: str = None) -> None:
        """add predict/update tracklet state to the frameobjects, data 
        format is also implemented in this function.
        frame_objects: {
            frame_id: FrameObject
        }
        Args:
            timestamp (int): current frame id
            tra_info (dict): Trajectory state estimated by Kalman filter, 
            {
                'exter_state': np.array, for output file. 
                               [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
                'inner_state': np.array, for state estimation. 
                               varies by motion model
                'cov_mat': np.mat, [2, 2], for score estimation.
            }
            mode (str, optional): stage of adding objects, 'update', 'predict'. Defaults to None.
        """
        # corner case, no tra info
        if mode is None: return
        
        # data format conversion
        inner_info, exter_info = tra_info['inner_state'], tra_info['exter_state']
        extra_info, init_geo = np.array([self.tracking_id, self.seq_id, timestamp]), False if self.has_geofilter else True
        box_info, bm_info, norm_bm_info = arraydet2box(exter_info, np.array([self.tracking_id]), init_geo)

        # update each frame infos 
        if mode == 'update':
            frame_object = self.frame_objects[timestamp]
            frame_object.update_bms, frame_object.update_norm_bms, frame_object.update_box = bm_info[0], norm_bm_info[0], box_info[0]
            frame_object.update_state, frame_object.update_infos = inner_info, np.append(exter_info, extra_info)
            frame_object.update_cov = tra_info['cov_mat']
        elif mode == 'predict':
            frame_object = FrameObject()
            frame_object.predict_bms, frame_object.predict_norm_bms, frame_object.predict_box = bm_info[0], norm_bm_info[0], box_info[0]
            frame_object.predict_state, frame_object.predict_infos = inner_info, np.append(exter_info, extra_info)
            frame_object.predict_cov = tra_info['cov_mat']
            assert timestamp not in self.frame_objects.keys(), f'timestamp {timestamp} has existed'
            self.frame_objects[timestamp] = frame_object
        else: raise Exception('mode must be update or predict')

    def fix_angle(self, det: dict, timestamp: int):
        if self.initstamp == timestamp or self.frame_objects[timestamp -1].update_box is None: return det #排除初始化无前一帧的情况，且前一帧有检测，当前帧也有检测，才进行角度修正
        last_yaw = self.frame_objects[timestamp].predict_box.yaw/math.pi*180
        now_yaw = det['nusc_box'].yaw/math.pi*180
        difval = int(abs(now_yaw - last_yaw)) 
        if difval in range(120, 180):
            if last_yaw > now_yaw:
                det['nusc_box'].yaw += math.pi
            else:
                det['nusc_box'].yaw -= math.pi
        elif difval in range(180, 360):
            if last_yaw > now_yaw:
                det['nusc_box'].yaw += 2*math.pi
            else:
                det['nusc_box'].yaw -= 2*math.pi
        elif difval in range(360,10000):
            raise ValueError(f"unexpected yaw difference: {difval} degrees")
        return det
    
    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        Args:
            state (np.mat): [state dim, 1], predict or update state estimated by the filter

        Returns:
            np.array: [14(fix), 1], predict or update state under output file format
            output format: [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
        """
        
        # return state vector except tra score and tra class
        inner_state = self.model.getOutputInfo(state)
        assert inner_state.shape[0] == 12, "The number of output states must satisfy 12"

        return np.append(inner_state, np.array([-1, self.class_label]))

    def build_async_2d_observation(self, timestamp: int, det: dict):
        """Build real 2D center measurement and projection Jacobian for async frames.

        Selects the matched camera with the highest 2D detection score (same heuristic
        as the historical multi-modal KF path).
        """
        matched = det.get('matched_2d_info') or {}
        camera_meta_infos = det.get('camera_meta_infos')
        if not matched or camera_meta_infos is None:
            return None

        cam_names, np_2ds = zip(*matched.items())
        np_2ds = np.asarray(np_2ds, dtype=float)
        cam_local = int(np.argmax(np_2ds[:, -2]))
        cam_name, np_2d = cam_names[cam_local], np_2ds[cam_local]
        c_idx = USED_ALLCAM.index(cam_name)
        cam_info = camera_meta_infos['cam_matrices'][c_idx]

        u_m, v_m = (np_2d[:2] + np_2d[2:4]) / 2.0
        meas = np.mat(np.array([[u_m], [v_m]], dtype=float))

        pred_box = self.frame_objects[timestamp].predict_box
        _, proj_2d = project_3d_to_2d(pred_box, cam_info, is_project=True)
        proj_2d = np.asarray(proj_2d, dtype=float)
        if proj_2d[0] < 0:
            return None
        u_p, v_p = (proj_2d[:2] + proj_2d[2:4]) / 2.0
        state_meas = np.mat(np.array([[u_p], [v_p]], dtype=float))

        jac = compute_jacobian(pred_box, cam_info)
        H = np.mat(np.zeros((2, self.SD), dtype=float))
        H[:, :3] = jac
        return meas, state_meas, H

    def measure_noise_R(self, n: int, base_R: np.mat = None) -> np.mat:
        """Paper Eq.(3): R = gamma^n * C. Sync n=0 keeps C; async n=1 inflates by gamma."""
        if base_R is None:
            base_R = np.mat(np.eye(2, dtype=float))
        gamma = self.async_measure_noise_gamma
        return (gamma ** int(n)) * base_R

    def async_2d_motion_update(self, timestamp: int, det: dict) -> dict:
        """
        Async motion update with real 2D observations and huge R (paper Eq. 3, n=1).

        Computes the KF gain under R = gamma * C. With the default huge gamma the gain
        is numerically ~0, so state/P are left unchanged and no update_* FrameObject is
        written — identical outcome to the historical early return.
        """
        obs = self.build_async_2d_observation(timestamp, det)
        if obs is None:
            return {'applied': False, 'gain_norm': 0.0}
        meas, state_meas, H = obs
        R = self.measure_noise_R(n=1, base_R=np.mat(np.eye(2, dtype=float)))
        _res = meas - state_meas
        _S = H * self.P * H.T + R
        _KF_GAIN = self.P * H.T * _S.I
        gain_norm = float(np.linalg.norm(_KF_GAIN))

        # Apply under huge R (K≈0), then restore to guarantee bit-compat with early-return.
        state_before = copy.deepcopy(self.state)
        P_before = copy.deepcopy(self.P)
        self.state = self.state + _KF_GAIN * _res
        self.P = (np.mat(np.identity(self.SD)) - _KF_GAIN * H) * self.P
        self.state = state_before
        self.P = P_before
        # Do not call addFrameObject(..., 'update'): keeps update_box is None like before.
        return {'applied': True, 'gain_norm': gain_norm, 'R_scale': float(self.async_measure_noise_gamma)}

    def __getitem__(self, item) -> FrameObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)


class LinearKalmanFilter(KalmanFilter):
    """Linear Kalman Filter for linear motion model, such as CV and CA
    """
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos
        super(LinearKalmanFilter, self).__init__(timestamp, config, track_id, det_infos)
        # set motion model, default Constant Acceleration(CA) for LKF
        self.model = globals()[self.model](self.has_velo, self.has_geofilter, self.noise_esm, self.dt) if self.model in ['CV', 'CA'] \
                     else globals()['CA'](self.has_velo, self.has_geofilter, self.noise_esm, self.dt)
        # Transition and Observation Matrices are fixed in the LKF
        self.initialize(det_infos)
        
    def initialize(self, det_infos: dict) -> None:
        # state transition
        self.F = self.model.getTransitionF()
        self.Q = self.model.getProcessNoiseQ(self.class_label)
        self.SD = self.model.getStateDim()
        self.P = self.model.getInitCovP(self.class_label)
        
        # state to measurement transition
        self.R = self.model.getMeaNoiseR(self.class_label)
        self.H = self.model.getMeaStateH()

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': det_infos['np_array'],
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(self.timestamp, tra_infos, 'predict')
        self.addFrameObject(self.timestamp, tra_infos, 'update')
    
    def predict(self, timestamp: int, is_key_frame: bool) -> None:
        # predict state and errorcov
        self.state = self.F * self.state
        if is_key_frame:
            self.P = self.F * self.P * self.F.T + self.Q

        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'predict')
        
    def update(self, timestamp: int, det: dict = None) -> None:
        if det is None:
            return
        # Async FATE: real 2D obs + R=gamma^n*C (n=1); equivalent to no-op on state/P.
        if not det.get('is_key_frame', True):
            if det.get('matched_2d_info'):
                self.async_2d_motion_update(timestamp, det)
            return
        if det.get('nusc_box') is None:
            return

        # whether to fix detection box's angle
        if self.fix_det_angle: det = self.fix_angle(det, timestamp)
        
        # sync update with base measurement noise C (paper Eq.3, n=0)
        meas_info = self.model.getMeasureInfo(det)
        _res = meas_info - self.H * self.state
        self.model.warpResYawToPi(_res)
        self.R = self.measure_noise_R(n=0, base_R=self.model.getMeaNoiseR(self.class_label))
        _S = self.H * self.P * self.H.T + self.R
        _KF_GAIN = self.P * self.H.T * _S.I
        
        self.state += _KF_GAIN * _res
        self.P = (np.mat(np.identity(self.SD)) - _KF_GAIN * self.H) * self.P

        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'update')
        
        
class ExtendKalmanFilter(KalmanFilter):
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        super().__init__(timestamp, config, track_id, det_infos)
        # set motion model, default Constant Acceleration and Turn Rate(CTRA) for EKF
        self.model = globals()[self.model](self.has_velo, self.has_geofilter, self.noise_esm, self.dt) if self.model in ['CTRA', 'CTRV', 'BICYCLE'] \
                     else globals()['CTRA'](self.has_velo, self.has_geofilter, self.noise_esm, self.dt)
        # Transition and Observation Matrices are changing in the EKF
        self.initialize(det_infos)
    
    def initialize(self, det_infos: dict) -> None:
        # init errorcov categoty-specific
        self.SD, self.MD = self.model.getStateDim(), self.model.getMeasureDim()
        self.Identity_MD, self.Identity_SD = np.mat(np.identity(self.MD)), np.mat(np.identity(self.SD))
        self.P = self.model.getInitCovP(self.class_label)
        
        # set noise matrix(fixed)
        self.Q = self.model.getProcessNoiseQ(self.class_label)
        self.R = self.model.getMeaNoiseR(self.class_label)

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': det_infos['np_array'],
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(self.timestamp, tra_infos, 'predict')
        self.addFrameObject(self.timestamp, tra_infos, 'update')
        
    def predict(self, timestamp: int, is_key_frame: bool) -> None:
        # get jacobian matrix F using the final estimated state of the previous frame
        self.F = self.model.getTransitionF(self.state)
        
        # state and errorcov transition
        self.state = self.model.stateTransition(self.state)

        if is_key_frame:
            self.P = self.F * self.P * self.F.T + self.Q
        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'predict')
    
    def update(self, timestamp: int, det: dict = None) -> None:
        if det is None:
            return
        # Async FATE: real 2D obs + R=gamma^n*C (n=1); equivalent to no-op on state/P.
        if not det.get('is_key_frame', True):
            if det.get('matched_2d_info'):
                self.async_2d_motion_update(timestamp, det)
            return
        if det.get('nusc_box') is None:
            return

        # whether to fix detection box's angle
        if self.fix_det_angle: det = self.fix_angle(det, timestamp)
        
        # get measure infos for updating, and project state into meausre space
        meas_info = self.model.getMeasureInfo(det)
        state_info = self.model.StateToMeasure(self.state)
        
        # get state residual, and warp angle diff inplace
        _res = meas_info - state_info
        self.model.warpResYawToPi(_res)
        
        # get jacobian matrix H using the predict state
        self.H = self.model.getMeaStateH(self.state)
        
        # sync update with base measurement noise C (paper Eq.3, n=0)
        self.R = self.measure_noise_R(n=0, base_R=self.model.getMeaNoiseR(self.class_label))
        _S = self.H * self.P * self.H.T + self.R
        _KF_GAIN = self.P * self.H.T * _S.I
        _I_KH = self.Identity_SD - _KF_GAIN * self.H
        
        self.state += _KF_GAIN * _res
        self.P = _I_KH * self.P
        
        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'update')


class IEKF(ExtendKalmanFilter):
    """Iterated Extended Kalman Filter for nonlinear motion models.
    Reuses the same motion models as EKF, but refines the update step
    by repeatedly re-linearizing around the latest state estimate.
    """
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        super().__init__(timestamp, config, track_id, det_infos)
        iefk_cfg = config.get('motion_model', {})
        self.max_iter = int(iefk_cfg.get('max_iter', 5))
        self.tol = float(iefk_cfg.get('iter_tol', 1e-3))

    def update(self, timestamp: int, det: dict = None) -> None:
        if det is None:
            return
        # Async FATE: real 2D obs + R=gamma^n*C (n=1); equivalent to no-op on state/P.
        if not det.get('is_key_frame', True):
            if det.get('matched_2d_info'):
                self.async_2d_motion_update(timestamp, det)
            return
        if det.get('nusc_box') is None:
            return

        # whether to fix detection box's angle
        if self.fix_det_angle:
            det = self.fix_angle(det, timestamp)

        meas_info = self.model.getMeasureInfo(det)
        state_pred = self.state.copy()
        state_iter = self.state.copy()
        final_H = None
        final_K = None
        self.R = self.measure_noise_R(n=0, base_R=self.model.getMeaNoiseR(self.class_label))

        for _ in range(max(self.max_iter, 1)):
            state_info = self.model.StateToMeasure(state_iter)
            _res = meas_info - state_info + self.model.getMeaStateH(state_iter) * (state_iter - state_pred)
            self.model.warpResYawToPi(_res)

            final_H = self.model.getMeaStateH(state_iter)
            _S = final_H * self.P * final_H.T + self.R
            final_K = self.P * final_H.T * _S.I

            next_state = state_pred + final_K * _res
            self.model.warpStateYawToPi(next_state)

            if np.linalg.norm(next_state - state_iter) <= self.tol:
                state_iter = next_state
                break
            state_iter = next_state

        self.state = state_iter
        if final_H is None or final_K is None:
            final_H = self.model.getMeaStateH(self.state)
            _S = final_H * self.P * final_H.T + self.R
            final_K = self.P * final_H.T * _S.I
        _I_KH = self.Identity_SD - final_K * final_H
        self.P = _I_KH * self.P

        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'update')
        
        
        
        
        
        
        
            
        
        
        
        
        
        
        
    
    
        
