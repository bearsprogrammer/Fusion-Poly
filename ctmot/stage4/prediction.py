"""All-track, class-independent timestamp policy; no scene/GT/target-ID inputs.

B1 retains each class's original model, changes mean time only.
B2/B3 retain B1 yaw/shape and replace geometric center with the CT ODE result.
Original covariance schedule/noise and KF correction remain unchanged.
"""
import math
import numpy as np

from ctmot.proto.association_replay import tracker_module
from pre_processing import arraydet2box
from ctmot.proto.predictor import ODEPredictor
from ctmot.proto.dynamics import ConstantVelocity, ConstantAcceleration
from ctmot.proto.history import estimate_acceleration


def physical_pv(model, state):
    if model.has_geofilter:
        raise ValueError('Geometry-filter adapter not implemented')
    s=np.asarray(state).ravel();kind=type(model).__name__
    p=np.array(model.getOutputInfo(state)[:3])
    if kind=='CTRV':
        v=np.array([s[6]*math.cos(s[7]),s[6]*math.sin(s[7]),0.])
    elif kind=='CV':
        v=np.r_[s[6:8], 0.]  # Original CV transition freezes Z and zeros vz.
    elif kind=='BICYCLE':
        speed,theta,sigma=s[6],s[8],s[9]
        beta,_,lr=model.getBicBeta(s[4],sigma)
        omega=speed/lr*math.sin(beta) if abs(beta)>.001 else 0.
        offset=model.graToGeoDist(s[4])
        # p_geo = p_gravity - offset * [cos(theta), sin(theta)].
        v=np.array([speed*math.cos(theta+beta)+offset*math.sin(theta)*omega,
                    speed*math.sin(theta+beta)-offset*math.cos(theta)*omega,0.])
    else:
        raise ValueError('Unsupported original model '+kind)
    return np.r_[p,v]


def actual_transition(model,state,delta_t):
    if not np.isfinite(delta_t) or delta_t<=0: raise ValueError('Prediction requires increasing timestamps')
    nominal=model.dt
    try:
        model.dt=delta_t
        if type(model).__name__=='CV':
            result=model.getTransitionF()*state
        else:
            result=model.stateTransition(state)
        model.warpStateYawToPi(result)
        return result
    finally:
        model.dt=nominal


def set_geometric_center(model,state,center):
    state=state.copy()
    if type(model).__name__=='BICYCLE':
        state[:2]=np.mat(model.geoCenterToGraCenter(center[:2],float(state[8,0]),float(state[4,0]))).T
        state[2,0]=center[2]
    else:
        state[:3]=np.mat(center).T
    return state


def install_mean(tracker,tid,state):
    motion=tracker.valid_tras[tid].motion_management;frame=motion[tracker.frame_id]
    motion.state=state;frame.predict_state=state
    row=motion.getOutputInfo(state);row[12]=frame.predict_infos[12]
    boxes,bottoms,normals=arraydet2box(row,np.array([tid]),True)
    frame.predict_infos=np.r_[row,frame.predict_infos[14:]]
    frame.predict_box,frame.predict_bms,frame.predict_norm_bms=boxes[0],bottoms[0],normals[0]
    i=tracker._index_by_id[tid]
    for key,value in [('np_tras',frame.predict_infos),('np_tras_bottom_corners',bottoms[0]),
                      ('np_tras_norm_corners',normals[0]),('all_valid_boxes',boxes[0])]:
        tracker.tra_infos[key][i]=value


class AllTrackPredictor(tracker_module.Tracker):
    def __init__(self,tracker_config,variant,experiment):
        super().__init__(tracker_config,None)
        if variant not in ('B0','B1','B2','B3'): raise ValueError(variant)
        self.variant,self.experiment=variant,experiment
        self.anchors,self.histories={},{}
        self.previous_timestamp=None
        self.counters={'predictions':0,'ca_predictions':0,'ca_fallback_cv':0,'matched_3d':0}
        self.cv=ODEPredictor(ConstantVelocity(),**experiment['solver'])

    def advance(self,data,timestamp_us):
        self.query_timestamp=int(timestamp_us)
        if data['is_first_frame']:
            self.anchors,self.histories={},{};self.previous_timestamp=None
        if self.previous_timestamp is not None and self.query_timestamp<=self.previous_timestamp:
            raise ValueError('Actual timestamps must increase')
        self.tracking(data)
        self.previous_timestamp=self.query_timestamp

    def tras_predict(self):
        previous={tid:t.motion_management.state.copy() for tid,t in self.valid_tras.items()}
        super().tras_predict()
        if self.variant=='B0' or not previous: return
        dt=(self.query_timestamp-self.previous_timestamp)/1e6
        self._index_by_id={int(tid):i for i,tid in enumerate(self.tra_infos['all_valid_ids'])}
        for tid,state in previous.items():
            motion=self.valid_tras[tid].motion_management
            mean=actual_transition(motion.model,state,dt)
            if self.variant in ('B2','B3'):
                anchor=self.anchors[tid]
                predictor=self.cv
                if self.variant=='B3':
                    history=self.histories.get(tid,[])[-self.experiment['history_observations']:]
                    accel=estimate_acceleration(history,anchor['timestamp_us'],self.experiment['minimum_observations'])
                    if accel['status']=='ok':
                        predictor=ODEPredictor(ConstantAcceleration(accel['acceleration_mps2']),**self.experiment['solver'])
                        self.counters['ca_predictions']+=1
                    else:
                        self.counters['ca_fallback_cv']+=1
                prediction=predictor.predict(anchor['pv'],anchor['timestamp_us'],self.query_timestamp)
                mean=set_geometric_center(motion.model,mean,prediction.state[:3])
            install_mean(self,tid,mean)
            self.counters['predictions']+=1

    def tras_update(self,association,data):
        old_ids=list(self.valid_tras);accepted={}
        for key,det_key in [('m_asso_1','mix_np_dets_3d'),('m_asso_2','single_np_dets_3d')]:
            for di,ti in association[key].items():
                accepted[old_ids[ti]]=data[det_key][di].copy()
        result=super().tras_update(association,data)
        if self.variant=='B0': return result
        for tid,track in self.valid_tras.items():
            if tid in accepted:
                det=accepted[tid]
                self.histories.setdefault(tid,[]).append(dict(timestamp_us=self.query_timestamp,
                    velocity_xy_mps=det[6:8].tolist(),position_m=det[:3].tolist()))
                self.histories[tid]=self.histories[tid][-self.experiment['history_observations']:]
                self.counters['matched_3d']+=1
            # Only birth or a real 3D correction establishes a new posterior anchor.
            # Camera-only updates in this repository leave the motion mean/P unchanged.
            if tid not in self.anchors or tid in accepted:
                self.anchors[tid]=dict(timestamp_us=self.query_timestamp,
                    pv=physical_pv(track.motion_management.model,track.motion_management.state))
        for tid in set(self.anchors)-set(self.valid_tras):
            self.anchors.pop(tid,None);self.histories.pop(tid,None)
        return result
