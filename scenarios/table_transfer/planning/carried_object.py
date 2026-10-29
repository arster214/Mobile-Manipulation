"""Rigid TCP attachment and conservative oriented-box collision auditing."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    result=np.eye(4)
    result[:3,:3]=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix()
    result[:3,3]=pose[:3]
    return result


def matrix_pose(matrix):
    return np.r_[matrix[:3,3],Rotation.from_matrix(matrix[:3,:3]).as_quat(scalar_first=True)]


def boxes_overlap(a, size_a, b, size_b, tolerance=1e-6):
    """SAT for two oriented boxes; touching support surfaces are not penetration."""
    axes=[*a[:3,:3].T,*b[:3,:3].T]
    axes += [np.cross(x,y) for x in a[:3,:3].T for y in b[:3,:3].T]
    delta=b[:3,3]-a[:3,3]
    for axis in axes:
        norm=np.linalg.norm(axis)
        if norm<1e-10:continue
        axis=axis/norm
        radius=np.abs(a[:3,:3].T@axis)@(np.asarray(size_a)/2)+np.abs(b[:3,:3].T@axis)@(np.asarray(size_b)/2)
        if radius-abs(delta@axis)<=tolerance:return False
    return True


def boxes_overlap_batch(transforms, size_a, other, size_b, tolerance=1e-6):
    """Vectorized SAT; same axes, touching rule and tolerance as boxes_overlap."""
    rotations = transforms[:,:3,:3]
    axes_a = rotations.transpose(0,2,1)
    axes_b = np.broadcast_to(other[:3,:3].T,axes_a.shape)
    cross = np.cross(axes_a[:,:,None,:],axes_b[:,None,:,:]).reshape(len(transforms),9,3)
    axes = np.concatenate([axes_a,axes_b,cross],axis=1)
    norms = np.linalg.norm(axes,axis=-1)
    valid = norms >= 1e-10
    axes = axes / np.where(valid,norms,1.)[:,:,None]
    radius_a = np.abs(np.einsum('nji,nkj->nki',rotations,axes)) @ (np.asarray(size_a)/2)
    radius_b = np.abs(axes @ other[:3,:3]) @ (np.asarray(size_b)/2)
    distance = np.abs(np.einsum('ni,nki->nk',other[:3,3]-transforms[:,:3,3],axes))
    return ~np.any(valid & (radius_a+radius_b-distance<=tolerance),axis=1)


@dataclass
class CarriedObject:
    tcp_from_object: np.ndarray
    dimensions: np.ndarray
    contact_links: tuple[str,...]

    @classmethod
    def attach(cls, tcp_pose, object_pose, radius, height, arm):
        return cls.attach_box(tcp_pose, object_pose, [2*radius,2*radius,height], arm)

    @classmethod
    def attach_box(cls, tcp_pose, object_pose, dimensions, arm):
        return cls(np.linalg.inv(pose_matrix(tcp_pose))@pose_matrix(object_pose),
                   np.asarray(dimensions, dtype=float),
                   (f'{arm}_Link_finger1',f'{arm}_Link_finger2',f'{arm}_base_link_rmg24'))

    def poses(self, tcp_poses):
        return np.array([pose_matrix(pose)@self.tcp_from_object for pose in tcp_poses])

    def audit(self, planner, positions, environment):
        transforms=self.poses(planner.tcp_pose(positions))
        obstacles=[(name,pose_matrix(box['pose']),box['dims']) for name,box in environment.obstacles['cuboid'].items()]
        hits = []
        for order,(name,other,dims) in enumerate(obstacles):
            frames = np.flatnonzero(boxes_overlap_batch(transforms,self.dimensions,other,dims))
            if len(frames):
                hits.append((int(frames[0]),order,name))
        if hits:
            frame,_,name = min(hits)
            return {'passed':False,'obstacle':name,'sample':frame}
        kin=planner.checker.get_kinematics(planner.tensor(positions[:,planner.to_internal])[None])
        spheres=kin.robot_spheres[0].detach().cpu().numpy()
        params=planner.checker.kinematics.config.kinematics_config
        names=[params.all_link_names[i] for i in params.link_sphere_idx_map.cpu().tolist()]
        enabled=np.array([name not in self.contact_links for name in names])
        local = np.einsum('nsi,nij->nsj',spheres[:,:,:3]-transforms[:,None,:3,3],transforms[:,:3,:3])
        separation = np.maximum(np.abs(local)-self.dimensions/2,0.)
        gaps = np.linalg.norm(separation,axis=-1)-spheres[:,:,3]
        bad = np.argwhere(enabled[None] & (spheres[:,:,3]>0) & (gaps < -1e-6))
        if len(bad):
            frame,index = bad[0]
            return {'passed':False,'robot_link':names[index],'sample':int(frame)}
        return {'passed':True,'samples':len(transforms),'representation':'oriented box'}
