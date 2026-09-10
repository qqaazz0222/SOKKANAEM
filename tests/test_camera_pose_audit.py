import numpy as np
from scripts.audit_camera_pose import summarize


def poses():
    a=np.zeros((4,8));a[:,0]=np.arange(4)*.03;a[:,7]=1
    return a


def test_stationary_is_not_verified_mount():
    r=summarize(poses(),[0,.03,.06,.09])
    assert r['label']=='low_motion_not_mount_verified'
    assert not r['fixed_mount_verified']


def test_quaternion_sign_does_not_create_rotation():
    a=poses();a[2:,7]=-1
    assert summarize(a)['rotation_max_deg']==0


def test_translation_detected():
    a=poses();a[:,1]=[0,.1,.2,.3]
    assert summarize(a)['label']=='camera_motion_present'


def test_incomplete_coverage_not_classified_stationary():
    r=summarize(poses(),[0,.03,50,100])
    assert r['coverage']==.5 and r['label']=='unclassified_incomplete_pose'


def test_rotation_detected():
    a=poses();a[1:,6]=np.sin(np.deg2rad(5));a[1:,7]=np.cos(np.deg2rad(5))
    assert abs(summarize(a)['rotation_max_deg']-10)<1e-9
