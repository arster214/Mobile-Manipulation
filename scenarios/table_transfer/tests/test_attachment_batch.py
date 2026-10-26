"""Vectorized SAT must preserve contact and arbitrary-rotation scalar decisions."""
from pathlib import Path
import sys,unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.carried_object import boxes_overlap,boxes_overlap_batch
class AttachmentBatchTests(unittest.TestCase):
 def test_rotated_random_boxes(self):
  rng=np.random.default_rng(822);a=np.tile(np.eye(4),(600,1,1));a[:,:3,:3]=Rotation.random(600,random_state=rng).as_matrix();a[:,:3,3]=rng.uniform(-.3,.3,(600,3))
  b=np.eye(4);b[:3,:3]=Rotation.from_euler('xyz',[.2,.4,-.6]).as_matrix();size_a=[.06,.06,.12];size_b=[.25,.18,.1]
  np.testing.assert_array_equal(boxes_overlap_batch(a,size_a,b,size_b),[boxes_overlap(t,size_a,b,size_b) for t in a])
 def test_support_touch_and_tolerance(self):
  a=np.tile(np.eye(4),(4,1,1));a[:,2,3]=[.1,.1-5e-7,.1-2e-6,.2]
  np.testing.assert_array_equal(boxes_overlap_batch(a,[.1]*3,np.eye(4),[.1]*3),[False,False,True,False])
if __name__=='__main__':unittest.main()
