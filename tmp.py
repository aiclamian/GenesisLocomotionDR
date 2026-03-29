# 打印 robot.joints 的 name
import genesis as gs
from env import Go2Env

gs.init()
e = Go2Env(1, False, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
print([joint.name for joint in e.robot.joints])
print([joint.dof_start for joint in e.robot.joints])
