import time
import face
f = face.Face()


for v in range(128):
    print("Speed = ",v)	
    f.setSpeedLips(v)
    f.moveLips(0)
    time.sleep(1)
    f.moveLips(10)
    time.sleep(1)
print("Test for SetSpeedLips completed successfully")
