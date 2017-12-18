import time
import face
f = face.Face()


for v in range(10,128,15):
    print("Speed = ",v)	
    f.setSpeedEyes(v)
    f.setSpeedHead(v)
    f.moveHead(0,0)
    time.sleep(2)
    f.moveHead(800,800)
    time.sleep(2)
print("Test for setSpeedHead completed successfully")
