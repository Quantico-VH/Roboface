import time
import face
f = face.Face()


f.setSpeedAll(100)
f.setSpeedHead(50)
f.neutral()
time.sleep(2)
f.angry()
time.sleep(1)
f.neutral()
time.sleep(2)
f.happy()
time.sleep(1)
f.neutral()
time.sleep(2)
f.sad()
time.sleep(1)
f.neutral()
time.sleep(2)
f.happy()
time.sleep(1)
f.neutral()
time.sleep(2)
f.unsure()
time.sleep(1)

