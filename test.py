import time
import face
f = face.Face()


f.setServosSpeed([10,9,6,3],[50,50,50,50])
for i in range(30):
	f.moveLips(0)
	time.sleep(1)
	f.moveLips(10)
	time.sleep(1)
