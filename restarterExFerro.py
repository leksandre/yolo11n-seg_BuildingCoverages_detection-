import subprocess
import time
import random
subprocess.run(["python3.11", "./aexferro.py"])
while True:
    randWaith = random.randint(1,6)
    print('will start after:'+str(randWaith))
    time.sleep(randWaith)
    print('try start')
    if True:
        time.sleep(randWaith)
        subprocess.run(["python3.11", "./aexferro.py"])
