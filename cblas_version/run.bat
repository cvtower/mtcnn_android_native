@echo off

::adb root
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push libs/armeabi-v7a/mtcnn_baseline /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push data/Onet.txt  /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push data/Pnet.txt  /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push data/Rnet.txt  /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push data/women.jpg /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe shell chmod 777 /data/local/tmp/mtcnn_baseline
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe shell /data/local/tmp/mtcnn_baseline /data/local/tmp/women.jpg
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe pull /data/local/tmp/result.jpg