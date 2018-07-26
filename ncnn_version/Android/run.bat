::adb root
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push libs/armeabi-v7a/mtcnn_baseline /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push ./../mtcnn/det1.param  /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push ./../mtcnn/det1.bin  /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push ./../mtcnn/det2.param  /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push ./../mtcnn/det2.bin  /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push ./../mtcnn/det3.param  /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push ./../mtcnn/det3.bin  /data/local/tmp

C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe push data/women.jpg /data/local/tmp
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe shell chmod 777 /data/local/tmp/mtcnn_baseline
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe shell /data/local/tmp/mtcnn_baseline   /data/local/tmp/women.jpg
C:\Users\tower.zhang\Desktop\adb1.0.32\adb\adb.exe pull /data/local/tmp/result.jpg