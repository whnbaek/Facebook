# run face recognition
#  input: images in target, textbook directories
# output: target_crop, textbook_crop directories
python3 face_recog.py

# run head pose estimation
#  input: images in target_crop, textbook_crop directories
# output: target_angle.txt, textbook_angle.txt, each has (num, yaw, pitch, roll)
cd FSA-Net/demo/
KERAS_BACKEND=tensorflow python3 demo_FSANET_ssd.py

# run warp
#  input: images in target_crop, target_angle.txt, textbook_angle.txt
# output: warped directory
cd ../../
python3 warp.py

# merging warped and croped
#  input: images in textbook_croped, warped
# output: merged directory
cd face-parsing.PyTorch
python3 test.py
