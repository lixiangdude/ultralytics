import os
import random
import shutil

imgs = os.listdir('det_merged')

imgs = random.sample(imgs, 20)

ori_imgs = os.listdir('imgs')

compare_dir = 'compare1'

if not os.path.exists(compare_dir):
    os.makedirs(compare_dir)

for img in imgs:
    shutil.copyfile(os.path.join('det_merged', img), os.path.join(compare_dir, img))
    if img.replace('_det_result', '') in ori_imgs:
        shutil.copyfile(os.path.join('imgs', img.replace('_det_result', '')), os.path.join(compare_dir, img.replace('_det_result', '')))
