import io
import math
import os.path
import time

import requests
import torch

from ultralytics import YOLO, YOLOWorld
import cv2
import numpy as np
import requests
import torch
from equilib import equi2pers
from PIL import Image, ImageDraw, ImageFont
from concurrent import futures
from multiprocessing import Pool

images = [
    '/home/lixiang/PycharmProjects/ultralytics/imgs/0c2d58d8f45d4a4184d9df65d77a61f5.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/0cec5eb87ce14537b666b09dcb3c8849.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/0e737a627235479c8ff3a1a76100bb9f.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/1cc1beb7ae3f46a29b7a56cc1478d41e.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/1d932411e2414a64beb94f12428edb16.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/1f1c63f7dcbe43f7ba174688d2bfc01f.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/1f76e22065e9433fa63d9b3843e09f17.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/1ff0ca6dd1354fe6ad651a3fb929356f.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/2d8831de52814dc18909d788e05d438a.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/3c59b5c23153471b8a37cd39d23b0033.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/3d9ed423e76e4074b47fcfafd3af5bb3.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/3f08a283daf74389b21743f91afdeb38.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/4ae3658f9f794ca08b462c6fe6cc7390.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/4e0d93f70796430580ce7fe5d7a4e22a.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/4f13bf389b5d49d8b698b630be4b6e25.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/4ff6860c34864b4d95ae08c2f7023e6f.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/5b152c37a9f6473dbf931080b7698c94.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/5b619abf3b29451f9d7850789f3577a2.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/5dd5f033eb134b1ab3c40a082c28bc61.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/06d54efd006e490d8aa5603260ccb4c2.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/6e0aeebd172640b88c72d3fc88f3fab9.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/6ee72c73a2f44f689349f1e041ea233d.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/7c728674787f4301b2f8f811a008112b.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/8cac48fc625543839aa1bae0898d88a4.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/08e00b6412c847ecb18c84b81449ec15.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/8f855773ff664bf99d9a5d990c09cd08.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/9a686709e7584f9d8ca10578590416c7.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/9b7e6f36f1eb4bfb838a30630b9182eb.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/9cd7690101e745ce8ef6d3a8a008474d.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/09ce3f8aaf6347eaab1c9210f2b93ad4.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/9e6b4cce1bf84c40829a4cad796fe967.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/9f2a417c4e1b488a9f340c33f38f7eeb.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/12ca6d7291734bc2924d64f9097e4bef.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/013e92db2aa0445b9b7c606d53c2c923.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/26d62670f5c84f1e92c0402b0e824130.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/51f9d80eaa8746919ab777ba974e8678.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/59c57873ff1c46449017f7e58d0f0e1f.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/71e86432ce6d480bafa2c43493ec4568.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/72fadd4ee38a4ad392332aca03c85faa.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/77f7cea57d564527832efb59017481f4.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/85a20c16d6ab41d18661271a44f88aed.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/395d151408804ae981d5d3a636d12fdd.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/461f3403acc547bfb98f02d8da5bfe1c.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/536d1670615b4e1ba17a3f61890a099f.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/683a80b9f97e4f8583b386671872a405.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/748f05324f1444fd8f6b232300ccec6e.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/0761f55764f248e586bca24dcfca41dd.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/789a03d9970d433cbdfb62e24498454d.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/926c8249e8454d2c83ba81f4b06b30c0.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/2107cc43c470409a9fcaa120e100476e.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/3071d1779de24408a740633e8aa16f64.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/3211f225c2d44afe85f301f55b0c09d9.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/3980c02e40054692ae46c3ad6d55bbea.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/7289a96720c34e3a83adf62867fc6615.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/59117f01c8eb4579afbb04b6e9163925.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/259166c365d4401792112ddd71366559.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/5283352cbdf343bd8e4ce197cbbd084a.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/40664431b6574654bb77b9ebebfbbd55.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/226134332a1943f2b7ddd8f93e94c439.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/18351469262047a58a6e93baecd1105f.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/a0fb7455a45044739216e2cca627f473.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/a6a88f1785fe4042b24338b565c1fd63.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/a0302eaf6dc84a84842970feb9cab50f.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/afb4cb43e14941e7b074123c040f747b.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/ba39f9338c6f43298a03d923833ec62b.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/ba076d42948d4c5d8aab392a4a2dcc49.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/bbb833c6db16461caaadbc628f6f9bb7.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/be1c80cb99414663bdaa7500b74c8456.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/c81ada54b4e349d98786cd75da5a99c9.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/c262ab01af0347d2a61ea4f74e4d0458.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/cb3cdca3ddbe4805af66a4b88bb057d9.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/ce965bad47734fcbb026788bba50e1f7.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/d7fea3a81c0d4783b9f3dd3fb5c1ffee.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/d166aac619214437a4c9ee25fd763fcd.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/d966519fa0a14204b9ddf91092c54795.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/dcb8e696cc0f4218a43ce5bb94aaebd0.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/e3a4dd496cc142e5977fa841c1cdb588.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/e5a1c4398bf44bcab4675c0bb1e93046.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/eab9eed230cc484da5ea07a3d7cf2bef.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/ec33dbe2240b434a92900213d048c3d5.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/ec2869c2311b4129a830489f91440249.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/ed22d5f9a2094cf4bc5af6dc2f88b6c3.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/efa8564101ec4906843e8519c6d5f44f.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/f85c77a28cba414dbd1ed9a5a24307d9.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/f97e1b70d8f34083a4b093862f1d0cd8.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/f274bc26311c4ab28b2444082f2a8d49.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/f592b7cefd8a4f81b2ba83b846826502.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/f3936ca5de0e4887956dd88773502391.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/fdefb629609a409e8dc9125e82218d57.jpg',
    '/home/lixiang/PycharmProjects/ultralytics/imgs/fe146e68a6c84c868243a01bec88e4c8.jpg',
]


def rotate(rots):
    global fov_deg, pers_height, pers_width
    fov_deg = 90.0
    pers_height = 640
    pers_width = 640
    # pers_height = pers_should_height
    # pers_width = pers_should_width
    before_rot = time.time()
    pers_img = equi2pers(
        equi=equi_img,
        rots=rots,
        height=pers_height,
        width=pers_width,
        fov_x=fov_deg,
        mode="bilinear",
    )
    np.ascontiguousarray(pers_img, np.uint8)


for img_idx, image_url in enumerate(images):
    begin = time.time()
    now = time.time()
    # res = requests.get(image_url)
    # img = cv2.imdecode(np.fromstring(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.imread(image_url)
    if img is None:
        continue
    # img = cv2.imread(image_url)
    print(f"下载图片耗时{time.time() - now:.2f}s")
    # img = e2c(img, face_w=int(img.shape[0] / 3))
    now = time.time()
    (height, width, depth) = img.shape
    equi_img = np.transpose(img, (2, 0, 1))
    equi_img = torch.tensor(equi_img)
    pers_should_height = int(height / 4)
    pers_should_width = int(width / 4)
    pers_height = 480
    pers_width = 640
    pitch = math.radians(0)
    fov_deg = 90.0
    rectangles = []
    img_PIL = Image.fromarray(img[..., ::-1])  # 转成 PIL 格式
    draw = ImageDraw.Draw(img_PIL)  # 创建绘制对象
    # 俯仰角从0到30度，每次转动5度
    for j in range(7):
        now = time.time()
        yaw = math.pi
        pers_imgs = []
        # 偏航角每次转
        # 动10度
        rots = []
        for i in range(36):
            rots.append({
                "roll": 0.0,
                "pitch": pitch,  # rotate vertical
                "yaw": yaw,  # rotate horizontal
            })
            yaw -= np.pi / 18
            # cv2.imshow('img', cube_result)
            # cv2.waitKey(0)
        with Pool(20) as p:
            p.map(rotate, rots)
        pitch += math.radians(5)
        print(f"转动视角耗时{time.time() - now}秒")