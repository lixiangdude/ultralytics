import json
import os
import time
from concurrent import futures

import cv2
import numpy as np
import oss2
import requests
from oss2.credentials import EnvironmentVariableCredentialsProvider

idx = 0

failed = 0

urls = [
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/057c166dcc8648e69cba69009a8535b4.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/4043a96369e045e394904ad06b75f041.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/369f40a91e3d4e2fa27cffe7cb1ab4ad.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/f41690b41f97466aa8de98d8ec69502f.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/731c981176c64ff5ba8077869307686c.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/794ed6430e114c9794a074205d1e446e.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/cf908dde34ad43aea51d7703612e4f19.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/1aadeb046b1e498cae78c71299c6b707.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/fe58ece1090c46509b1284edc7130690.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/87a27486f187400984c97c24078af01e.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/b016365a362e4cffa4bcc53f30a214a5.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9cf26a02681f44f68b718762cd0f5494.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/b85eccd8649d45f5aed4f58c81dbba83.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/3cd388793abf41cda26d109d865d9bb4.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/194e02aad0fd4a6ba330ffa87835cf4e.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/f25e8a6a4470475db073d1fca2b11204.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/fc03851ba5a743e29677d569c0691171.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/952b7e29c8ae47dd8f658ced3072046f.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/28695c23e1e14ad08e60e6f0506112c2.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/7230fb3471be46d082b7a90a6b35fc05.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9aa5a8001c7f463c94a8a47fdd0af068.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/5499a3969ef241a3ace3215baccc760f.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/41834f75c8374046885eee2484a6835f.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/11dcaaedaa4d40379c93e2b4141ed401.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/6176dca744e64bd6bf8a02dd92f466eb.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/723fbb77ae71495e8407265cb3d8f7db.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/aa00a97db0a34c0e90eea6d393cedde2.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/33fa8f17405f415195055f6a714f4c09.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/9c0ac1beb6334b9492957d716e69f737.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/ab23d6dcb3c1458b9157bc44af1ecfa0.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/a9d17063e03f4324aa577e8edaa04781.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/8bcbebc5a02440d8b62e98e82b8fbb1c.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/85eae54d86ef42558f526146f66e00da.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/e88fd9afd2e24a18b46fc546265ea014.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/0a9a9417f1cf460d805c7a41de2833e9.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/f1f3d8d908674e00bdf11a53ece1ac3a.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/93cac405270643d7825075efd3b25586.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/8cacd73e50ec4d00ae6db4fe3e727e89.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/c8b186e3068c43f6a3ac14e98defc07b.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/a46e3bba1cae4e3883568f4952387a16.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/ee63c527c94d47f884912778e02fca7d.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/cd4372f62d0244b081aca7a703c80d69.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/79765b0f9d3541f9bc796ac2dc7f1d29.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/d5c8a742f7464f96979633f0dc2f10aa.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/551db6bd96204c0fb90658ec74faf532.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/194fa1eb216c48048c7aca3ff3b00a9e.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/f56254d7d9584494883f609ac19014c5.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/701ff63a05214e2dba1a13cee73dbdb0.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/d92593ffbcf74d80aa12afaa43a26584.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/bba2093db70e47daa8dbc82a5f0860ce.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/39f5dbc542154d1ea26e594078c42c94.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/5bd431ec5bc244d690fb8829a04f203f.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/6d5143742343405b9162cdf61d466752.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/cbed3b1b836f4b3ea48067bd17be9b9e.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/7868ef382f84429fb79047fba3978b67.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/2d1e05db877a4b1d9b7482bbc8f1f6cf.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/e1e599f026834cd0ae7f0a714123175d.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/2c8273bc772c433e8e2def48d49290d1.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/61523fb326234be194e5d41afa06b17b.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/1c3367372a47450eaf1b5c92651f1d0e.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/46f57779597247fead442a257d589ed3.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/1d963ddf5e444ca8b59e7a7d99ed230e.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/908d84d6d21245f680cf03dffbcfc87f.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/4563bf6787974c829c8787406ab9e095.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/880661dc90054251818aef3d47e6841b.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/74483c83efb84cf7a41f55c314bffcb8.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/809d5a96fee24fc294fb4a8d5445412a.jpg',
'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/ccdc1c07f3fa45238be0403188e2f782.jpg',
]

MAX_WORKERS = 1

# res = requests.get(image_url)
# img = cv2.imdecode(np.fromstring(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
# 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。
auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
# 填写源Bucket名称，例如srcexamplebucket。
src_bucket_name = "ow-prod"
# 填写与源Bucket处于同一地域的目标Bucket名称，例如destexamplebucket。
# 当在同一个Bucket内拷贝文件时，请确保源Bucket名称和目标Bucket名称相同。
dest_bucket_name = "ow-prod"
# yourEndpoint填写Bucket所在Region对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
bucket = oss2.Bucket(auth, "https://oss-cn-beijing.aliyuncs.com", dest_bucket_name)
url_batches = np.array_split(urls, MAX_WORKERS)


def copy_img(url):
    # 填写不包含Bucket名称在内源Object的完整路径，例如srcexampleobject.txt。
    src_object_name = url[1:]
    # 填写不包含Bucket名称在内目标Object的完整路径，例如destexampleobject.txt。
    dest_object_name = os.path.join("美丽海淀图片", url.split("/")[-1])
    # 将源Bucket中的某个Object拷贝到目标Bucket。
    result = bucket.copy_object(src_bucket_name, src_object_name, dest_object_name)
    # 查看返回结果的状态。如果返回值为200，表示执行成功。
    # img = cv2.imread(image_url)
    print(result)
    if result.status != 200:
        print(result.resp.response.url)


# idx = 0
begin = time.time()
# copy_img('/platform_id_1/app_id_null/roled_user_id_null/type_1/809d5a96fee24fc294fb4a8d5445412a.jpg')
for urls in url_batches:
    batch_begin = time.time()
    # with futures.ThreadPoolExecutor(max_workers=max(MAX_WORKERS, len(urls))) as executor:  # 实例化线程池
    #     res = executor.map(copy_img, urls)

        # 跟内置的map很像，对序列进行相同操作，注意是异步、非阻塞的！
        # 返回的是一个生成器，需要调用next
    idx += len(urls)
    print(f"已复制{idx}张图片，当前批次耗时{time.time() - batch_begin}秒，总耗时{time.time() - begin}秒")
    for image_url in urls:
        res = requests.get(image_url)
        img = cv2.imdecode(np.fromstring(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join("longzeyuan", image_url.split("/")[-1]), img)
