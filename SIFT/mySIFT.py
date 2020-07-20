"""
190785 李克
190914 方懿德
190773 徐恒
190784 杜静
190786 孙世若
"""
import numpy as np
import functools
import cv2


S = 3  # 在每组octave中检测S个尺度的极值点
SIGMA = 1.6  # 高斯滤波σ参数
IMAGE_BORDER_WIDTH = 5  # 极值点检测时预设的图像边界
THRESHOLD = 0.04  # 极值点检测阈值
MAX_ATTEMPT_TIMES = 5  # 极值点迭代拟合的最大次数
EIGENVALUE_RATIO = 10  # 本征值比
NUM_BINS = 36  # 特征点角度值的划分区间数
FLOAT_TOLERANCE = 1e-7  # 判别接近0的容差
WIN_SIZE = 4  # 生成Descriptor时的窗口大小
BINS = 8  # 生成Descriptor时的角度划分区间数


def extract_feature(image):

    gaussian_pyramid = generate_gaussian_pyramid(image)   # 高斯金字塔生成
    dog_pyramid = generate_dog_pyramid(gaussian_pyramid)  # DoG金字塔生成

    keypoints = generate_keypoints(gaussian_pyramid, dog_pyramid)  # 尺度空间极值点定位和特征点方向计算

    keypoints = process_keypoints(keypoints)  # 特征点排序与冗余去除

    descriptors = generate_descriptors(keypoints, gaussian_pyramid)  # 描述子生成

    return keypoints, descriptors


def generate_gaussian_pyramid(image):
    """
    生成高斯金字塔
    :param image: 基础图像
    :return: 形如[num_octaves * (S + 3)]的二维列表，元素为高斯图像
    """
    image = image.astype('float32')
    # 根据图像短边长计算octave的数量
    num_octaves = int(np.log(min(image.shape)) / np.log(2) - 0.5)

    gaussian_images = []
    for i in range(num_octaves):
        gaussian_images_in_octave = []
        # 对octave中的S+3张图像进行不同尺度的高斯滤波
        for j in range(S + 3):
            if i > 0 and j == 0:
                gaussian_images_in_octave.append(image)
                continue
            gaussian_kernel = np.sqrt((SIGMA ** 2) - 1) if j == 0 else np.sqrt((SIGMA ** 2) * (2 ** (2 * j / S) - 2 ** (2 * ((j - 1) / S))))
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        # 将前一组倒数第三张图像缩作为高斯金字塔上一组的输入
        octave_base = gaussian_images_in_octave[-3]
        image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
    return np.array(gaussian_images)


def generate_dog_pyramid(gaussian_images):
    """
    生成高斯差分DoG金字塔
    :param gaussian_images: 高斯金字塔
    :return: 形如[num_octaves * (S + 2)]的二维列表，元素为DoG图像
    """
    dog_images = []
    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for i in range(len(gaussian_images_in_octave) - 1):
            # 相邻两张高斯图像相减
            dog_images_in_octave.append(cv2.subtract(gaussian_images_in_octave[i], gaussian_images_in_octave[i + 1]))
        dog_images.append(dog_images_in_octave)
    return np.array(dog_images)


def generate_keypoints(gaussian_images, dog_images):
    """
    查找图像金字塔中所有比例空间局部极值的像素位，置局部极值所在位置取为key location
    :param gaussian_images: 高斯金字塔， 形如[num_octaves * (S + 3)]的二维列表，元素为高斯图像
    :param dog_images: 高斯差分金字塔，形如[num_octaves * (S + 2)]的二维列表，元素为DoG图像
    :return: 特征点列表
    """
    keypoints = []

    # 下面的for循环是为了处理每一个octave
    for octave_index, dog_images_in_octave in enumerate(dog_images):
        img_h, img_w = dog_images_in_octave[0].shape[:]
        # 在每一个octave中循环取图像，每次取三张相邻图像
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # 形成一个3x3的窗口，(i, j)为窗口中心像素
            # 图像边界宽度IMAGE_BORDER_WIDTH，为全局变量，设为5
            for i in range(IMAGE_BORDER_WIDTH, img_h - IMAGE_BORDER_WIDTH):
                for j in range(IMAGE_BORDER_WIDTH, img_w - IMAGE_BORDER_WIDTH):
                    # 遍历除去边界外的所有点，判断是否是极值点。3x3的窗口在3张图像上形成一个3x3x3的正方体
                    # 如果窗口中心像素绝对值大于阈值且像素值大于或小于其邻域所有26个像素，则该像素为极值点，返回True；否则返回False。
                    if judge_extremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2]):
                        # 调用judge_extremum函数，判断是否是符标准的极值点。
                        # 如果是极值点，则通过围绕该点邻域的二次拟合来迭代细化比例空间极值的像素位置
                        localization_result = localize_extremum(i, j, image_index + 1, octave_index, dog_images_in_octave)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = compute_keypoints_orientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints


def judge_extremum(first_image_window, second_image_window, third_image_window):
    """
    判断是否满足极值点条件
    :param first_image_window: 3x3x3窗口的第一层
    :param second_image_window: 3x3x3窗口的第二层（中间层）
    :param third_image_window: 3x3x3窗口的第三层
    :return:是极值返回1，否则返回0
    """
    center_pixel_value = second_image_window[1, 1]
    # 取空间27个点的中心点，即第二张图片的中心点
    if abs(center_pixel_value) > np.floor(THRESHOLD / S * 255):
        if center_pixel_value > 0:
            # 如果中心点像素值大于0，判断其是否是极大值点
            # 下面return的功能是，判断中心点是否大于等于周围26个点，若是返回1，否则返回0
            return np.all(center_pixel_value >= first_image_window) and \
                   np.all(center_pixel_value >= third_image_window) and \
                   np.all(center_pixel_value >= second_image_window[0, :]) and \
                   np.all(center_pixel_value >= second_image_window[2, :]) and \
                   center_pixel_value >= second_image_window[1, 0] and \
                   center_pixel_value >= second_image_window[1, 2]
        elif center_pixel_value < 0:
            # 如果中心点像素值小于0，判断中心点是否是极小值点
            # 下面return的功能是，判断中心点是否小于等于周围26个点，若是返回1，否则返回0
            return np.all(center_pixel_value <= first_image_window) and \
                   np.all(center_pixel_value <= third_image_window) and \
                   np.all(center_pixel_value <= second_image_window[0, :]) and \
                   np.all(center_pixel_value <= second_image_window[2, :]) and \
                   center_pixel_value <= second_image_window[1, 0] and \
                   center_pixel_value <= second_image_window[1, 2]
    return False


def localize_extremum(i, j, image_index, octave_index, dog_images_in_octave):
    """
    通过二次拟合定位极值
    :param i: 当前像素点的横坐标
    :param j: 当前像素点的纵坐标
    :param image_index: 当前图像在octave中的编号
    :param octave_index: 当前图像处于第几组octave
    :param dog_images_in_octave: 当前octave中的所有图像
    :return: 成功拟合至收敛则创建并返回keypoint及其image_index，否则返回None
    """
    img_h, img_w = dog_images_in_octave[0].shape[:]

    # 尝试进行拟合直到收敛
    pixel_cube = gradient = hessian = extremum_update = None
    # 最大拟合次数定为5
    for attempt_cnt in range(MAX_ATTEMPT_TIMES):

        # 选取中心像素所在的3x3x3的邻域矩阵并将像素值重新缩放为[0, 1]，pixel_cube.shape=(3, 3, 3)
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        # 3x3x3的像素块 pixel_cube
        pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                               second_image[i-1:i+2, j-1:j+2],
                               third_image[i-1:i+2, j-1:j+2]]) / 255.

        # 计算邻域矩阵的梯度矩阵，gradient.shape = (3,)
        gradient = compute_gradient(pixel_cube)
        # 计算邻域矩阵的海森矩阵，hessian.shape=(3, 3)
        hessian = compute_hessian(pixel_cube)
        # 使用梯度矩阵和海森矩阵迭代进行基于最小二乘法的迭代，最小二乘解extremum_update.shape = (3,)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        # 最小二乘解均小于0.5则认定为收敛，终止迭代
        if np.all(abs(extremum_update) < 0.5):
            break
        # 否则进行迭代
        j += int(np.round(extremum_update[0]))
        i += int(np.round(extremum_update[1]))
        image_index += int(np.round(extremum_update[2]))

        # 确保新的pixel_cube完全位于图像内，如果出界则直接返回None
        if i < IMAGE_BORDER_WIDTH or i >= img_h - IMAGE_BORDER_WIDTH or j < IMAGE_BORDER_WIDTH or j >= img_w - IMAGE_BORDER_WIDTH or image_index < 1 or image_index > S:
            return None
        # 如果达到最大尝试次数仍未收敛则直接返回None
        if attempt_cnt == MAX_ATTEMPT_TIMES - 1:
            return None

    # 计算迭代过程中像素点值的大小。如果超过预先设点的THRESHOLD，localizeExtremumViaQuadraticFit就return none
    value_updated_extremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    # 如果迭代过程中的像素值大于阈值，大于阈值才有效
    if abs(value_updated_extremum) * S >= THRESHOLD:
        xy_hessian = hessian[:2, :2]
        # 取海森矩阵的[0,0][0,1][1,0][1,1]元素
        xy_hessian_trace = np.trace(xy_hessian)
        # 计算xy_hessian的迹
        xy_hessian_det = np.linalg.det(xy_hessian)
        # 计算方阵的行列式的值
        if xy_hessian_det > 0 and EIGENVALUE_RATIO * (xy_hessian_trace ** 2) < ((EIGENVALUE_RATIO + 1) ** 2) * xy_hessian_det:
            # 实例化KeyPoint对象
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(np.round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = SIGMA * (2 ** ((image_index + extremum_update[2]) / np.float32(S))) * (2 ** (octave_index + 1))             # octave_index + 1 because the input image was doubled
            keypoint.response = abs(value_updated_extremum)
            return keypoint, image_index
    return None


def compute_gradient(pixel_array):
    """
    使用中心差分公式计算当前3x3x3矩阵的中心像素点梯度，x, y, s分别对应第二一三维度
    f'(x) = (f(x + 1) - f(x - 1)) / 2
    """
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return np.array([dx, dy, ds])


def compute_hessian(pixel_array):
    """
    使用中心导数公式计算当前3x3x3矩阵的Hessian矩阵，Hessian矩阵是多元函数的二阶偏导数，三元分别对应x,y,s
    f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    """
    dxx = pixel_array[1, 1, 2] - 2 * pixel_array[1, 1, 1] + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * pixel_array[1, 1, 1] + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * pixel_array[1, 1, 1] + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])


def compute_keypoints_orientations(keypoint, octave_index, gaussian_image):
    """
    创建关键点邻域的每一个像素的梯度直方图，使用平方邻域计算关键点的方向
    KeyPoint的attribute
    point2f pt;   位置坐标
    float size;   特征点邻域直径
    float angle;   特征点的方向，值为[零, 三百六十)，负值表示不使用
    int octave;   特征点所在的图像金字塔的组
    int class_id;   用于聚类的id
    :param keypoint: 输入的特征点对象
    :param octave_index: 当前特征点处于第几组octave
    :param gaussian_image: 当前特征点所在的高斯图像
    :return: 带有方向的特征点对象
    """

    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = 1.5 * keypoint.size / np.float32(2 ** (octave_index + 1))  # scale是高斯权重的标准差

    radius = int(np.round(3 * scale))  # radius_factor默认为3，3倍的标准差涵盖99.7% 比例

    weight_factor = -0.5 / (scale ** 2)

    raw_histogram = np.zeros(NUM_BINS)
    smooth_histogram = np.zeros(NUM_BINS)

    # 对关键点邻域内像素遍历
    for i in range(-radius, radius + 1):

        region_y = int(np.round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i

        if 0 < region_y < image_shape[0] - 1:

            for j in range(-radius, radius + 1):

                region_x = int(np.round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j

                if 0 < region_x < image_shape[1] - 1:

                    # 计算梯度
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]

                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)  # 计算区域像素的梯度的幅度
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))  # 计算区域像素的梯度的角度，不是弧度

                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))  # 归一化
                    # 计算直方图
                    histogram_index = int(np.round(gradient_orientation * NUM_BINS / 360.))

                    raw_histogram[histogram_index % NUM_BINS] += weight * gradient_magnitude

    # 滑动窗口对梯度直方图进行一个平滑，均匀化角度之间的值
    for n in range(NUM_BINS):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % NUM_BINS]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % NUM_BINS]) / 16.

    orientation_max = max(smooth_histogram)  # 最大的梯度
    # 找到峰值方向下标，即大于左右值，where返回的是一个tuple所以要用[0]
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]

    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]

        if peak_value >= 0.8 * orientation_max:
            # 平方峰值内插
            left_value = smooth_histogram[(peak_index - 1) % NUM_BINS]
            right_value = smooth_histogram[(peak_index + 1) % NUM_BINS]
            # peak_index是峰值的index，p = 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)，
            # peak_index + p 是内插峰值位置
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % NUM_BINS
            # 内插的峰值方向
            orientation = 360. - interpolated_peak_index * 360. / NUM_BINS

            # 方向极度接近360设置为0
            if abs(orientation - 360.) < FLOAT_TOLERANCE:
                orientation = 0

            # 生成带有角度的关键点
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)

            keypoints_with_orientations.append(new_keypoint)

    return keypoints_with_orientations


def compare_keypoints(keypoint1, keypoint2, verbose=None):
    """
    比较两个keypoint
    :param keypoint1: 第一个特征点
    :param keypoint2: 第二个特征点
    :param verbose: 是否输出日志信息
    :return: keypoint1 > keypoint2则为True否则为False
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        print("pt[0]:".ljust(10), keypoint1.pt[0] - keypoint2.pt[0]) if verbose else None
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        print("pt[1]:".ljust(10), keypoint1.pt[1] - keypoint2.pt[1]) if verbose else None
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        print("size:".ljust(10), keypoint2.size - keypoint1.size) if verbose else None
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        print("angle:".ljust(10), keypoint1.angle - keypoint2.angle) if verbose else None
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        print("response:".ljust(10), keypoint2.response - keypoint1.response) if verbose else None
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        print("octave:".ljust(10), keypoint2.octave - keypoint1.octave) if verbose else None
        return keypoint2.octave - keypoint1.octave
    print("class_id:".ljust(10), keypoint2.class_id - keypoint1.class_id) if verbose else None
    return keypoint2.class_id - keypoint1.class_id


def process_keypoints(keypoints):
    """
    排序并删除冗余特征点
    :param keypoints: 原始的特征点列表
    :return: 经过处理的特征点列表
    """
    # 如果传入的关键点长度小于2则直接返回
    if len(keypoints) < 2:
        return keypoints
    # 对关键点从小到大排序
    keypoints.sort(key=functools.cmp_to_key(compare_keypoints))
    # 新的不重复的关键点列表
    new_keypoints = []
    for i, point in enumerate(keypoints):
        # point表示原关键点列表目前到的关键点（待比较），last表示新关键点列表中最后一个关键点
        # 新列表中的0～n-1个关键点是不重复的
        if i != 0:
            last = new_keypoints[-1]
            # 剔除 pt / size / angle 均相等的关键点
            if (last.pt == point.pt) and (last.size == point.size) and (last.angle == point.angle):
                continue
        point.octave = (point.octave & ~255) | ((point.octave - 1) & 255)
        new_keypoints.append(point)
    return new_keypoints


def generate_descriptors(keypoints, guassion_pyramid):
    """
    生成描述子
    :param keypoints: 特征点列表
    :param guassion_pyramid: 高斯金字塔
    :return: 描述子列表
    """
    descriptors = []
    for keypoint in keypoints:
        # 计算keypoint的octave, layer, scale
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        # 获取对应尺度的高斯图像
        image = guassion_pyramid[octave + 1, layer]
        num_rows, num_cols = image.shape
        # 计算点在当前octave上的坐标
        point = np.round(scale * np.array(keypoint.pt) * 0.5).astype('int')
        idx = BINS / 360.
        # 获取旋转角度
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))

        row_bins = []
        col_bins = []
        mag_list = []
        orient_bins = []
        # 直方图
        hist_bins = np.zeros((WIN_SIZE + 2, WIN_SIZE + 2, BINS))

        # 根据关键点邻域直径大小确定关键点邻域宽度
        hist_width = 3 * 0.5 * scale * keypoint.size * 0.5
        # 计算实际图像的区域半径
        radius = int(np.round(hist_width * np.sqrt(2) * (WIN_SIZE + 1) * 0.5))
        radius = int(min(radius, np.sqrt(num_rows ** 2 + num_cols ** 2)))
        for row in range(-radius, radius + 1):
            for col in range(-radius, radius + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                # 计算旋转后的采样点落在子区域的下标（因为选取的是正方形，所以部分点不符合条件）
                row_bin = (row_rot / hist_width) + 0.5 * WIN_SIZE - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * WIN_SIZE - 0.5
                if -1 < row_bin < WIN_SIZE and -1 < col_bin < WIN_SIZE:
                    # 当前点在图像上的坐标
                    window_row = int(np.round(point[1] + row))
                    window_col = int(np.round(point[0] + col))
                    if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                        # 计算梯度
                        dx = image[window_row, window_col + 1] - image[window_row, window_col - 1]
                        dy = image[window_row - 1, window_col] - image[window_row + 1, window_col]
                        # 计算梯度模值
                        mag_grad = np.sqrt(dx * dx + dy * dy)
                        # 计算方向
                        grad_orient = np.rad2deg(np.arctan2(dy, dx)) % 360
                        # 对梯度高斯加权
                        weight = np.exp((-1 / (2 * (0.5 * WIN_SIZE) ** 2)) * (
                                    (row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bins.append(row_bin)  # 保存子区域下标
                        col_bins.append(col_bin)
                        # 保存加权模值
                        mag_list.append(weight * mag_grad)
                        # 所在方向的bin
                        orient_bins.append((grad_orient - angle) * idx)

        for row_bin, col_bin, magnitude, orient_bin in zip(row_bins, col_bins, mag_list, orient_bins):
            # 转化成整型，表示对应的bin，确定需要累加梯度方向的bin
            row_bin_near, col_bin_near, orient_bin_near = np.floor([row_bin, col_bin, orient_bin]).astype(int)
            dr, dc, do = row_bin - row_bin_near, col_bin - col_bin_near, orient_bin - orient_bin_near
            if orient_bin_near < 0:
                orient_bin_near += BINS
            if orient_bin_near >= BINS:
                orient_bin_near -= BINS
            # 根据所得采样点在子区域中的下标计算其对每个种子点的贡献
            # dr = dr, dc = dc, do = do
            # 累加梯度
            hist_bins[row_bin_near + 1, col_bin_near + 1, orient_bin_near] += \
                magnitude * (1 - dr) * (1 - dc) * (1 - do)  # 对当前方向bin的累加值
            hist_bins[row_bin_near + 1, col_bin_near + 1, (orient_bin_near + 1) % BINS] += \
                magnitude * (1 - dr) * (1 - dc) * do
            hist_bins[row_bin_near + 1, col_bin_near + 2, orient_bin_near] += \
                magnitude * (1 - dr) * dc * (1 - do)  # 对当前方向bin 列+1 的累加值
            hist_bins[row_bin_near + 1, col_bin_near + 2, (orient_bin_near + 1) % BINS] += \
                magnitude * (1 - dr) * dc * do
            hist_bins[row_bin_near + 2, col_bin_near + 1, orient_bin_near] += \
                magnitude * dr * (1 - dc) * (1 - do)  # 对当前方向bin 行+1 的累加值
            hist_bins[row_bin_near + 2, col_bin_near + 1, (orient_bin_near + 1) % BINS] += \
                magnitude * dr * (1 - dc) * do
            hist_bins[row_bin_near + 2, col_bin_near + 2, orient_bin_near] += \
                magnitude * dr * dc * (1 - do)
            hist_bins[row_bin_near + 2, col_bin_near + 2, (orient_bin_near + 1) % BINS] += \
                magnitude * dr * dc * do  # 对当前方向bin+1 行+1 列+1的累加值

        # 去除掉直方图当中的第0、5行列，实际特征维度是4×4×8
        descriptor_vector = hist_bins[1:-1, 1:-1, :].flatten()
        # 归一化处理
        threshold = np.linalg.norm(descriptor_vector) * 0.2
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), FLOAT_TOLERANCE)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')
