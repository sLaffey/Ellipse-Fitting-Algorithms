import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 生成一个带有噪声的椭圆数据
def generate_ellipse_data():
    # 椭圆参数
    a = 50  # 圆心 x 坐标
    b = 50  # 圆心 y 坐标
    c = 30  # 长半轴
    d = 15  # 短半轴
    theta = np.radians(30)  # 旋转角度（弧度）

    # 参数方程生成椭圆上的点
    num_points = 10
    t = np.linspace(0, 2 * np.pi, num_points)
    x = a + c * np.cos(t) * np.cos(theta) - d * np.sin(t) * np.sin(theta)
    y = b + c * np.cos(t) * np.sin(theta) + d * np.sin(t) * np.cos(theta)

    # 添加噪声
    noise = np.random.normal(0, 3, x.shape)
    x_noisy = x + noise
    y_noisy = y + noise

    return x_noisy, y_noisy, a, b, c, d, theta

# Hough 变换拟合椭圆（简化版）
def hough_ellipse(x, y):
    # 对参数空间进行采样
    a_range = np.arange(30, 70, 2)
    b_range = np.arange(30, 70, 2)
    c_range = np.arange(20, 40, 1)
    d_range = np.arange(10, 20, 1)
    theta_range = np.arange(0, np.pi, np.radians(10))

    max_votes = 0
    best_a, best_b, best_c, best_d, best_theta = 0, 0, 0, 0, 0

    # 遍历参数空间
    for a in a_range:
        for b in b_range:
            for c in c_range:
                for d in d_range:
                    for theta in theta_range:
                        # 计算椭圆方程在当前参数下的点
                        t = np.linspace(0, 2 * np.pi, 20)
                        x_ellipse = a + c * np.cos(t) * np.cos(theta) - d * np.sin(t) * np.sin(theta)
                        y_ellipse = b + c * np.cos(t) * np.sin(theta) + d * np.sin(t) * np.cos(theta)

                        # 简单计算投票数
                        # 这里只是计算给定点中有多少在椭圆附近
                        distance = np.sqrt((x[:, np.newaxis] - x_ellipse)**2 + (y[:, np.newaxis] - y_ellipse)**2)
                        votes = np.sum(distance < 5, axis=1)
                        current_max_votes = np.max(votes)

                        # 更新最佳参数
                        if current_max_votes > max_votes:
                            max_votes = current_max_votes
                            best_a, best_b, best_c, best_d, best_theta = a, b, c, d, theta

    return best_a, best_b, best_c, best_d, best_theta

# 绘制结果
def plot_results(x, y, a, b, c, d, theta):
    plt.figure(figsize=(10, 8))

    # 绘制原始数据点
    plt.scatter(x, y, color='blue', label='Noisy Data Points')

    # 绘制拟合的椭圆
    t = np.linspace(0, 2 * np.pi, 100)
    x_ellipse = a + c * np.cos(t) * np.cos(theta) - d * np.sin(t) * np.sin(theta)
    y_ellipse = b + c * np.cos(t) * np.sin(theta) + d * np.sin(t) * np.cos(theta)
    plt.plot(x_ellipse, y_ellipse, color='red', label='Fitted Ellipse')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ellipse Fitting using Hough Transform (Simplified)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# 主程序
if __name__ == "__main__":
    # 生成数据
    x_noisy, y_noisy, true_a, true_b, true_c, true_d, true_theta = generate_ellipse_data()

    # 拟合椭圆（简化版 Hough 变换）
    fitted_a, fitted_b, fitted_c, fitted_d, fitted_theta = hough_ellipse(x_noisy, y_noisy)

    # 绘制结果
    plot_results(x_noisy, y_noisy, fitted_a, fitted_b, fitted_c, fitted_d, fitted_theta)