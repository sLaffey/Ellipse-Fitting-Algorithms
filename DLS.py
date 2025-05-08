import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt

def fit_ellipse(points):
    # 构造设计矩阵 D
    x = points[:, 0]
    y = points[:, 1]
    D = np.column_stack([x**2, x*y, y**2, x, y, np.ones_like(x)])

    # 构造约束矩阵 C
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1

    # 计算散射矩阵 S
    S = D.T @ D

    # 计算广义特征值问题：S * a = λ * C * a
    eigvals, eigvecs = sl.eig(S, C)

    # 找到最小的正特征值对应的特征向量
    positive_eigvals_indices = np.where(np.real(eigvals) > 0)[0]
    if len(positive_eigvals_indices) < 1:
        raise ValueError("没有找到正特征值")
    min_eigval_idx = positive_eigvals_indices[np.argmin(np.real(eigvals[positive_eigvals_indices]))]
    a = np.real(eigvecs[:, min_eigval_idx])

    # 归一化参数，使其满足 4ac - b² = 1
    a, b, c, d, e, f = a / a[0]
    norm_factor = 4 * a * c - b**2
    a, b, c, d, e, f = a / norm_factor, b / norm_factor, c / norm_factor, d / norm_factor, e / norm_factor, f / norm_factor

    return np.array([a, b, c, d, e, f])

def plot_ellipse(ellipse_params, points):
    """
    绘制拟合的椭圆

    参数：
    ellipse_params (numpy array): 拟合椭圆的参数，形状为 (6,)，对应 ax² + bxy + cy² + dx + ey + f = 0
    points (numpy array): 二维点数组，形状为 (n, 2)，其中 n 是点的数量
    """
    a, b, c, d, e, f = ellipse_params

    # 计算椭圆的中心 (x0, y0)
    x0 = (2*c*d - b*e) / (b**2 - 4*a*c)
    y0 = (2*a*e - b*d) / (b**2 - 4*a*c)

    # 计算椭圆的长轴和短轴的长度
    numerator = 2*(a*e**2 + c*d**2 + f*b**2 - 2*b*d*e - 4*a*c*f)
    denominator1 = b**2 - 4*a*c
    denominator2 = np.sqrt((a - c)**2 + b**2)
    minor_axis = np.sqrt(np.abs(numerator / (denominator1 * (a + c + denominator2))))
    major_axis = np.sqrt(np.abs(numerator / (denominator1 * (a + c - denominator2))))

    # 计算椭圆的旋转角度
    theta = 0.5 * np.arctan(b / (a - c))

    # 绘制原始数据点
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Data Points', s=10)

    # 绘制拟合的椭圆
    theta_vals = np.linspace(0, 2*np.pi, 100)
    x_vals = x0 + major_axis * np.cos(theta_vals) * np.cos(theta) - minor_axis * np.sin(theta_vals) * np.sin(theta)
    y_vals = y0 + major_axis * np.cos(theta_vals) * np.sin(theta) + minor_axis * np.sin(theta_vals) * np.cos(theta)
    plt.plot(x_vals, y_vals, color='red', label='Fitted Ellipse')

    # 添加图例和标题
    plt.legend()
    plt.title('Ellipse Fitting')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

# 示例用法
if __name__ == "__main__":
    # 创建一个椭圆示例数据，添加一些噪声
    center_x, center_y = np.random.randn(2)
    major_axis, minor_axis = 12, 10
    theta = np.linspace(0, np.pi, 50)
    x = major_axis * np.cos(theta) + np.random.choice([-1, 1]) * 0.1 * np.random.randn(50) + center_x
    y = minor_axis * np.sin(theta) + np.random.choice([-1, 1]) * 0.1 * np.random.randn(50) + center_y
    points = np.column_stack([x, y])

    # 拟合椭圆
    ellipse_params = fit_ellipse(points)
    print("拟合椭圆的参数：", ellipse_params)

    # 绘制拟合的椭圆
    plot_ellipse(ellipse_params, points)