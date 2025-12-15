import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import constants, fft
from scipy.interpolate import griddata, LinearNDInterpolator
import scipy.io

data = scipy.io.loadmat('CST_RP_new_probe_WR62_20_001_freq_12.mat')
print(data.keys())

Probe_phi = data['phiCST_fin_new']
Probe_theta = data['thetaCST_fin_new']
Probe_RP_phi_Xpol = data['RP_phi_CST_Xpol_norm']
Probe_RP_phi_Ypol = data['RP_phi_CST_Ypol_norm']
Probe_RP_theta_Xpol = data['RP_theta_CST_Xpol_norm']
Probe_RP_theta_Ypol = data['RP_theta_CST_Ypol_norm']

# Константы и параметры
freq = 12000
freq_hz = freq*1e6  #Частота в герцах
c = constants.c
k = 2 * np.pi * freq_hz / c
wavelength = c / freq_hz

# Параметры сканирования
n_x = 59
dx = 6e-3  # метры
n_y = 81
dy = 6e-3  # метры

# Загрузка координат с восстановлением структуры измерений
def load_coords(filename):
    df = pd.read_csv(filename, delimiter=';', header=None, skiprows=4)
    coords = df.iloc[:, :2].astype(float).values
    return coords[:, 0], coords[:, 1]

x_coords, y_coords = load_coords('rupor6_Probe_X.txt')

x_min, x_max = x_coords.min(), x_coords.max()
y_min, y_max = y_coords.min(), y_coords.max()

# Создаем регулярную сетку
x_regular = np.linspace(x_min, x_max, n_x)
y_regular = np.linspace(y_min, y_max, n_y)

# Загрузка S-параметров
def load_s21(filename, freq_index):
    df = pd.read_csv(filename, sep=r'\s+', skiprows=5, comment='#',
                     header=None, engine='python')
    freq_data = df[df.iloc[:, 0] == freq_index]
    real_part = freq_data.iloc[:, 5].astype(float).values
    imag_part = freq_data.iloc[:, 6].astype(float).values
    return real_part + 1j * imag_part


S21_x = load_s21('rupor6_Probe_X.s2p', freq)
S21_y = load_s21('rupor6_Probe_Y.s2p', freq)

# print(S21_x)
# print(S21_y)

# Ресэмплинг данных на регулярную сетку
X, Y = np.meshgrid(x_regular, y_regular)
E_x = np.zeros((n_y, n_x), dtype=complex)
E_y = np.zeros((n_y, n_x), dtype=complex)

# print(X)
# print(Y)
# print(E_x)
# print(E_y)


points = np.column_stack((x_coords, y_coords))
values_x = S21_x
E_x = griddata(points, values_x, (X, Y), method='cubic', fill_value=0+0j)

# Интерполяция для вертикальной поляризации
values_y = S21_y
E_y = griddata(points, values_y, (X, Y), method='cubic', fill_value=0+0j)

# print(E_x)
# print(E_y)

def calculate_angular_spectrum(E_near, dx, dy, k, z_scan=0.06):
    """
    Вычисляет угловой спектр (F_x или F_y) из измерений ближнего поля.
    Без компенсации зонда.

    Параметры:
    - E_near: ближнее поле, форма (Ny, Nx)
    - dx, dy: шаги сканирования [м]
    - k: волновое число [рад/м]
    - z_scan: расстояние от антенны до плоскости сканирования [м]
             (положительно, если плоскость сканирования перед антенной)

    Возвращает:
    - F: угловой спектр на сетке (kx, ky)
    - theta: углы места [град]
    - phi: азимутальные углы [град]
    - mask: маска propagating waves
    """
    Ny, Nx = E_near.shape

    #2D БПФ
    E_spectral = fft.fft2(E_near) * dx * dy

    #Волновые числа (рад/м)
    # kx = 2π * циклические частоты
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)  # рад/м
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)  # рад/м
    KX, KY = np.meshgrid(kx, ky, indexing='xy')  # форма (Ny, Nx)

    #Радиальная компонента и маска распространяющихся волн
    kz_sq = k ** 2 - KX ** 2 - KY ** 2 + 0j
    kz = np.sqrt(np.where(kz_sq >= 0, kz_sq, 0))  # только действительные kz
    mask = (KX ** 2 + KY ** 2) <= k ** 2  # распространяющиеся волны


    E_spectral_corrected = E_spectral * np.exp(-1j * kz * z_scan)


    #Отображение (kx, ky) -> (θ, φ)
    # Стандартное сферическое преобразование:
    # kx = k * sinθ * cosφ
    # ky = k * sinθ * sinφ
    # kz = k * cosθ

    k_rho = np.sqrt(KX ** 2 + KY ** 2 + 1e-12)  # избегаем деления на 0

    # Угол места: θ ∈ [-90°, 90°]
    # sinθ = k_rho / k
    sin_theta = k_rho / k
    sin_theta_clipped = np.clip(sin_theta, -1.0, 1.0)

    # Вычисление угла с сохранением знака
    theta_sign = np.sign(KY)
    theta_rad = theta_sign * np.arcsin(sin_theta_clipped)
    theta = theta_rad * 180 / np.pi
    phi = np.arctan2(KY, KX) * 180 / np.pi

    #Сдвиг нулевой частоты в центр
    F = np.fft.fftshift(E_spectral_corrected)
    theta = np.fft.fftshift(theta)
    phi = np.fft.fftshift(phi)
    mask = np.fft.fftshift(mask)

    return F, theta, phi, mask


F_x, theta, phi, mask = calculate_angular_spectrum(E_x, dx, dy, k)
F_y, _, _, _ = calculate_angular_spectrum(E_y, dx, dy, k)

plt.figure()
plt.imshow(np.abs(F_x), extent=[phi.min(), phi.max(), theta.min(), theta.max()],
                    cmap='jet', aspect='auto')
plt.colorbar()
plt.title('F_x magnitude')
plt.show()
# np.savetxt('theta.txt', theta,
#            fmt='%.6f', delimiter=' ', header='Theta matrix')
# np.savetxt('phi.txt', phi,
#            fmt='%.6f', delimiter=' ', header='phi matrix')
# np.savetxt('P_theta.txt', Probe_theta,
#            fmt='%.6f', delimiter=' ', header='Theta matrix')
# np.savetxt('P_phi.txt', Probe_phi,
#            fmt='%.6f', delimiter=' ', header='phi matrix')
# print(F_x.shape, F_y.shape)
# print(theta.min(), theta.max(), theta.shape)
# print(phi.min(), phi.max(), phi.shape)
# print(f"Probe_phi shape: {Probe_phi.shape}")
# print(f"Probe_theta shape: {Probe_theta.shape}")
# print(f"Probe_RP_theta_Xpol shape: {Probe_RP_theta_Xpol.shape}")

# Проверка диапазонов углов
# print(f"Probe phi range: [{Probe_phi.min():.1f}, {Probe_phi.max():.1f}]")
# print(f"Probe theta range: [{Probe_theta.min():.1f}, {Probe_theta.max():.1f}]")
# print(f"FFT phi range: [{phi.min():.1f}, {phi.max():.1f}]")
# print(f"FFT theta range: [{theta.min():.1f}, {theta.max():.1f}]")

plt.figure()
plt.imshow(np.abs(Probe_RP_theta_Xpol),  extent=[Probe_phi.min(), Probe_phi.max(), Probe_theta.min(), Probe_theta.max()],
                    cmap='jet', aspect='auto')
plt.colorbar()
plt.title('Probe RP_theta X polarization')
plt.figure()
plt.imshow(np.abs(Probe_RP_theta_Ypol),  extent=[Probe_phi.min(), Probe_phi.max(), Probe_theta.min(), Probe_theta.max()],
                    cmap='jet', aspect='auto')
plt.colorbar()
plt.title('Probe RP_theta Y polarization')
plt.figure()
plt.imshow(np.abs(Probe_RP_phi_Xpol),  extent=[Probe_phi.min(), Probe_phi.max(), Probe_theta.min(), Probe_theta.max()],
                    cmap='jet', aspect='auto')
plt.colorbar()
plt.title('Probe RP_phi X polarization')
plt.figure()
plt.imshow(np.abs(Probe_RP_phi_Ypol),  extent=[Probe_phi.min(), Probe_phi.max(), Probe_theta.min(), Probe_theta.max()],
                    cmap='jet', aspect='auto')
plt.colorbar()
plt.title('Probe RP_phi Y polarization')
plt.show()
# def expand_probe_data_symmetric():
#     # Исходные данные: Probe_theta, Probe_phi, Probe_RP_... в диапазоне φ: [-90, 90]
#
#     # Создаем расширенные массивы
#     theta_full = np.concatenate([Probe_theta, Probe_theta])
#     phi_full = np.concatenate([Probe_phi, -Probe_phi])  # отражение
#
#     # Значения для отраженной полусферы (используем комплексное сопряжение для сохранения фазы)
#     # Для многих зондов E(θ, -φ) = E(θ, φ)*
#     RP_theta_Xpol_full = np.concatenate([Probe_RP_theta_Xpol, np.conj(Probe_RP_theta_Xpol)])
#     RP_phi_Xpol_full = np.concatenate([Probe_RP_phi_Xpol, np.conj(Probe_RP_phi_Xpol)])
#     RP_theta_Ypol_full = np.concatenate([Probe_RP_theta_Ypol, np.conj(Probe_RP_theta_Ypol)])
#     RP_phi_Ypol_full = np.concatenate([Probe_RP_phi_Ypol, np.conj(Probe_RP_phi_Ypol)])
#
#     return theta_full, phi_full, RP_theta_Xpol_full, RP_phi_Xpol_full, RP_theta_Ypol_full, RP_phi_Ypol_full
#
#
# # Расширяем данные
# (theta_full, phi_full,
#  RP_thx_full, RP_phx_full,
#  RP_thy_full, RP_phy_full) = expand_probe_data_symmetric()


def interpolate_probe_pattern(phi_fft, theta_fft):

    # Подготавливаем точки зонда
    points_probe = np.column_stack([Probe_theta.ravel(), Probe_phi.ravel()])

    # Создаем интерполяторы для каждой компоненты
    interp_E_thx = LinearNDInterpolator(points_probe, Probe_RP_theta_Xpol.ravel(), fill_value=0 + 0j)
    interp_E_phx = LinearNDInterpolator(points_probe, Probe_RP_phi_Xpol.ravel(), fill_value=0 + 0j)
    interp_E_thy = LinearNDInterpolator(points_probe, Probe_RP_theta_Ypol.ravel(), fill_value=0 + 0j)
    interp_E_phy = LinearNDInterpolator(points_probe, Probe_RP_phi_Ypol.ravel(), fill_value=0 + 0j)

    # Подготавливаем точки для интерполяции
    points_fft = np.column_stack([theta_fft.ravel(), phi_fft.ravel()])

    # Интерполяция
    E_thx = interp_E_thx(points_fft).reshape(phi_fft.shape)
    E_phx = interp_E_phx(points_fft).reshape(phi_fft.shape)
    E_thy = interp_E_thy(points_fft).reshape(phi_fft.shape)
    E_phy = interp_E_phy(points_fft).reshape(phi_fft.shape)

    return E_thx, E_phx, E_thy, E_phy

E_thx, E_phx, E_thy, E_phy = interpolate_probe_pattern(phi, theta)

print(E_thx.shape, E_phx.shape)
print(E_thy.shape, E_phy.shape)
plt.figure()
plt.imshow(np.abs(E_thx), aspect='auto')
plt.colorbar()
plt.title('Interpolated E_thx magnitude')
plt.show()
# Переводим углы в радианы для вычислений
theta_rad = np.radians(theta)
phi_rad = np.radians(phi)
cos_theta = np.cos(theta_rad)
lambda_sq = wavelength**2

# Вычисление знаменателя
denominator = (E_thx * E_phy - E_phx * E_thy)

# Защита от деления на ноль
epsilon = 1e-12
denominator_safe = np.where(np.abs(denominator) > epsilon, denominator, epsilon + 0j)

# Формулы компенсации
F_theta = (cos_theta / lambda_sq) * (F_x * E_phy - F_y * E_phx) / denominator_safe
F_phi = (cos_theta / lambda_sq) * (F_x * E_thy - F_y* E_thx) / denominator_safe

F_theta[~mask] = 0 + 0j
F_phi[~mask] = 0 + 0j

#Расчет ко- и кроссполяризации по Ludwig-3 (φ' = 90°, ζ = φ)
F_co = F_theta * np.sin(phi_rad) + F_phi * np.cos(phi_rad)
F_cross = F_theta * np.cos(phi_rad) - F_phi * np.sin(phi_rad)

#Диаграмма направленности
P_far = np.abs(F_x) ** 2 + np.abs(F_y) ** 2
P_dB = 10 * np.log10(P_far / np.max(P_far))

P_far_x = np.abs(F_cross)
P_dB_x = 10 * np.log10(P_far_x / np.max(P_far_x))
P_far_y = np.abs(F_co)
P_dB_y = 10 * np.log10(P_far_y / np.max(P_far_y))

print(P_dB_x.max(), P_dB_x.min())
print(theta.min(), theta.max())
print(phi.min(), phi.max())

np.savetxt('P_DB_x.txt', E_thx,
           fmt='%.6f', delimiter=' ', header='phi matrix')
np.savetxt('P_DB_y.txt', E_phx,
           fmt='%.6f', delimiter=' ', header='phi matrix')
print(f"Форма S21_x: {S21_x.shape}")
print(f"Ожидаемая форма: ({n_y}, {n_x}) = {n_y * n_x} элементов")

E_x_mag = np.abs(E_x)
print(np.max(E_x_mag))
# S21_x_norm = E_x / np.max(E_x)
S21_x_norm = E_x_mag.reshape(n_y, n_x)
print(S21_x_norm.min(), S21_x_norm.max())
print(f"Форма S21_x: {S21_x_norm.shape}")

E_y_mag = np.abs(E_y)
print(np.max(E_y_mag))
# S21_y_norm = E_y / np.max(E_y)
S21_y_norm = E_y_mag.reshape(n_y, n_x)
print(S21_y_norm.min(), S21_y_norm.max())
print(f"Форма S21_y: {S21_y_norm.shape}")

E_x_phase = np.degrees(np.angle(E_x))
print(np.max(E_x_phase))
# S21_x_norm = E_x / np.max(E_x)
S21_x_phase_norm = E_x_phase.reshape(n_y, n_x)
print(S21_x_phase_norm.min(), S21_x_phase_norm.max())
print(f"Форма S21_x: {S21_x_phase_norm.shape}")

E_y_phase = np.degrees(np.angle(E_y))
print(np.max(E_y_phase))
# S21_y_norm = E_y / np.max(E_y)
S21_y_phase_norm = E_y_phase.reshape(n_y, n_x)
print(S21_y_phase_norm.min(), S21_y_phase_norm.max())
print(f"Форма S21_y: {S21_y_phase_norm.shape}")

# Построение
# fig1, axes_1 = plt.subplots(1, 2, figsize=(14, 5))
fig2, axes_2 = plt.subplots(1, 2, figsize=(14, 5))
fig3, axes_3 = plt.subplots(1, 2, figsize=(14, 5))
#2D диаграмма(общая)
# im_1 = axes_1[0].imshow(P_dB,
#                     extent=[phi.min(), phi.max(), theta.min(), theta.max()],
#                     cmap='jet',
#                     aspect='auto',
#                     vmin=-40, vmax=0,
#                     origin='lower')
# axes_1[0].set_xlabel('Азимут φ, град')
# axes_1[0].set_ylabel('Угол места θ, град')
# axes_1[0].set_title('Диаграмма направленности (2D)')
# fig1.colorbar(im_1, ax=axes_1[0], label='Усиление, дБ')
#
# # Срезы
# theta_cut = theta[:, theta.shape[1] // 2]
# P_cut_dB = P_dB[:, theta.shape[1] // 2]
# axes_1[1].plot(theta_cut, P_cut_dB, 'b-', linewidth=2)
# axes_1[1].set_xlabel('Угол θ, град')
# axes_1[1].set_ylabel('Усиление, дБ')
# axes_1[1].set_ylim(-50, 0)
# axes_1[1].grid(True, alpha=0.3)
# axes_1[1].set_title('Срез ДН в плоскости φ=0°')
# fig1.suptitle('Суммарная диаграмма направленности', fontsize=14)
print("\n=== Отладка размерностей ===")
print(f"theta shape: {theta.shape}")
print(f"phi shape: {phi.shape}")
print(f"P_dB_x shape: {P_dB_x.shape}")
print(f"theta min/max: {theta.min():.1f}, {theta.max():.1f}")
print(f"phi min/max: {phi.min():.1f}, {phi.max():.1f}")
print(F_x.shape, F_y.shape)
print(theta.min(), theta.max())
print(phi.min(), phi.max())
print(P_dB_x.shape)
print(P_dB_y.shape)
# 2D диаграмма (Кросс)
im_2 = axes_2[0].imshow(P_dB_x,
                    extent=[phi.min(), phi.max(), theta.min(), theta.max()],
                    cmap='jet',
                    aspect='auto',
                    vmin=-40, vmax=0,
                    origin='lower')
axes_2[0].set_xlabel('Азимут φ, град')
axes_2[0].set_ylabel('Угол места θ, град')
axes_2[0].set_title('Диаграмма направленности (2D)')
fig2.colorbar(im_2, ax=axes_2[0], label='Усиление, дБ')

# Срезы
theta_cut = theta[:, theta.shape[1] // 2]
P_cut_dB_x = P_dB_x[:, theta.shape[1] // 2]
axes_2[1].plot(theta_cut, P_cut_dB_x, 'b-', linewidth=2)
axes_2[1].set_xlabel('Угол θ, град')
axes_2[1].set_ylabel('Усиление, дБ')
axes_2[1].set_ylim(-25, 0)
axes_2[1].grid(True, alpha=0.3)
axes_2[1].set_title('Срез ДН в плоскости φ=0°')
fig2.suptitle('Кросс диаграмма направленности', fontsize=14)

# 2D диаграмма (основная поляризация)
im_3 = axes_3[0].imshow(P_dB_y,
                    extent=[phi.min(), phi.max(), theta.min(), theta.max()],
                    cmap='jet',
                    aspect='auto',
                    vmin=-40, vmax=0,
                    origin='lower')
axes_3[0].set_xlabel('Азимут φ, град')
axes_3[0].set_ylabel('Угол места θ, град')
axes_3[0].set_title('Диаграмма направленности (2D)')
fig3.colorbar(im_3, ax=axes_3[0], label='Усиление, дБ')

# Срезы
theta_cut = theta[:, theta.shape[1] // 2]
P_cut_dB_y = P_dB_y[:, theta.shape[1] // 2]
axes_3[1].plot(theta_cut, P_cut_dB_y, 'b-', linewidth=2)
axes_3[1].set_xlabel('Угол θ, град')
axes_3[1].set_ylabel('Усиление, дБ')
axes_3[1].set_ylim(-30, 0)
axes_3[1].grid(True, alpha=0.3)
axes_3[1].set_title('Срез ДН в плоскости φ=0°')
fig3.suptitle('Основная диаграмма направленности', fontsize=14)

# Компановка и отображение
# fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()


plt.figure(figsize=(10, 8))
im = plt.imshow(S21_x_norm,
                extent=[x_regular.min(), x_regular.max(), y_regular.min(), y_regular.max()],
                cmap='jet',
                aspect='auto',
                vmin=0, vmax=0.0035,
                origin='lower')

plt.colorbar(im, label='Измеренные амплитуды S21')
plt.xlabel('Ось Х, мм')
plt.ylabel('Ось Y, мм')
plt.title('Измеренные S21 (Кросс поляризация)')
plt.grid(True, alpha=0.3)

plt.figure(figsize=(10, 8))
im = plt.imshow(S21_y_norm,
                extent=[x_regular.min(), x_regular.max(), y_regular.min(), y_regular.max()],
                cmap='jet',
                aspect='auto',
                vmin=0, vmax=0.008,
                origin='lower')

plt.colorbar(im, label='змеренные амплитуды S21')
plt.xlabel('Ось Х, мм')
plt.ylabel('Ось Y, мм')
plt.title('Измеренные S21 (Основная поляризация)')
plt.grid(True, alpha=0.3)

plt.figure(figsize=(10, 8))
im = plt.imshow(S21_x_phase_norm,
                extent=[x_regular.min(), x_regular.max(), y_regular.min(), y_regular.max()],
                cmap='jet',
                aspect='auto',
                origin='lower')

plt.colorbar(im, label='Измеренная фаза S21')
plt.xlabel('Ось Х, мм')
plt.ylabel('Ось Y, мм')
plt.title('Измеренные S21 (Кросс поляризация)')
plt.grid(True, alpha=0.3)

plt.figure(figsize=(10, 8))
im = plt.imshow(S21_y_phase_norm,
                extent=[x_regular.min(), x_regular.max(), y_regular.min(), y_regular.max()],
                cmap='jet',
                aspect='auto',
                origin='lower')

plt.colorbar(im, label='измеренная фаза S21')
plt.xlabel('Ось Х, мм')
plt.ylabel('Ось Y, мм')
plt.title('Измеренные S21 (Основная поляризация)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
