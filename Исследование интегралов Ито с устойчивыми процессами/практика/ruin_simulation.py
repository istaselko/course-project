import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import levy_stable
import warnings

warnings.filterwarnings("ignore")

"""ЭТАП 1: Загрузка реальных данных"""

ticker = "BTC-USD"
data = yf.download(ticker, start="2019-01-01", end="2024-01-01", progress=False)

# Считаю ежедневную логарифмическую доходность
returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

# Беру отрицательные доходности (падения рынка) и делаю их положительными "страховыми убытками"
losses = -returns[returns < 0].values.flatten()

# Очищаю данные от возможных бесконечностей, которые появляются из-за нулевых цен на бирже
losses = np.array(losses, dtype=np.float64)
losses = losses[np.isfinite(losses)]

print(f"Извлечено {len(losses)} дней с убытками.")


"""ЭТАП 2: Оценка параметров альфа-устойчивого распределения"""

# Подгоняю данные. В страховании волнуют катастрофические убытки (правый хвост),
# поэтому фиксирую асимметрию beta = 1 (максимальный перекос вправо) и сдвиг loc = 0.
alpha_est, beta_est, loc_est, scale_est = levy_stable.fit(losses, fbeta=1.0, floc=0.0)

print(f"Найденный индекс стабильности (Alpha): {alpha_est:.4f}")
print(f"Найденный параметр масштаба (Sigma): {scale_est:.4f}")

if alpha_est >= 2:
    print("Распределение близко к нормальному.")
else:
    print("Альфа < 2. Дисперсия бесконечна, присутствуют тяжелые хвосты.")


"""ЭТАП 3: Симуляция капитала страховой компании (Метод Монте-Карло)"""

u = 2.0  # Начальный капитал
c = 0.25  # Ежедневная страховая премия (доход компании)
X_port = 1.0  # Объем портфеля (множитель X(t))
T = 1.0  # Горизонт моделирования (1 год)
N = 252  # Количество шагов (рабочих дней)
dt = T / N  # Временной шаг (dt)
M = 2000  # Количество симулируемых траекторий

np.random.seed(42)

# Масштабирующий множитель времени dt^(1/alpha)
time_scale = dt ** (1.0 / alpha_est)

U_trajectories = np.zeros((M, N + 1))
U_trajectories[:, 0] = u

ruined_count = 0

print("Генерация траекторий СДУ по схеме Эйлера-Маруямы")
# Генерирую матрицу устойчивых скачков(алторитм Чемберса-Маллоуса-Стюка)
xi_matrix = levy_stable.rvs(alpha=alpha_est, beta=1.0, loc=0, scale=1, size=(M, N))

for j in range(M):
    for k in range(N):
        # Приращение стохастического интеграла
        dL = time_scale * xi_matrix[j, k] * scale_est
        # Динамика капитала
        U_trajectories[j, k + 1] = U_trajectories[j, k] + c * dt - X_port * dL

    if np.any(U_trajectories[j, :] < 0):
        ruined_count += 1


"""ЭТАП 4: Построение графиков"""

ruin_prob = ruined_count / M
print(
    f"\nВероятность разорения составила: {ruin_prob:.2%} ({ruined_count} компаний из {M})"
)

time_grid = np.linspace(0, T, N + 1)
plt.figure(figsize=(12, 7))

plotted_ruined = False
plotted_survived = False

for j in range(100):  # Рисую только 100, чтобы не перегрузить график
    if np.any(U_trajectories[j, :] < 0):
        label = "Разорение" if not plotted_ruined else ""
        plt.plot(
            time_grid,
            U_trajectories[j, :],
            color="red",
            alpha=0.6,
            linewidth=1,
            label=label,
        )
        plotted_ruined = True
    else:
        label = "Выживание" if not plotted_survived else ""
        plt.plot(
            time_grid,
            U_trajectories[j, :],
            color="blue",
            alpha=0.3,
            linewidth=1,
            label=label,
        )
        plotted_survived = True

plt.axhline(
    0, color="black", linestyle="--", linewidth=2, label="Уровень разорения (0)"
)
plt.title(
    f"Моделирование капитала компании (Интеграл по $\\alpha$-устойчивому процессу, $\\alpha={alpha_est:.2f}$)",
    fontsize=14,
)
plt.xlabel("Время $t$ (доли года)", fontsize=12)
plt.ylabel("Резервный капитал $U(t)$", fontsize=12)
plt.legend(loc="upper left", fontsize=10)
plt.grid(True, alpha=0.3)

info_text = f"Параметры модели:\n$\\alpha$ = {alpha_est:.3f}\nНачальный капитал $u$ = {u}\nВероятность разорения $\\Psi$ = {ruin_prob:.1%}"
plt.text(
    0.75,
    0.85,
    info_text,
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    fontsize=12,
)

plt.tight_layout()
plt.show()
