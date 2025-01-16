import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Constants and parameters
BATTERY_CAPACITY = 75  # kWh, total capacity of the battery
CHARGE_EFFICIENCY = 0.95  # Charging efficiency
DISCHARGE_EFFICIENCY = 0.90  # Discharging efficiency
AMBIENT_TEMPERATURE = 25  # Celsius
TEMPERATURE_COEFFICIENT = 0.01  # Temperature effect on efficiency
CYCLE_LIFESPAN = 2000  # Total charge/discharge cycles before 80% capacity

# Simulation parameters
simulation_time = 24  # hours
time_step = 0.1  # hours
charge_rate = 20  # kW
discharge_rate = 15  # kW

# Initial conditions
state_of_charge = 0.5  # 50% SoC
battery_temperature = AMBIENT_TEMPERATURE
capacity = BATTERY_CAPACITY
cycles = 0

def update_soc(soc, rate, efficiency, capacity, time_step, charging=True):
    if charging:
        delta_soc = (rate * efficiency * time_step) / capacity
    else:
        delta_soc = -(rate * efficiency * time_step) / capacity
    return max(0, min(1, soc + delta_soc))

def update_temperature(current_temp, ambient_temp, load, coeff):
    temp_rise = coeff * load
    return current_temp + temp_rise

def update_capacity(initial_capacity, cycles, lifespan):
    return initial_capacity * (1 - cycles / lifespan)

# Simulation data storage
time_points = []
soc_history = []
temperature_history = []
capacity_history = []
cycles_history = []

# Simulation loop
current_time = 0
while current_time < simulation_time:
    time_points.append(current_time)
    soc_history.append(state_of_charge)
    temperature_history.append(battery_temperature)
    capacity_history.append(capacity)
    cycles_history.append(cycles)

    # Simulate charge/discharge based on time (simple schedule)
    if 8 <= current_time % 24 < 16:  # Charging hours
        state_of_charge = update_soc(state_of_charge, charge_rate, CHARGE_EFFICIENCY, capacity, time_step, charging=True)
    else:  # Discharging hours
        state_of_charge = update_soc(state_of_charge, discharge_rate, DISCHARGE_EFFICIENCY, capacity, time_step, charging=False)

    # Update temperature
    load = charge_rate if 8 <= current_time % 24 < 16 else discharge_rate
    battery_temperature = update_temperature(battery_temperature, AMBIENT_TEMPERATURE, load, TEMPERATURE_COEFFICIENT)

    # Update capacity based on cycles
    if state_of_charge == 1 or state_of_charge == 0:
        cycles += 0.5  # Half cycle for full charge/discharge
        capacity = update_capacity(BATTERY_CAPACITY, cycles, CYCLE_LIFESPAN)

    # Advance time
    current_time += time_step

# Prepare data for ML
features = np.column_stack((cycles_history, soc_history, temperature_history))
target = np.array(capacity_history)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse:.2f} kWh")

# Visualization
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(time_points, soc_history, label="State of Charge (SoC)")
plt.ylabel("SoC (%)")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_points, temperature_history, label="Battery Temperature")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_points, capacity_history, label="Battery Capacity")
plt.plot(time_points, model.predict(features), label="Predicted Capacity", linestyle="--")
plt.ylabel("Capacity (kWh)")
plt.xlabel("Time (hours)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
