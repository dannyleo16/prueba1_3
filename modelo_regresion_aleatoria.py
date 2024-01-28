import time
import pandas as pd
import numpy as np
start_time = time.time()
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Carga de datos
df = pd.read_csv("sensor-data.csv")
df['time'] = pd.to_datetime(df['time'])
df['day_of_week'] = df['time'].dt.dayofweek
df['hour_of_day'] = df['time'].dt.hour
numeric_columns = ['power', 'temp', 'humidity', 'light', 'CO2', 'day_of_week', 'hour_of_day']
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# División de datos
X_train, X_test, y_train, y_test = train_test_split(df[numeric_columns], df['dust'], test_size=0.2, random_state=42)

# Modelo con PyTorch
class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(7, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

# Creando el modelo PyTorch
model = PyTorchModel()

# Usando el optimizador Adam y la función de pérdida MSE
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

# Entrenamiento del modelo
for epoch in range(100):
    # Convierte los datos de entrenamiento a tensores de PyTorch
    X_train_tensor = torch.from_numpy(X_train.to_numpy()).float().view(-1, 7)
    y_train_tensor = torch.from_numpy(y_train.to_numpy()).float().view(-1, 1)

    # Propagación hacia adelante
    y_pred = model(X_train_tensor)

    # Cálculo de la pérdida
    loss = loss_fn(y_pred, y_train_tensor)

    # Limpieza de gradientes
    optimizer.zero_grad()

    # Retropropagación del error
    loss.backward()

    # Actualización de pesos
    optimizer.step()

# Evaluación del modelo
y_pred = model(torch.from_numpy(X_test.to_numpy()).float().view(-1, 7))
mse = loss_fn(y_pred, torch.from_numpy(y_test.to_numpy()).float().view(-1, 1)).item()
print("Error cuadrático medio:", mse)

# Evaluación del modelo RANDONM FOREST
model = RandomForestRegressor()
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Puntuaciones de validación cruzada:", scores)
print("Puntuación media de validación cruzada:", scores.mean())

end_time = time.time()
execution_time = end_time - start_time
print(f"El tiempo de ejecución fue: {execution_time} segundos")
