import numpy as np
import pandas as pd

p = 0.5
q = 0.5
pz11 = 0.5
pz10 = 0.6
pz01 = 0.9
pz00 = 0.9

marginal = np.array([p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)])
cond = np.array([pz11, pz10, pz01, pz00])
joint = marginal * cond
x = np.array([1, 1, 0, 0])
y = np.array([1, 0, 1, 0])

data = pd.DataFrame(np.array([x, y, marginal, cond, joint]).T,\
        columns=["x", "y", "marginal", \
        "cond", "joint"])

# sum over x, use gy
# sum over y, use gx
print(data)
gx = data[["x", "marginal", "cond", "joint"]].groupby("x").sum().reset_index()
gy = data[["y", "marginal", "cond", "joint"]].groupby("y").sum().reset_index()
print(gx)
print(gy)
alpha = gx[gx.x == 1]["marginal"].to_numpy()[0]
print(f"alpha = {alpha}")


beta = gx[gx.x == 1]["joint"] / data["joint"].sum()
beta = beta.to_numpy()[0]
print(f"beta = {beta}")

gamma = data.marginal[0] * data.cond[0] / gy[gy.y == 1].joint
gamma = gamma.to_numpy()[0]
print(f"gamma = {gamma}")

