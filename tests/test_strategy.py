from stable_baselines3 import PPO
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np

model = PPO.load("./models/model_3_rewards")
num_points = 100

res = []
observations_x = []
observations_y = []
for x in range(num_points):
    for y in range(num_points):
        observations_x.append(x)
        observations_y.append(y)

        r = model.predict([x, y], deterministic=True)[0].tolist()
        res.append(r)


scatter = go.Scatter3d(
    x=observations_x,
    y=observations_y,
    z=res,
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        opacity=0.8
    )
)


# Creazione del layout
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='Bad points', range=[0, 100]),
        yaxis=dict(title='Pareto points', range=[0, 100]),
        zaxis=dict(title='Evaluation', range=[0, 1])
    )
)

# Creazione della figura
fig = go.Figure(data=[scatter], layout=layout)

# Salvataggio della figura in formato HTML
pio.write_html(fig, file='scatter3d.html', auto_open=True)

print(np.sum(res))