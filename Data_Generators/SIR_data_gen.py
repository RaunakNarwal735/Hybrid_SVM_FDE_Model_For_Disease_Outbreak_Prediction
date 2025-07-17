import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_sir_data(N=10000, beta=0.3, gamma=0.1, I0=10, days=120, noise=False):
    S = N - I0
    I = I0
    R = 0

    S_list, I_list, R_list = [S], [I], [R]

    for _ in range(1, days):
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I

        S += dS
        I += dI
        R += dR

        if noise:
            I_noisy = I + np.random.normal(0, I*0.05)  # 5% noise
            I_list.append(max(I_noisy, 0))
        else:
            I_list.append(I)
        S_list.append(S)
        R_list.append(R)

    df = pd.DataFrame({
        "Day": np.arange(days),
        "Susceptible": S_list,
        "Infected": I_list,
        "Recovered": R_list
    })
    return df

# Generate data
data = generate_sir_data(noise=True)
print(data.head())

# Plot
plt.figure(figsize=(10,6))
plt.stackplot(data["Day"],  data["Infected"], data["Susceptible"],data["Recovered"], labels=[ 'Infected','Susceptible', 'Recovered'], colors=['#F6BE00','#008B8B', '#353E43'])
plt.xlim(0, max(data["Day"]))
plt.ylim(0, data["Susceptible"].iloc[0] + data["Infected"].iloc[0] + data["Recovered"].iloc[0])
plt.xlabel("Day")
plt.ylabel("Population")
plt.legend()
plt.title("Synthetic SIR Simulation")
plt.show()
