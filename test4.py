import numpy as np
import pandas as pd

# Example dataset for simulated and measured concentrations (replace with actual data)
data = {
    "City": ["Abu Dhabi", "Melbourne", "Dhaka", "Bujumbura", "Halifax", "Sherbrooke", "Beijing", "Addis Ababa", "Bandung",
             "Haifa", "Rehovot", "Kanpur", "Seoul", "Ulsan", "Mexico City", "Ilorin", "Fajardo", "Kaohsiung", "Taipei",
             "Pasadena", "Johannesburg", "Pretoria"],
    "Csim": [2.603483532, 0.431163175, 4.747680126, 3.673715311, 0.224381786, 0.362798662, 1.385444595, 4.799646778,
             4.02492436, 0.845562015, 1.169340151, 3.833072212, 1.175011504, 0.7798648, 2.008797088, 2.326521987, 0.10290891,
             1.33695288, 0.830166517, 0.474454487, 2.381180572, 2.013390368], # affected by Covid
    "Cmeas": [2.673810294, 0.431163175, 5.56315254, 3.673715311, 0.23148047, 0.363877719, 1.398329746, 4.799646778, 3.663149692,
             0.845562015, 1.159011749, 3.833072212, 1.196440665, 0.7798648, 2.073496912, 2.982349549, 0.10684992, 1.33695288,
             0.830166517, 0.474454487, 2.381180572, 2.098747274] # full dataset
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the Normalized Mean Bias (NMB)
nmb = np.sum(df['Csim'] - df['Cmeas']) / np.sum(df['Cmeas'])

# Calculate the Normalized Mean Difference (NMD)
nmd = np.sum(np.abs(df['Csim'] - df['Cmeas'])) / np.sum(df['Cmeas'])

# Print the results
print(f"Normalized Mean Bias (NMB): {nmb}")
print(f"Normalized Mean Difference (NMD): {nmd}")
