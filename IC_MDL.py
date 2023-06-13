import pandas as pd
import numpy as np
import openpyxl

## single calculation
# Define the coefficients
c0 = -0.0076225
c1 = 0.36366
c2 = -0.041003
c3 = 0.0022876

# Prompt the user to input the value of y
y = float(input("Enter the value of y: "))

# Define the variable x
x = 0

# Calculate the value of x
while True:
    fx = c0 + c1*x + c2*x**2 + c3*x**3
    if abs(fx - y) < 0.00001:
        break
    dfx = c1 + 2*c2*x + 3*c3*x**2
    x = x - (fx - y) / dfx

# Print the value of x
print("The value of x is:", x)



# Load the Excel file into a pandas dataframe
filepath = '/Users/renyuxuan/Desktop/Research/IC_MDL/2023_3_8_MDL-301_MDL-302_MDL-303_C.xlsx'
df = pd.read_excel(filepath, sheet_name='Summary', header=None)

# Extract the y values from the dataframe and convert to floats
y_values = []
for i in [68,
          78, 79, 80, 81, 82, 83, 84, 85,
          90, 91, 92, 93, 94, 95, 96, 97,
          102, 103, 104, 105, 106, 107, 108, 109]:
    y_str = df.iloc[i, 4]
    if y_str != 'n.a.':
        y_values.append(float(y_str))

# Load the coefficients into a pandas dataframe
df_coef = pd.read_excel(filepath, sheet_name='Calibration', header=None)

# Extract the coefficients from the dataframe
c0 = df_coef.iloc[12, 4]
c1 = df_coef.iloc[12, 5]
c2 = df_coef.iloc[12, 6]
c3 = df_coef.iloc[12, 8]

# Calculate the x values using Newton-Raphson method
x_values = []
for y in y_values:
    x = 1.0 # initial guess
    while True:
        fx = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3
        if abs(fx - y) < 0.00001:
            break
        dfx = c1 + 2 * c2 * x + 3 * c3 * x ** 2
        x = x - (fx - y) / dfx
    x_values.append(x)
    print(f'For y = {y}, x = {x}')

# Write the x values back to the Excel sheet using openpyxl
workbook = openpyxl.load_workbook(filepath)
worksheet = workbook['Summary']
# Write the x values or "n.a." back to the Excel sheet using openpyxl
x_cells = ['E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28',
           'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39', 'E40',
           'E45', 'E46', 'E47', 'E48', 'E49', 'E50', 'E51', 'E55']
for i, cell in enumerate(x_cells):
    if i < len(x_values):
        if not np.isnan(x_values[i]):
                worksheet[cell].value = x_values[i]
        else:
                worksheet[cell].value = "n.a."
    else:
            worksheet[cell].value = "n.a."

# Save the workbook
workbook.save(filepath)

