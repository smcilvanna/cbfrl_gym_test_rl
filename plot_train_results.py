import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the CSV file without headers
df = pd.read_csv('results.csv', header=None)

# Step 2: Format columns
# Convert all columns except the first to normal notation rounded to three decimal places
# Keep the first column in scientific notation
df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)

# The first column remains as is (in scientific notation)

# Step 3: Add custom column headings
# Adjust the column headings as needed
df.columns = ['Learn-Rate', 'Discount-Factor', 'Batch-Size', 'Average-Reward', 'Min-Reward']

# Step 4: Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(len(df.columns), len(df) / 2))  # Adjust figure size as needed

# Hide axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Create a table
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Set font size
table.auto_set_font_size(False)
table.set_fontsize(10)

# Adjust column width
table.auto_set_column_width(col=list(range(len(df.columns))))

# Save the figure
plt.savefig('table_image.png', bbox_inches='tight', dpi=300)
plt.close()
