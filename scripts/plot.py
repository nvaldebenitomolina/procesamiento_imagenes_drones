import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('output/area.csv')

print(df)


ax = sns.barplot(y="files", x="area", hue="tipo", data=df)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position

# Put a legend to the right side
ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)


fig = ax.get_figure()
fig.savefig('area.png', dpi=500)

