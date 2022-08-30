#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt


# In[6]:


data = pd.read_csv('data/lsd_math_score_data.csv')


# In[46]:


time = data[['Time_Delay_in_Minutes']]
lsd = data[['LSD_ppm']]
score = data[['Avg_Math_Test_Score']]
print(data)


# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.title('Tissue Conc. of LSD over Time', fontsize=17)
plt.xlabel('Time delay in min', fontsize=14)
plt.ylabel('LSD conc. in PPM', fontsize=14)

plt.xlim(0,500)
plt.ylim(1,7)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.text(x=0,y=-0.5, s='Wagner et el. 1968', fontsize=12)
plt.style.use('ggplot')

plt.plot(time['Time_Delay_in_Minutes'],lsd['LSD_ppm'], color='#123123', linewidth=3)
plt.show()


# In[97]:


reg = LinearRegression()
reg.fit(lsd,score)
print('Theta1: ', reg.coef_[0][0])
print('Theta2: ', reg.intercept_[0])
print('R-Squared: ', reg.score(lsd, score))
pred_score = reg.predict(lsd)


# In[103]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.title('Arthimatic VS LSD-25', fontsize=17)
plt.xlabel('LSD conc. in PPM', fontsize=14)
plt.ylabel('Avg Score', fontsize=14)

plt.xlim(1,7)
plt.ylim(25,85)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# plt.text(x=0,y=-0.5, s='Wagner et el. 1968', fontsize=12)
plt.style.use('fivethirtyeight')

plt.scatter(lsd,score, color='#0000ff', s=100, alpha=0.7)
plt.plot(lsd["LSD_ppm"],pred_score, color='#ff0000', linewidth=3)
plt.show()

