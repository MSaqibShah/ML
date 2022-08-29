import pandas
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = 'cost_revenue_clean.csv'
def model(dataset):
    # Read Data CSV
    data = pandas.read_csv(dataset)
    # data.describe()

    # Split data into Features and Label
    X = pandas.DataFrame(data, columns=["production_budget_usd"])
    Y = pandas.DataFrame(data,columns=["worldwide_gross_usd"])

    # Plot the dataset - Scatter Plot
    plt.figure(figsize=(10,6))
    plt.scatter(X,Y, alpha=0.3)
    plt.title('Production Budget vs Revenue in $')
    plt.xlabel('Production Budget')
    plt.ylabel('Revenue in $')
    plt.ylim(0,3000000000)
    plt.xlim(0,450000000)
    plt.show()

    # Create a Model (liner Regression) and Fit the dataset 
    reg = LinearRegression() # h(x) = theta0 +(theta1).(x) or y = mx+c 
    reg.fit(X,Y)

    # reg.coef_ # theta1 or m

    # reg.intercept_ # theta0 or c

    # Plot the Model along with dataset
    plt.figure(figsize=(10,6))
    plt.scatter(X,Y, alpha=0.3)
    plt.title('Production Budget vs Revenue in $')
    plt.xlabel('Production Budget')
    plt.ylabel('Revenue in $')
    plt.plot(X,reg.predict(X), color="red", linewidth=4)
    plt.ylim(0,3000000000)
    plt.xlim(0,450000000)
    plt.show()

    #calculate R-squared coefficient of the model.
    reg.score(X,Y)

    return reg


budget = [[10000000]]
r = model(dataset)  
print(r.predict(budget))

