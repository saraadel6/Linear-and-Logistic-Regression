import ml_assignment1 as d
from sklearn.metrics import r2_score #for R2
import matplotlib.pyplot as plot # for graph of MSE

def readCSV_data(test = 0):
    if test == 1:
        feature1 = d.X_test['Engine Size(L)'] #x1
        feature2 = d.X_test['Fuel Consumption City (L/100 km)'] #x2
        y = d.y_test['CO2 Emissions(g/km)'] #y
    else:
        feature1 = d.X_train['Engine Size(L)'] #x1
        feature2 = d.X_train['Fuel Consumption City (L/100 km)'] #x2
        y = d.y_train['CO2 Emissions(g/km)'] #y
    
    data = [list(pair) + [target] for pair, target in zip(zip(feature1, feature2), y)]
    return data

trainData= readCSV_data()
testData = readCSV_data(1)

#____________________ GENERAL EQUATION _____________________
def calc_y_dash(w1,w2,b,x1,x2): #  m is no. of rows
    return (w1*x1)+(w2*x2)+b

def summation(y_dash, y_actual, sq = 1):
    if sq == 1:
        return (y_dash-y_actual)**2
    else:
        return (y_dash-y_actual)

    
def Equation(w1,w2,b,constant, der = 0,var = 0):
    data = trainData
    m = len(data)

    EQ = (1/m) * constant
    sum=0
    for row in data:
        y_dash = calc_y_dash(w1,w2,b,float(row[0]),float(row[1]))
        if der == 0:
            sum += summation(y_dash, float(row[2])) #MSE
        else:
            if var == 1:
                sum += summation(y_dash, float(row[2]),0) *float(row[0]) #dy/dx1
            elif var == 2:
                sum += summation(y_dash, float(row[2]),0) *float(row[1]) #dy/dx2
            else:
                sum += summation(y_dash, float(row[2]),0) #dy/db

    EQ *= sum
    return EQ

#__________{{ MSE COST FUNCTION }}______________
def costFunction(w1,w2,b):
    return Equation(w1,w2,b,(1/2))

#______________{{ GRADIENT DESCENT }}________
def GD_W1(w1,w2,b):
    return Equation(w1,w2,b,1,1,1)

def GD_W2(w1,w2,b):
    return Equation(w1,w2,b,1,1,2)

def GD_B(w1,w2,b):
    return Equation(w1,w2,b,1,1)

cnt =0
w1 =0.0
w1_old = 0.0
w2 =0.0
w2_old = 0.0
b = 0.0
alpha = 0.01
old_train_MSE = float('inf')
error_values =[]
iterations_cnt=[]

while costFunction(w1,w2,b) < old_train_MSE:
    old_train_MSE =costFunction(w1,w2,b)
    error_values.append(old_train_MSE)
    w1 = w1_old - (alpha * GD_W1(w1,w2,b))
    w2 = w2_old - (alpha * GD_W2(w1_old,w2,b))
    b = b - (alpha * GD_B(w1_old,w2_old,b))
    w1_old = w1
    w2_old = w2
    cnt+=1
    iterations_cnt.append(cnt)


print("W1=",w1,"/ W2=",w2,"/ b=",b)
print("MSE = ",old_train_MSE)

# print(testData)
predict = []
te = []
for i in testData:
    predict.append(calc_y_dash(w1,w2,b,i[0],i[1]))
    te.append(i[2])

print("R2 = ",r2_score(te, predict))



plot.figure(figsize=(8, 5))
plot.plot(iterations_cnt, error_values, marker='o', linestyle='-', color='blue', label='MSE')

# Label the axes and the plot
plot.xlabel("Iterations")
plot.ylabel("Mean Squared Error (MSE)")
plot.title("Error Improvement over Iterations")
plot.legend()
plot.grid(True)
plot.show()
