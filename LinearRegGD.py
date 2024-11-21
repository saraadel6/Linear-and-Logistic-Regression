import csv

def readCSV_cnt_data():
    file_path = 'temp_data.csv'
    # logic to extract exact y and x
    cnt = 0
    data =[]
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if cnt != 0:
                data.append(row)
            cnt+=1 
    return cnt - 1, data

#____________________ GENERAL EQUATION _____________________
def calc_y_dash(w,b,x): #  m is no. of rows
    return (w*x)+b

def summation(y_dash, y_actual, sq = 1):
    if sq == 1:
        return (y_dash-y_actual)**2
    else:
        return (y_dash-y_actual)

    
def Equation(w,b,constant, der = 0,multiply = 0):
    # row[0] = x
    # row[1] = y
    getData = readCSV_cnt_data()
    m = getData[0]
    data = getData[1]
    EQ = (1/m) * constant
    sum=0
    for row in data:
        y_dash = calc_y_dash(w,b,int(row[0]))
        if der == 0:
            sum += summation(y_dash, int(row[1]))
        else:
            if multiply == 1:
                sum += summation(y_dash, int(row[1]),0)*int(row[0])
            else:
                sum += summation(y_dash, int(row[1]),0)

    EQ *= sum
    return EQ

#__________{{ MSE COST FUNCTION }}______________
def costFunction(w,b):
    return Equation(w,b,(1/2))

#______________{{ GRADIANT DESCENT }}________

def GD_W(w,b):
    return Equation(w,b,1,1,1)

def GD_B(w,b):
    return Equation(w,b,1,1)

w =0
w_old = 0
b = 0
alpha = 0.01
oldMSE= float('inf')
while costFunction(w,b) < oldMSE:
    oldMSE =costFunction(w,b)
    w = w_old - (alpha * GD_W(w,b))
    b = b - (alpha * GD_B(w_old,b))
    w_old = w
    
print(w,b)
# print(calc_y_dash(w,b,1)) # y dash will be approx equal to 2X

