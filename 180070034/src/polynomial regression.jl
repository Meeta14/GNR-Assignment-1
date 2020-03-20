# Import necessary modules; CSV is for reading .csv files and Plots is for plotting
using CSV
using Plots
using DataFrames
# Read the dataset from file
dataset = CSV.read("data//housingPriceData.csv")

price = dataset.price
bedroom = dataset.bedrooms
bathroom = dataset.bathrooms
sqrt_liv = dataset.sqft_living

function normalize(X)
    mean2 = sum(X)/length(X)
    var   = sum((X.-mean2)'*(X.-mean2))/length(X)
    var   = sqrt(var)
    nor     = (X.-mean2)/var
    return nor
end

function mean(X)
    mean = sum(X)/length(X)
end

m = length(price)
test = trunc(Int, m*.8)

para1 = dataset.price
para2 = dataset.bedrooms
para3 = dataset.sqft_living
para4 = para2.*para2
para5 = para3.*para3
para6 = para2.*para3

#para1 = normalize(para1)
para2_train = normalize(para2[1:test])
para3_train = normalize(para3[1:test])
para4_train = normalize(para4[1:test])
para5_train = normalize(para5[1:test])
para6_train = normalize(para6[1:test])

para2_test = normalize(para2[test:end])
para3_test = normalize(para3[test:end])
para4_test = normalize(para4[test:end])
para5_test = normalize(para5[test:end])
para6_test = normalize(para6[test:end])



x0_train = ones(test)
X_train = cat(x0_train, para2_train, para3_train, para4_train, para5_train, para5_train, dims=2)
Y_train = para1[1:test]

m_1 = length(price[test:end])
x0_test = ones(m_1)
X_test = cat(x0_test, para2_test, para3_test, para4_test, para5_test, para6_test, dims=2)
Y_test =  para1[test:end]


function costFunction(X, Y, B)
    m = length(Y)
    cost = sum(((X * B) - Y)' * ((X * B) - Y))
    cost = cost/(2*m)
    v = sum(cost)
    return v
end

B = zeros(6, 1)
intialCost = costFunction(X_train, Y_train, B)

function gradientDescent(X, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    m = length(Y)
    for iteration in 1:numIterations
        loss = (X * B) - Y
        gradient = (X' * loss)/m
        B = B - learningRate * gradient
        cost = costFunction(X, Y, B)
        costHistory[iteration] = cost
    end
    return B, costHistory
end


learningRate = 0.001
newB, costHistory = gradientDescent(X_train, Y_train, B, learningRate, 30000)
YPred = X_test * newB

function rsme(YPred, Y)
    temp = (YPred-Y)' * (YPred-Y)
    temp = sum(sqrt(temp/m))
    return temp
end
function rsq(YPred, Y)
    temp = (YPred-Y)' * (YPred-Y)
    temp =1- sum(temp/((Y.-(mean(Y)))'*(Y.-(mean(Y)))))
    return temp
end

rsme2 = rsme(YPred,Y_test)
rsq2 = rsq(YPred,Y_test)

#plot(costHistory)

df = DataFrame(YPred)
CSV.write("data\\1b.csv", df)
