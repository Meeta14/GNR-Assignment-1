# Import necessary modules; CSV is for reading .csv files and Plots is for plotting
using CSV
using Plots
using DataFrames


dataset = CSV.read("data//housingPriceData.csv")
dataset = DataFrame(dataset)
#dataset = CSV.read("housingPriceData.csv")

# Extract columns from the dataset
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
bedroom_train = normalize(bedroom[1:test])
bathroom_train = normalize(bathroom[1:test])
sqrt_liv_train = normalize(sqrt_liv[1:test])

bedroom_test = normalize(bedroom[test:end])
sqrt_liv_test = normalize(sqrt_liv[test:end])
bathroom_test = normalize(bathroom[test:end])

x0_train = ones(test)
X_train = cat(x0_train, bedroom_train, bathroom_train, sqrt_liv_train, dims=2)
Y_train = price[1:test]



m_1 = length(price[test:end])
x0_test = ones(m_1)
X_test = cat(x0_test, bedroom_test, bathroom_test, sqrt_liv_test, dims=2)
Y_test = price[test:end]

function costFunction(X, Y, B)
    m = length(Y)
    cost = ((X * B) - Y)' * ((X * B) - Y)
    cost = cost/(2*m)
    v = sum(cost)
    return v
end

B = zeros(4, 1)
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


learningRate = 0.003
newB, costHistory = gradientDescent(X_train, Y_train, B, learningRate, 5000)

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

rsme1 = rsme(YPred,Y_test)
rsq1 = rsq(YPred,Y_test)
#plot(costHistory)
df = DataFrame(YPred)
CSV.write("data\\1b.csv", df)
