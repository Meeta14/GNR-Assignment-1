# Import necessary modules; CSV is for reading .csv files and Plots is for plotting
using CSV
using Plots
using DataFrames

dataset = CSV.read("data//housingPriceData.csv")
dataset = DataFrame(dataset)
function partitionTrainValidationTest(data, at, at_1)
    n = nrow(data)
    idx = (1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    validation_idx = view(idx, (floor(Int, at*n)+1):(floor(Int, at_1*n)))
    test_idx = view(idx, (floor(Int, at_1*n)+1):n)
    data[train_idx,:], data[validation_idx,:], data[test_idx,:]
end
function standard(X)
    mean = sum(X)/length(X)
     var = sum((X .- (sum(X)/length(X))).^2) / (length(X))
     sd  = sqrt(var)
     Y = (X .- mean)/(sd)
    return Y
end
# Extract columns from the dataset
train,validation,test = partitionTrainValidationTest(dataset, 0.6,0.8)
course1 = train.price
course2 = standard(train.bedrooms)
course3 = standard(train.bathrooms)
course4 = standard(train.sqft_living)

m = length(course1)
x0 = ones(m)
#
# # Define the features array
X = cat(x0, course2, course3, course4, dims=2)
# column wise concatenation => dims=2
# Get the variable we want to regress
Y = course1

# Define a function to calculate cost function
function costFunction(X, Y, B, L)
    m = length(Y)
    cost = sum(((X * B) - Y).^2)/(2*m) + L*sum(abs.(B))/(2*m) - L*abs.(B[1])/(2*m)
    return cost
end

# # Initial coefficients
B = ones(4, 1)
L = 0
# Calcuate the cost with intial model parameters B=[0,0,0]
intialCost = costFunction(X, Y, B, L)

function soft_threshold(rho,Alpha)
    if rho < -1 * Alpha / 2
        return (rho + Alpha/2)
    elseif  rho > Alpha / 2
        return (rho - Alpha/2)
    else
        return 0
    end
end

function coordinateDescent(Xtrain, Xvalidate, Ytrain, Yvalidate, B, numIterations, Alpha)
    m = length(Ytrain)
    costHistory = zeros(numIterations)
    for iter in 1:numIterations
            for j in 1:length(B)
            rho = sum(Xtrain[(1:m),j] .* (Ytrain - (Xtrain * B) + (B[j].* Xtrain[(1:m),j]) )) /m
            if j == 1
                            B[j] = rho
                        else
                            B[j] = soft_threshold(rho, Alpha)

                        end
        end
        costHistory[iter] = costFunction(Xtrain,Ytrain,B,Alpha)
        end
    cost = costFunction(Xvalidate, Yvalidate, B, 0)
    return B,sqrt(2*cost),costHistory
end

################################################################
course1_validation = validation.price
course2_validation = standard(validation.bedrooms)
course3_validation = standard(validation.bathrooms)
course4_validation = standard(validation.sqft_living)
# Make predictions using the learned model; newB
m_val = length(course1_validation)
x0_val = ones(m_val)
#
# # Define the features array
X_val = cat(x0_val, course2_validation, course3_validation , course4_validation, dims=2)
# column wise concatenation => dims=2
# Get the variable we want to regress
Y_val = course1_validation

learningRate = 0.003
lambda = 20000
newB,  rmse_val, costHistory= coordinateDescent(X, X_val, Y, Y_val, B, 1000, lambda)

YPred = X_val * newB

r_sq_val = 1 - (sum((YPred.-Y_val).^2))/(sum((Y_val.-(sum(Y_val)/length(Y_val))).^2))

##############################################################
course1_test = test.price
course2_test = standard(test.bedrooms)
course3_test = standard(test.bathrooms)
course4_test = standard(test.sqft_living)
# Make predictions using the learned model; newB
m_test = length(course1_test)
x0_test = ones(m_test)

X_test = cat(x0_test, course2_test, course3_test, course4_test, dims=2)

YPred_test = X_test * newB
Y_test = course1_test

rmse = ((sum((YPred_test.-Y_test).^2))/m_test)^0.5

r_sq = 1 - (sum((YPred_test.-Y_test).^2))/(sum((Y_test.-(sum(Y_test)/length(Y_test))).^2))


df = DataFrame(YPred_test)
CSV.write("data\\2b.csv", df)
