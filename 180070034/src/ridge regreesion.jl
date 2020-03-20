using CSV
using Plots
using DataFrames

function standard(X)
    mean = sum(X)/length(X)
     var = sum((X .- (sum(X)/length(X))).^2) / (length(X))
     sd  = sqrt(var)
     Y = (X .- mean)/(sd)
    return Y
end

function rmse1(X,XPred)
    Z = sqrt(sum((XPred.-X).^2)/(length(X)))
    return Z
end

function r2score(X,XPred)
    Y = 1 - (sum((XPred-X).^2))/(sum((X.-(sum(X)/length(X))).^2))
    return Y
end


dataset = CSV.read("data/housingPriceData.csv");
m = length(dataset.bedrooms);


bedroom_total = (dataset.bedrooms)
bathroom_total = (dataset.bathrooms)
sqft_total = (dataset.sqft_living)

bedroom = standard(bedroom_total[1:12967])
bathroom = standard(bathroom_total[1:12967])
sqft = standard(sqft_total[1:12967])
price = dataset.price[1:12967];

bedroom_valid = standard(bedroom_total[12968:17289])
bathroom_valid = standard(bathroom_total[12968:17289])
sqft_valid = standard(sqft_total[12968:17289])
price_valid = dataset.price[12968:17289];

bedroom_test = standard(bedroom_total[17290:end])
bathroom_test = standard(bathroom_total[17290:end])
sqft_test = standard(sqft_total[17290:end])
price_test = dataset.price[17290:end];

m = length(bedroom);
x0 = ones(m);
X  = cat(x0,bedroom,bathroom,sqft,dims=2);
Y  = price;

x0 = ones(12967);
X  = cat(x0,bedroom,bathroom,sqft,dims=2);
Y  = price;
x1 = ones(4324);
X_test = cat(x1,bedroom_test,bathroom_test,sqft_test,dims=2);
x2 = ones(4322);
X_valid = cat(x2,bedroom_valid,bathroom_valid,sqft_valid,dims=2);

function costFunction(X, Y, B, λ)
    m = length(Y)
    J = sum(((X * B) - Y).^2)/(2*m)
    J= J+ λ * sum(B[2:4,1].^2)
    return J
end

B = zeros(4, 1)
learningRate = 0.3
λ = 100
numIterations = 1000
initialCost = costFunction(X, Y, B,λ)


function gradientDescent(X, Y, B, learningRate, numIterations, λ)
    costHistory = zeros(numIterations)
    m = length(Y)
    for iteration in 1:numIterations
        loss_mse = (X * B) - Y
        gradient_mse = (X' * loss_mse)/m
        dB = 2 .* B
        gradient_reg = λ * dB / m
        gradient_reg[1] = 0
        gradient = gradient_mse + gradient_reg
        B = B - learningRate * gradient
        cost = costFunction(X, Y, B, λ)
        costHistory[iteration] = cost
    end
    return B, costHistory
end


newB, costHistory = gradientDescent(X, Y, B, learningRate, numIterations,λ);

Y_validPred = X_valid * newB;

rmse_value = rmse1(price_valid,Y_validPred)
r2score_value = r2score(price_valid,Y_validPred)

println("################### Q2A #####################");
print("VALID RMSE = ");
print(rmse_value);
print("    VALID R2score = ");
println(r2score_value);
print("learning rate = ");
print(learningRate);
print("          iterations = ");
println(numIterations);
print("λ = ");
println(λ);

Y_testPred = X_test*newB;
rmse_value = rmse1(price_test,Y_testPred)
r2score_value = r2score(price_test,Y_testPred)
print("TEST RMSE = ");
print(rmse_value);
print("    TEST R2score = ");
println(r2score_value);
println(newB)
df = DataFrame(
               Price_Predicted = Y_testPred[:]
               )
CSV.write("data/2a.csv", df,writeheader = false)
