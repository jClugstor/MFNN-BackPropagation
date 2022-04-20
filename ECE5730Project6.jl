using Distributions
using Plots
include("NeuralNetandBackProp.jl")

normalDist1 = Normal(1,1)
normalDist2 = Normal(-1,2)

class1Samples = [rand(normalDist1,500) rand(normalDist1,500)]
class2Samples = [rand(normalDist2,500) rand(normalDist2,500)]

scatter(class1Samples[:,1],class1Samples[:,2],label = "Class 1")
display(scatter!(class2Samples[:,1],class2Samples[:,2], label = "Class 2"))

trainingset = [class1Samples ; class2Samples]
trainingset = [trainingset fill(-1.0,size(trainingset,1))]

ones = fill(1.0,500)
negones = fill(-1.0,500)
dList = [ones ; negones]

η = 0.1

net = NeuralNet(3,20,1, net -> 1.716*tanh((2/3)*net), net -> 1.144*sech((2/3)*net)^2)

errors = NeuralNetBackPropagation!(net,trainingset,dList,η,1000)

print(errors)
plot(1:1000,errors)

output = NeuralNetOutput(net,[2.5,2.5,-1])
print(output)




