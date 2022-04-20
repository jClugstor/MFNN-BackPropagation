using Plots
include("NeuralNetandBackProp.jl")




VandW = [[-0.2 0.1 0.3], [-0.5 -0.5 0.5; 0.4 -0.3 0.4; 0.7 -0.1 0.6]]

η = 0.01

net = NeuralNet(3,3,1, net -> 1.716*tanh((2/3)*net), net -> 1.144*sech((2/3)*net)^2)

xor_training_set = [-1.0 -1.0 -1.0; -1.0 1.0 -1.0; 1.0 -1.0 -1.0; 1.0 1.0 -1.0]

xor_DList = [-1.0,1.0,1.0,-1.0]

errors = NeuralNetBackPropagation!(net,xor_training_set,xor_DList,η,1000)

updatedW = net.weight_Mats[1]
updatedV = net.weight_Mats[2]

z = [-1.0, 1.0, -1.0]

yvec = net.activation_function.(net.weight_Mats[2]*z)
yvec[3] = -1.0

out = net.activation_function.(net.weight_Mats[1] * yvec)

print(out)

plot(1:1000,errors)
