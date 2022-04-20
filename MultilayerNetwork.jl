using LinearAlgebra
using Random
using Plots

mutable struct NeuralNet
    #input_num
    #hidden_weight_num
    #output_num
    weight_Mats
    activation_function::Function
    activation_function_derivative::Function
end

function NeuralNetBackPropagation!(net::NeuralNet,training_set,dList,η)
    errorp = 1
    errors_list = []
    for i in 1:1000
        errorp = 0
        index = 1
        #permlist = shuffle(1:length(training_set[:,1]))
        #training_set = training_set[permlist,:]
        #dList = dList[permlist]

        for data_point in eachrow(training_set)
            Wweights = net.weight_Mats[1]
            Vweights = net.weight_Mats[2]

            nety = Vweights * data_point
            y = net.activation_function.(nety)
            y[3] = -1.0   

            o = net.activation_function.(Wweights ⋅ y)[1]

            

            errorp = errorp + (1/2)*abs((dList[index] - o))^2
            
             

            δo = (dList[index] - o) * net.activation_function_derivative(Wweights ⋅ y) 
            δy = net.activation_function_derivative.(nety) .* (δo .* Wweights)


            net.weight_Mats[1] = Wweights+((η*δo) .* y)
            inter = δy * data_point'
            net.weight_Mats[2] = Vweights + η.*(δy * data_point')

            index = index + 1
        end 
        println(errorp)
        append!(errors_list,errorp)
    end
    errors_list
end


VandW = [[-0.2 0.1 0.3], [-0.5 -0.5 0.5; 0.4 -0.3 0.4; 0.7 -0.1 0.6]]

η = 0.01

net = NeuralNet(VandW, net -> 1.716*tanh((2/3)*net), net -> 1.144*sech((2/3)*net)^2)

xor_training_set = [-1.0 -1.0 -1.0; -1.0 1.0 -1.0; 1.0 -1.0 -1.0; 1.0 1.0 -1.0]

xor_DList = [-1.0,1.0,1.0,-1.0]

errors = NeuralNetBackPropagation!(net,xor_training_set,xor_DList,η)

updatedW = net.weight_Mats[1]
updatedV = net.weight_Mats[2]

z = [1.0, 1.0, -1.0]

yvec = net.activation_function.(net.weight_Mats[2]*z)
print(yvec)
out = net.activation_function.(net.weight_Mats[1]*yvec)

print(out)

plot(1:1000,errors)