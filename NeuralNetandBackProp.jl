using LinearAlgebra
using Random
using Distributions

mutable struct NeuralNet
    input_num
    hidden_neuron_num
    output_num
    weight_Mats
    activation_function::Function
    activation_function_derivative::Function
end

#Constructor that automatically sets up the weights based on the number of inputs, hidden neurons, and outputs.
NeuralNet(input_num::Number, hidden_neuron_num::Number, output_num::Number, activation_function::Function, activation_function_derivative::Function) = NeuralNet(input_num, hidden_neuron_num, output_num,[rand(Uniform(-2.4/input_num,2.4/input_num),(output_num,hidden_neuron_num)), rand(Uniform(-2.4/input_num,2.4/input_num), (hidden_neuron_num,input_num))], activation_function, activation_function_derivative)

function NeuralNetBackPropagation!(net::NeuralNet,training_set, dList, η, iters)
    errorp = 1
    errors_list = []
    for i in 1:iters
        errorp = 0
        index = 1
        permlist = shuffle(1:length(training_set[:,1]))
        training_set = training_set[permlist,:]
        dList = dList[permlist]

        for data_point in eachrow(training_set)
            Wweights = net.weight_Mats[1]
            Vweights = net.weight_Mats[2]

            nety = Vweights * data_point
            y = net.activation_function.(nety)
            y[3] = -1.0

            netk = Wweights * y
            o = net.activation_function.(netk)
            errorp = errorp + (1/2)*norm((dList[index] .- o))^2
            
            δo = (dList[index] .- o) .* net.activation_function_derivative.(netk) 
            δy = net.activation_function_derivative.(nety) .* (Wweights .* δo)'

            net.weight_Mats[1] = Wweights+((η*δo) .* y')
            net.weight_Mats[2] = Vweights + η.*(δy * data_point')

            index = index + 1
        end 
        append!(errors_list,errorp)
    end
    errors_list
end

function NeuralNetOutput(net::NeuralNet,input)
    yvec = net.activation_function.(net.weight_Mats[2]*input)
    yvec[length(input)] = -1.0

    out = net.activation_function.(net.weight_Mats[1] * yvec)
    return out
end


