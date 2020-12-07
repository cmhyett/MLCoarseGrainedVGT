using Flux;
using DiffEqFlux;
using DifferentialEquations;
using DiffEqSensitivity;
using StochasticDiffEq;
using Zygote;
using LinearAlgebra;
using Optim;

include("learningProb.jl");

# run the below commented code if you don't have Zygote version > 0.4.20
# ignore(f) = f()
# Zygote.@adjoint ignore(f) = ignore(f), _ -> nothing

function TangentLearning(prob::LearningProblem)
    #TODO update batch data every few loops after debugging is done.
    numDims, numSteps, numTrajs = size(prob.trainingData);
    dt = prob.tsteps[2]-prob.tsteps[1];
    
    batchData = prob.trainingData[:,:,1:prob.miniBatchSize];
    tangents = [[(batchData[:,j+1,i] - batchData[:,j,i])/dt for j=1:(numSteps-1)] for i=1:prob.miniBatchSize];

    precisionMat = inv(prob.cov_(batchData[:,1,1], prob.initialParams, 0.0)); #TODO this only works because we have constant noise
    function loss_(p)
        ignore() do
            batchData = prob.trainingData[:,:,rand(1:numTrajs, prob.miniBatchSize)];
            tangents = [[(batchData[:,j+1,i] - batchData[:,j,i])/dt for j=1:(numSteps-1)] for i=1:prob.miniBatchSize];
        end
        f_θ = [[prob.drift_(batchData[:,j,i], p, 0.0) for j=1:(numSteps-1)] for i=1:prob.miniBatchSize];
        δ = f_θ - tangents;
        loss = [[dt*(transpose(δ[i][j])*precisionMat*δ[i][j]) - log(det(precisionMat)) for i=1:length(δ)] for j=1:length(δ[1])];
        return sum(sum(loss));
    end

    epoch = 1;
    function cb(p,l)
        display(l);
        prob.lossArray[epoch] = l;
        epoch += 1;
    end

    prob.trainingResults = DiffEqFlux.sciml_train!(loss_,
                                                  prob.initialParams,
                                                  cb=cb,
                                                  prob.optimizer,
                                                  maxiters = prob.maxiters);
end
