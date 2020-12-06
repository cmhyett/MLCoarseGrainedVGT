using Flux;
using DiffEqFlux;
using DifferentialEquations;
using DiffEqSensitivity;
using StochasticDiffEq;
using Zygote;
using LinearAlgebra;
using Optim;

include("learningProb.jl");

function TangentLearning(prob::LearningProblem)
    batchData = prob.trainingData[:,:,1:prob.miniBatchSize];
    numDims, numSteps, numTrajs = size(batchData);
    dt = prob.tsteps[2]-prob.tsteps[1];
    
    tangents[:,j,i] = [(batchData[:,j+1,i] - batchData[:,j,i])/dt for j=1:(numSteps-1) for i=1:numTrajs];

    precisionMat = inv(prob.cov_); #TODO this only works because we have constant noise
    function loss_(p)
        f_θ = [prob.drift_(batchData[:,j,i], p, 0.0) for j=1:(numSteps-1) for i=1:numTrajs];
        δ = f_θ - tangents;
        loss = [dt*(transpose(δ)*precisionMat*δ) - log(det(precisionMat))];
        return sum(loss);
    end

    epoch = 1;
    function cb(p,l)
        display(l);
        prob.lossArray[epoch] = l;
        epoch += 1;
    end

    prob.trainingResults = DiffEqFlux.sciml_train(loss_,
                                                  prob.initialParams,
                                                  cb=cb,
                                                  prob.optimizer,
                                                  maxiters = prob.maxiters);
end
