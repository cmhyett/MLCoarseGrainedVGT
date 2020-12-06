using Flux;
using DiffEqFlux;
using DifferentialEquations;
using DiffEqSensitivity;
using StochasticDiffEq;
using Zygote;
using LinearAlgebra;
using Optim;
using Plots;
using Serialization;

include("learningProb.jl");
include("tetradLearningKL.jl");
include("postProcessing.jl");

#--------------------begin problem setup--------------------#
dataInputPath = "/home/cmhyett/Dropbox/2019/Math586/Summer/Prototype/trainingData.dat";
trainingData = deserialize(dataInputPath);
#split into training/test
# TODO shouldn't have hardcoded values here
validationData = trainingData[:,:,1001:end];
trainingData = trainingData[:,:,1:1000];
numDims, numSteps, numTrajs = size(trainingData);
tsteps = Array{Float32, 1}(range(0.0, 1.0, length=numSteps));
miniBatchSize = 10;
maxiters = 100;
optimizer = ADAM(0.05);

drift_nn = FastChain(FastDense(18,50,tanh),
                     FastDense(50,50,tanh),
                     FastDense(50,9));

function drift(u,p,t)
    ρ = reshape(u[1:9],(3,3));
    M = reshape(u[10:18],(3,3));
    k = inv(ρ);
    Π = (k*k')/tr(k*k');
    placeHolder = [(M'*ρ); (-M^2 + reshape(drift_nn(u,p), (3,3)))];
    #placeHolder = [(M'*ρ); (-M^2 + tr(M^2)*Π)];
    return [reshape(placeHolder[1:3,1:3],9); reshape(placeHolder[4:6,1:3],9)] #du/dt = [ dρdt; dMdt]
end

function diff(u,p,t)
    return 0.1*zeros(size(u));
end

covMat = Diagonal(diff(zeros(numDims), 0, 0).^(-2))*Matrix(I, (numDims, numDims));
function cov(u,p,t)
    return covMat;
end

intialParams = initial_params(driftNN);
#--------------------end problem setup--------------------#


lProb = LearningProblem(trainingData,
                        validationData,
                        tsteps,
                        miniBatchSize,
                        maxiters,
                        optimizer,
                        drift,
                        diff,
                        cov,
                        initialParams);

TangentLearning(lProb);

postProcess(lProb);

