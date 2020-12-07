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
include("TangentSpaceLearning.jl");
include("postProcess.jl");

# args is expected to contain miniBatchSize
#  anything else will be neglected.
function main(args)
    #--------------------begin problem setup--------------------#
    #dataInputPath = "/home/cmhyett/Dropbox/2019/Math586/Summer/Prototype/trainingData.dat";
    dataInputPath = "../data/cleanODEData.dat";
    trainingData = deserialize(dataInputPath);
    #split into training/test
    # TODO shouldn't have hardcoded values here
    validationData = trainingData[:,:,:]; #TODO debug
    trainingData = trainingData[:,:,:];
    numDims, numSteps, numTrajs = size(trainingData);
    tsteps = Array{Float64, 1}(range(0.0, 0.2, length=numSteps));
    miniBatchSize = parse(Int,args[1]);
    maxiters = 500;
    optimizer = Flux.Optimise.Optimiser(ExpDecay(0.001, 0.1, 10000, 1e-7), RADAM(0.001));
    #optimizer = Flux.Optimise.Optimiser(ExpDecay(0.001, 0.1, 300, 1e-5), ADAM(0.001))
    # see https://fluxml.ai/Flux.jl/stable/training/optimisers/

    drift_nn = FastChain(FastDense(18,70,tanh),
                         FastDense(70,70,tanh),
                         FastDense(70,9));

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

    covMat = 1.0*Matrix(I, (numDims, numDims));#(0.1^(-2))*Matrix(I, (numDims, numDims));
    function cov(u,p,t)
        return covMat;
    end

    initialParams = Array{Float64, 1}(initial_params(drift_nn));
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

    println("Setup complete, beginning learning");

    @time TangentLearning(lProb);

    p = plot(lProb.lossArray,
             yaxis=:log,
             title="Loss for minibatch size of $(lProb.miniBatchSize), ODE",
             xlabel="epoch");
    savefig(p, "/groups/chertkov/cmhyett/APS/FollowUp/HyperParameterTuning/miniBatch_$(lProb.miniBatchSize).png");
    #postProcess(lProb);

end
