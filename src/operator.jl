using Flux;
using DifferentialEquations;
using DiffEqSensitivity;
using StochasticDiffEq;
using Zygote;
using LinearAlgebra;
using Optim;
using Plots;
using Serialization;

include("learningProb.jl");
#include("postProcess.jl"); #post processing currently broken
include("stochasticExtensionOfTetrad.jl");
include("generateNoisyTetradData.jl");

function parseTrainingData(data, dt)
    numDims, numSteps, numTrajs = size(data);
    
    x = zeros(numDims, numTrajs*(numSteps-1)); #the last time-step is not a valid sample
    y = [(data[:,j+1,i] - data[:,j,i])/dt for i in 1:numTrajs for j in 1:(numSteps-1)]; #generate array of array of tangents
    y = hcat(y...); #y is 9900 arrays of 18 each. We want size(y)=(18,9900)

    stride = numSteps-1;
    for i in 1:numTrajs
        for j in 1:(numSteps-1)
            x[:,stride*(i-1)+(j-1)+1] = data[:,j,i]; #the perils of 1-based indexing.
                                                     #I suppose i,j could start at 0, and then add one on RHS.
        end
    end

    return x,y;
end


# for this test args will be ignored
function main(args)
    #--------------------begin problem setup--------------------#
    initialConditionsPath = "/home/cmhyett/Dropbox/Research/TetradTurbulence/StochasticExtensionOfTetrad/MLCoarseGrainedVGT/src/julia_NiceInitialConditions.dat";
    initialConditions = deserialize(initialConditionsPath);
    initialConditions = initialConditions[:,1:1000];
    tsteps = Array{Float64, 1}(range(0.0, 0.1, length=30));
    trainingData = generateNoisyTetradData(initialConditions, tsteps);
    
    #split into training/test
    validationData = trainingData[:,:,:];
    trainingData = trainingData[:,:,:];
    numDims, numSteps, numTrajs = size(trainingData);
    dt = tsteps[2] - tsteps[1];

    x,y = parseTrainingData(trainingData, dt);
    
    miniBatchSize = 500; #see HyperParameterTuning for justification
    maxiters = 1000; #plateau's around 4k
    optimizer = Flux.Optimise.Optimiser(ExpDecay(0.001, 0.5, 1000, 1e-7), ADAM(0.001));
    # see https://fluxml.ai/Flux.jl/stable/training/optimisers/

    hiddenDim = 80; #seems to be good mix of speed and expressibility
    θ_f, drift_nn = Flux.destructure(Chain(Dense(18,hiddenDim,tanh),
                                           Dense(hiddenDim,hiddenDim,tanh),
                                           Dense(hiddenDim,9)));

    function parameterizedClosure(u,p)
        return drift_nn(p)(u);
    end

    function parameterizedPrecisionMat(u,p)
        return p;
    end

    driftLength = 0; #length(θ_f);
    diffLength = 18;
    
    #apparently Zygote has trouble if there are extraneous values being calculated, in the midst of computation
    # so remove your shit
    #This is still painfully slow...It seems a lot of time is spent allocating/de-allocating
    # function drift(u,p,t)
    #     numSamples = size(u)[end];
    #     ρ = [reshape(u[1:9,i], (3,3)) for i in 1:numSamples];
    #     M = [reshape(u[10:18,i], (3,3)) for i in 1:numSamples];
    #     # k = [inv(ρ[i]) for i in 1:numSamples];
    #     # Π = [(k[i]*(k[i]'))/tr(k[i]*(k[i]')) for i in 1:numSamples];
    #     closure = parameterizedClosure(u,p);#drift_nn(p)(u); #of size u
    #     intermediate = [reshape(closure[:,i], (3,3)) for i in 1:numSamples];
    #     placeholder = [ [reshape(M[i]'*ρ[i], 9); reshape(-M[i]^2 + intermediate[i], 9)] for i in 1:numSamples];
    #     return hcat(placeholder...);
    # end

    function perfectDrift(u,p,t)
        numSamples = size(u)[end];
        ρ = [reshape(u[1:9,i], (3,3)) for i in 1:numSamples];
        M = [reshape(u[10:18,i], (3,3)) for i in 1:numSamples];
        k = [inv(ρ[i]) for i in 1:numSamples];
        Π = [(k[i]*(k[i]'))/tr(k[i]*(k[i]')) for i in 1:numSamples];
        closure = [tr(M[i]^2)*Π[i] for i in 1:numSamples];
        placeholder = [ [reshape(M[i]'*ρ[i], 9); reshape(-M[i]^2 + closure[i], 9)] for i in 1:numSamples];
        return hcat(placeholder...);
    end
#     function drift(u,p,t)
#         display(size(u));
#         ρ = reshape(u[1:9],(3,3));
#         M = reshape(u[10:18],(3,3));
#         k = inv(ρ);
#         Π = (k*k')/tr(k*k');
#         placeHolder = [(M'*ρ); (-M^2 + reshape(drift_nn(u,p), (3,3)))];
# #        placeHolder = [(M'*ρ); (-M^2 + tr(M^2)*Π)];
#         return [reshape(placeHolder[1:3,1:3],9); reshape(placeHolder[4:6,1:3],9)] #du/dt = [ dρdt; dMdt]
#     end

    #TODO: at this point we should focus on precision matrix, and the covariance should be a thought for post-processing
    #TODO: post processing will currently break since cov *should* use Diagonal(p.^(-1))
    #currently, the post-processing depends on the covariance matrix being a constant.
    # If you want some kind of scaling behavoir, it should occur in diff
    function cov(u,p,t)
        return Diagonal(p[(driftLength + diffLength + 1):end]);
    end
    function cov(p)
        return Diagonal(p[(driftLength + diffLength + 1):end]);
    end
    #I believe this is the correct functionality if we want a constant scaled/correlated
    # noise process.
    function diff(u,p,t)
        return ones(size(u));
    end
    
    #covariance matrix is length(u)^2
    initialParams = [];
    if (driftLength > 0)
        initialParams = vcat(initialParams, θ_f)
    end
    if (diffLength > 0)
        initialParams = vcat(initialParams, ones(diffLength)); #currently using diagonal covariance
    end
    initialParams = Array{Float64}(initialParams);
    initialParams = ones(18);

    @assert length(initialParams) == (driftLength + diffLength)
    
    #--------------------end problem setup--------------------#

    lProb = LearningProblem(trainingData,
                            x,
                            y,
                            validationData,
                            tsteps,
                            miniBatchSize,
                            maxiters,
                            optimizer,
                            perfectDrift,
                            diff,
                            cov,
                            initialParams,
                            driftLength,
                            diffLength,
                            parameterizedClosure,
                            parameterizedPrecisionMat);

    println("Setup complete, beginning learning");


    #@time TangentLearning(lProb);

    #basepath = "/groups/chertkov/cmhyett/StochasticExtensionOfTetrad/MLCoarseGrainedVGT/results";
    #direc = "FirstTryAtTraining"
    #serialize("$(basepath)/$(direc)/trainedProblem.dat", lProb);
    # differentiator = "noWeight";
    # savefig(plts[1], "$(basepath)/$(direc)/trueRQEvolution_$(differentiator).png");
    # savefig(plts[2], "$(basepath)/$(direc)/learnedRQEvolution_$(differentiator).png");
    # savefig(plts[3], "$(basepath)/$(direc)/diffRQEvolution_$(differentiator).png");
    # savefig(plts[4], "$(basepath)/$(direc)/RQEvolutionRollup_$(differentiator).png");
    # savefig(plts[5], "$(basepath)/$(direc)/loss_hiddenDim_$(differentiator).png");

    return lProb;
end

#main(ARGS)
