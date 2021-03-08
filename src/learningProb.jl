using Plots, Flux, Optim;

# unfortunately order does matter here, so be careful when modifying
mutable struct LearningProblem

    #--------------set by operator--------------#
    trainingData::Array{Float64, 3}; #shape = (18, length(tsteps), trajectories)
    
    x::Array{Float64, 2}; #shape=(numdims, numTrajs*(numSteps-1)
    y::Array{Float64, 2}; #shape=(numdims, numTrajs*(numSteps-1)
    
    validationData::Array{Float64, 3};
    tsteps::Array{Float64, 1};
    miniBatchSize::Int;
    maxiters::Int;
    optimizer; #e.g., GradientDescent(), SGD(), etc.

    drift_; #should take u,p,t as params
            # and return tangent of length(u)

    diff_; #should take u,p,t as params
           # and return noise of length(u)

    # TODO, can probably remove cov_
    cov_; #should take u,p,t as params
          # and return covariance matrix of size (length(u), length(u))
    
    params::Array{Float64, 1};

    #parameter lengths
    driftLength;
    diffLength;

    #functions called by drift_, diff_ respectively
    parameterizedDriftClosure_;
    parameterizedPrecisionMat_;

    #--------------set by training--------------#
    lossArray::Array{Float64, 1};


    #--------------set by post-processing--------------#
    plts; #dictionary

end

#--------------constructor for use by operator--------------#
function LearningProblem(trainingData::Array{Float64, 3},
                         x::Array{Float64, 2},
                         y::Array{Float64, 2},
                         validationData::Array{Float64, 3},
                         tsteps::Array{Float64, 1},
                         miniBatchSize::Int,
                         maxiters::Int,
                         optimizer,
                         drift_,
                         diff_,
                         cov_,
                         initialParams::Array{Float64, 1},
                         driftLength,
                         diffLength,
                         parameterizedDriftClosure_,
                         parameterizedPrecisionMat_)
    
    return LearningProblem(trainingData,
                           x,
                           y,
                           validationData,
                           tsteps,
                           miniBatchSize,
                           maxiters,
                           optimizer,
                           drift_,
                           diff_,
                           cov_,
                           initialParams,
                           driftLength,
                           diffLength,
                           parameterizedDriftClosure_,
                           parameterizedPrecisionMat_,
                           zeros(maxiters), #lossArray
                           Dict{String, Plots.Plot{Plots.GRBackend}}()); #plots
end
