using Plots, Flux, Optim;

# unfortunately order does matter here, so be careful when modifying
mutable struct LearningProblem

    #--------------set by operator--------------#
    trainingData::Array{Float64, 3}; #shape = (18, length(tsteps), trajectories)
    validationData::Array{Float64, 3};
    tsteps::Array{Float64, 1};
    miniBatchSize::Int;
    maxiters::Int;
    optimizer; #e.g., GradientDescent(), SGD(), etc.

    drift_; #should take u,p,t as params
            # and return tangent of length(u)

    diff_; #should take u,p,t as params
           # and return noise of length(u)

    cov_; #should take u,p,t as params
          # and return covariance matrix of size (length(u), length(u))
    
    initialParams::Array{Float64, 1};


    #--------------set by training--------------#
    lossArray::Array{Float64, 1};
    trainingResults;


    #--------------set by post-processing--------------#
    plts; #dictionary

end

#--------------constructor for use by operator--------------#
function LearningProblem(trainingData::Array{Float64, 3},
                         tsteps::Array{Float64, 1},
                         miniBatchSize::Int,
                         maxiters::Int,
                         optimizer,
                         drift_,
                         diff_,
                         cov_,
                         initialParams::Array{Float64, 1})
    
    return LearningProblem(trainingData,
                           validationData,
                           tsteps,
                           miniBatchSize,
                           maxiters,
                           optimizer,
                           drift_,
                           diff_,
                           cov_,
                           initialParams,
                           zeros(maxiters), #lossArray
                           nothing, #trainingResults
                           Dict{String, Plots.Plot{Plots.GRBackend}}()); #plots
end
