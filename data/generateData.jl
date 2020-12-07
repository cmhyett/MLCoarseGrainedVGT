using Plots, Statistics, LinearAlgebra
using Flux, DiffEqFlux, StochasticDiffEq, DiffEqBase.EnsembleAnalysis
using FileIO, Serialization
using Distributions
using Optim
# don't call this function unless you really want to. Expensive.
# function generateTrueData(u0, tspan, tsteps, numTrajectories)
function generateTrueData(tspan, tsteps, outputPath)
    
    data = deserialize("/home/cmhyett/Dropbox/2019/Math586/Summer/Prototype/trainingData.dat")
    u0 = data[:,1,:];
    numTrajectories = size(u0)[2];

    function trueDriftFunc(du, u, p, t)
        ρ = reshape(u[1:9],(3,3));
        M = reshape(u[10:18],(3,3));
        k = inv(ρ);
        Π = (k*k')/tr(k*k');
        placeHolder = [(M'*ρ); (-M^2 + tr(M^2)*Π)];
        du .= [reshape(placeHolder[1:3,1:3],9); reshape(placeHolder[4:6,1:3],9)] #du/dt = [ dρdt; dMdt]
    end

    function prob_func(prob, i, repeat)
        remake(prob,u0=u0[:,i]);
    end

    trueOdeProb = ODEProblem(trueDriftFunc, u0[:,1], tspan);

    ensembleProb = EnsembleProblem(trueOdeProb,
                                   prob_func = prob_func);#start each new problem with a different initial conditions

    #ensembleSol = solve(ensembleProb, SOSRI(), trajectories = numTrajectories);
    @time ensembleSol = solve(ensembleProb, Tsit5(), tstops = tsteps, trajectories = numTrajectories, saveat = tsteps);

    #return Array.(timeseries_point_meanvar(ensembleSol, tsteps));
    #save(ensembleSol,"/groups/chertkov/cmhyett/PureModelData/LearningTetrad/finalState.dat")
    #serialize("/groups/chertkov/cmhyett/PureModelData/LearningTetrad/trainingData.dat", ensembleSol)
    serialize(outputPath, ensembleSol);
    return ensembleSol;
end
