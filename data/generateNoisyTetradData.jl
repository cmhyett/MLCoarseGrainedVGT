using DifferentialEquations
using StochasticDiffEq
using Serialization


function generateNoisyTetradData(u0, tsteps)
    tspan = (tsteps[1], tsteps[end]);

    function trueDrift(du, u, p, t)
        ρ = reshape(u[1:9],(3,3));
        M = reshape(u[10:18],(3,3));
        k = inv(ρ);
        Π = (k*k')/tr(k*k');
        placeHolder = [(M'*ρ); (-M^2 + tr(M^2)*Π)];
        du .= [reshape(placeHolder[1:3,1:3],9); reshape(placeHolder[4:6,1:3],9)] #du/dt = [ dρdt; dMdt]
    end

    function diff(du, u, p, t)
        du .= 1.0;
    end

    groundTruthCov = 0.5*ones(18);
    groundTruthCovMat = Diagonal(groundTruthCov);
    
    noise = CorrelatedWienerProcess!(groundTruthCovMat, 0.0, groundTruthCov, zeros(18));

    sdeProb = SDEProblem(trueDrift, diff, u0[:,1], tspan, noise=noise);
    function prob_remake(problem, i, repeat)
        remake(problem, u0=u0[:,i]);
    end
    eProb = EnsembleProblem(sdeProb, prob_func=prob_remake);
    
    sol = solve(eProb, EM(), tstops=tsteps, saveat=tsteps, trajectories=size(u0)[end]);
end
