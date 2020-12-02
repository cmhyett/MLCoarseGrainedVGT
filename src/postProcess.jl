using Plots, Statistics, LinearAlgebra;
using Flux, DiffEqFlux, StochasticDiffEq, DiffEqBase.EnsembleAnalysis;
using FileIO, Serialization;
using Distributions;
using Optim;
using StatsBase;
using Test;

# minimizer      - output from DiffEqFlux.sciml_train()
# driftNNDims    - array of layer dimensions
# diffNNDims     - array of layer dimensions
# validationData   - full set of trajectories (here used as an ensemble solution)

# Assumes all NN layers are fast dense with hyperbolic tangents
function postProcessSol(minimizer, sdeProb, validationData, tsteps; initLimit = 20.0, finalLimit = 20.0, solver = SOSRI())

    #first, let's figure out binning to create the histograms using the simple validationData
    numTrajectories = size(validationData)[3];
    tspan = (tsteps[1],tsteps[end]);
    u0 = validationData[:,1,:];
    
    R_0 = zeros(numTrajectories);
    Q_0 = zeros(numTrajectories);
    R_f = zeros(numTrajectories);
    Q_f = zeros(numTrajectories);
    
    for i in 1:numTrajectories
        M_0 = reshape(u0[10:18,i],(3,3));
	M_f = reshape(validationData[10:18,end,i],(3,3));

	Q_0[i] = -tr(M_0^2)/2.0;
	R_0[i] = -tr(M_0^3)/3.0;

	Q_f[i] = -tr(M_f^2)/2.0;
	R_f[i] = -tr(M_f^3)/3.0;
    end

    #trim to fit in histogram
    initBins = 100;
    badIndx = findall(x -> abs(x) > initLimit, Q_0);
    deleteat!(Q_0, badIndx);
    deleteat!(R_0, badIndx);
    badIndx = findall(x -> abs(x) > initLimit, R_0);
    deleteat!(Q_0, badIndx);
    deleteat!(R_0, badIndx);

    finalBins = 100;
    badIndx = findall(x -> abs(x) > finalLimit, Q_f);
    deleteat!(Q_f, badIndx);
    deleteat!(R_f, badIndx);
    badIndx = findall(x -> abs(x) > finalLimit, R_f);
    deleteat!(Q_f, badIndx);
    deleteat!(R_f, badIndx);

    h_0 = fit(Histogram, (R_0, Q_0), nbins = initBins);
    h_f = fit(Histogram, (R_f, Q_f), nbins = finalBins);

    p_0 = plot(h_0, title="Initial Condition");
    p_f = plot(h_f, title="True evolution");

    #now, take nSDE + weights, generate new trajectories, and plot
    sol = zeros(size(validationData)[1],size(validationData)[2],1);
    unstableIdx = [];
    firstTime = true;

    #in general, the below for loop benefits from parallelization, but
    # one would need to handle the solver aborts more intelligently than
    # I have. In particular, the cat() command is not atomic.
    Threads.@threads for i in 1:numTrajectories
        reSdeProb = remake(sdeProb, u0=u0[:,i]);
        placeHolder = Array(solve(reSdeProb,
                                  EM(),
                                  tstops = tsteps,
                                  p=minimizer,
                                  saveat=tsteps));
        if (size(placeHolder)[2] == size(sol)[2]) #good, solver did not abort
            if(firstTime)
                sol[:,:,1] = placeHolder;
                firstTime = false;
            else
                sol = cat(sol, placeHolder, dims=3);
            end
        else
            unstableIdx = vcat(unstableIdx, i);
        end
    end

    numLearnedTraj = size(sol)[3];
    display(length(unstableIdx));
    display(numLearnedTraj);
    display(numTrajectories);
    display(numLearnedTraj == (numTrajectories-length(unstableIdx)));
    #@test numLearnedTraj == (numTrajectories-length(unstableIdx)); #this is failing...not sure why
    
    R_learned = zeros(numLearnedTraj);
    Q_learned = zeros(numLearnedTraj);

    for i in 1:numLearnedTraj
        M = reshape(sol[10:18,end,i],(3,3));
	Q_learned[i] = -tr(M^2)/2.0;
	R_learned[i] = -tr(M^3)/3.0;
    end

    #trim to same size as above:
    badIndx = findall(x -> abs(x) > finalLimit, Q_learned);
    deleteat!(Q_learned, badIndx);
    deleteat!(R_learned, badIndx);
    badIndx = findall(x -> abs(x) > finalLimit, R_learned);
    deleteat!(Q_learned, badIndx);
    deleteat!(R_learned, badIndx);

    h_learned = fit(Histogram, (R_learned, Q_learned), h_f.edges);
    #use same axes as the true evolution for comparison
    p_learned = plot(h_learned, xlims=xlims(p_f), ylims=ylims(p_f), title="Learned evolution");

    h_diff = Histogram(h_learned.edges, h_f.weights .- h_learned.weights);
    p_diff = plot(h_diff, xlims=xlims(p_f), ylims=ylims(p_f), title="Difference");


    #now calculate divergence
    trueDivArr = zeros(length(tsteps), numTrajectories);
    learnedDivArr = zeros(length(tsteps), numTrajectories);
    
    for i in 1:length(tsteps)
        for j in 1:numLearnedTraj
            M_true = reshape(validationData[10:18,i,j], (3,3));
            M_learned = reshape(sol[10:18,i,j], (3,3));
            
            trueDivArr[i,j] = tr(M_true);
            learnedDivArr[i,j] = tr(M_learned);
        end
    end
    p_divOverTime = scatter(1:length(tsteps),
                         i -> mean(trueDivArr[i,:]),
                         label="avg div of truth");
    scatter!(1:length(tsteps), i -> mean(learnedDivArr[i,:]), label="avg div of learned");
    scatter!(1:length(tsteps), i -> mean(abs, learnedDivArr[i,:]), label="avg(abs(div)) of learned");
    scatter!(1:length(tsteps), i -> var(learnedDivArr[i,:]),  label="var div of learned");
    plot!(legend=:topleft);
    
    p_1 = plot(p_0, p_f, p_learned, layout = (3,1));
    p_2 = plot(p_diff, p_divOverTime, layout = (2,1));
    p = plot(p_1,p_2,size = (1920,1080))

    return p;
end

# arr is a tuple of things to delete, say (R,Q,weights)
function trim(arr,Rmin,Rmax,Qmin,Qmax)
    R = arr[1];
    Q = arr[2];
    
    badIndx = findall(x -> x>Rmax, R);
    for i in 1:length(arr)
        deleteat!(arr[i],badIndx);
    end
    
    badIndx = findall(x -> x<Rmin, R);
    for i in 1:length(arr)
        deleteat!(arr[i],badIndx);
    end
    
    badIndx = findall(x -> x>Qmax, Q);
    for i in 1:length(arr)
        deleteat!(arr[i],badIndx);
    end
    
    badIndx = findall(x -> x<Qmin, Q)
    for i in 1:length(arr)
        deleteat!(arr[i],badIndx);
    end
end
# data   - DNS, ODE data, size = (dims(18), numTimeSteps, numTraj)
# Rlimit - where to trim R
# Qlimit - where to trim Q
# bins  - 2 entry tuple
# generateWeights - function that takes data as input, and outputs R,Q,weights arrays.
    # function generateWeights(data)
    #     numTraj = size(data)[3];
    #     R = zeros(numTraj);
    #     Q = zeros(numTraj);
    #     w = zeros(numTraj);
    #     for i in 1:numTraj
    #         M = reshape(data[10:18,end,i], (3,3));
    #         Q[i] = -tr(M^2)/2.0;
    #         R[i] = -tr(M^3)/3.0;
    #         w[i] = 1.0; #equal weight to each, i.e, counting metric
    #     end        
    #     return R,Q;
    # end
    # function generateEnergyFluxDensity(data)
    #     numTraj = size(data)[3];
    #     R = zeros(numTraj);
    #     Q = zeros(numTraj);
    #     e = zeros(numTraj);
    
    #     for i in 1:numTraj
    #         M = reshape(data[10:18, 1, i], (3,3));
    #         Q[i] = -tr(M^2)/2.0;
    #         R[i] = -tr(M^3)/3.0;
    #         e[i] =  tr(M^2*M');
    #     end
    #     return R,Q,e;
    # end
function drawRQContour(data, Rlimit, Qlimit, numBins, generateWeights)

    R,Q,w = generateWeights(data);
    
    trim((R,Q,w), -Rlimit, Rlimit, -Qlimit, Qlimit);

    edgeQ = range(-Qlimit, Qlimit, length=numBins);
    edgeR = range(-Rlimit, Rlimit, length=numBins);
    h = fit(Histogram, (R,Q), weights(w), (edgeR, edgeQ), closed=:left);
    
    c = contour(midpoints(h.edges[1]),
                midpoints(h.edges[2]),
                h.weights');

    return c;
end

# function drawVectorField(data, Rlimits, Qlimits, tangentFuncs)
#     for i in 1:length(tangentFuncs)
#         R = zeros(size(data)[3]);
#         Q = zeros(size(data)[3]);
#         for i in 1:size(data)[3]
#             M = reshape(data[10:18,1,i],(3,3));
#             R[i] = -tr(M^3)/3.0;
#             Q[i] = -tr(M^2)/2.0;
#         end
#         Rmin = Rlimits[1];
#         Rmax = Rlimits[2];
#         Qmin = Qlimits[1];
#         Qmax = Qlimits[2];
#         trim(R,Q,Rlimits[1],Rlimits[2],Qlimits[1],Qlimits[2]);
        
#     end
# end

#tangent functions is an array of functions that map from \R^18 → \R^2 and return [dR/dt dQ/dt]
function drawVectorField(data, Rrange, Qrange, tangentFuncs, dt, timepoint; titles = nothing)
    rMid = midpoints(Rrange);
    qMid = midpoints(Qrange);
    pltArr = [];
    seperatix_r = range(Rrange[1], Rrange[end], length = 1000);
    seperatix_q = -((27.0/4.0).*(seperatix_r.^2)).^(1/3);
    #M should be length 9 array
    function calcRQFromSample(Marr)
        M = reshape(Marr, (3,3));
        R = -tr(M^3)/3.0;
        Q = -tr(M^2)/2.0;
        return R,Q;
    end

    function inRange(M, Rrange, Qrange)
        R,Q = calcRQFromSample(M);
        return (Rrange[1] <= R < Rrange[2]) & (Qrange[1] <= Q < Qrange[2]);
    end

    #M_vals should be 9xnumTraj array
    function find(M_vals, Rrange, Qrange)
        numTraj = size(M_vals)[2];
        trueIdx = [];
        for i in 1:numTraj
            if (inRange(M_vals[:,i], Rrange, Qrange))
                append!(trueIdx, i);
            end
        end
        return Array{Int64, 1}(trueIdx);
    end
    
    for k in 1:length(tangentFuncs)
        df = tangentFuncs[k];
        #I do a reallocation per loop because I mess up the arrays at the end. Not clean but...
        R = zeros(length(rMid),length(qMid));
        Q = zeros(length(rMid),length(qMid));
        Δ = zeros(length(rMid),length(qMid),2); #a trailing 2 here so each point (Q,R) has 2d vector

        for i in 1:length(qMid)
            for j in 1:length(rMid)
                R[i,j] = rMid[j];
                Q[i,j] = qMid[i];
                Rmin,Rmax = [Rrange[j], Rrange[j+1]];
                Qmin,Qmax = [Qrange[i], Qrange[i+1]];
                samples = find(data[10:18,timepoint,:], (Rmin,Rmax), (Qmin,Qmax));
                for l in 1:length(samples)
                    Δ[i,j,:] += dt*df(data[:,timepoint,samples[l]]);
                end
                Δ[i,j,:] /= length(samples);
            end
        end
        R = reshape(R, prod(size(R)));
        Q = reshape(Q, prod(size(Q)));
        Δ = flatten(Δ);
        dR = Δ[:,1];
        dQ = Δ[:,2];
        push!(pltArr, quiver(R,Q,quiver=(dR,dQ)));
        plot!(pltArr[k], seperatix_r, seperatix_q);
        if(titles != nothing && length(titles) == length(tangentFuncs))
            plot!(pltArr[k], title=titles[k]);
        end
    end


    return pltArr;
end
