using Flux;
using DiffEqFlux;
using DifferentialEquations;
using DiffEqSensitivity;
using StochasticDiffEq;
using Zygote;
using LinearAlgebra;
using Optim;


function tetradLearningKL(;seed = nothing, trainingData = nothing, miniBatchSize=1, t=nothing, maxiters = 1000)

    #I expect the user to pass in the entire batch to train on.
    # i.e., if one wants minibatch, miniBatchSize < numTrajectories
    # if one wants to train on the full set, miniBatchSize = numTrajectories
    numTrajectories = Int(size(trainingData)[3]);
    numDataPoints = Int(size(trainingData)[2]);
    tsteps = t;
    tspan = (tsteps[1], tsteps[end]);
    datasize = length(tsteps);
    dt = tsteps[2] - tsteps[1];
    
    sde_data = zeros(18, numTrajectories, numDataPoints); #18, 9 for ρ, 9 for M
    for i in 1:numTrajectories
        for j in 1:numDataPoints
            sde_data[:,i,j] .= trainingData[:,j,i]; #TODO this is dumb. I should just expect the data to be passed in correctly...but I don't want to break it rn
        end 
    end
    u0 = sde_data[:,:,1]; #grab initial condition for the solver
    
    drift_nn = FastChain(FastDense(18,30,tanh),
                         FastDense(30,30,tanh),
                         FastDense(30,30,tanh),
                         FastDense(30,18,tanh));
    #diff_nn = FastChain(FastDense(18,18));

    drift_length = length(initial_params(drift_nn));
    diff_length = 18^2;#length(initial_params(diff_nn));
    
    function drift_dudt(u,p,t)
        M = reshape(u[10:18],(3,3));
        M2 = vcat(zeros(9), reshape(M^2, 9));
        return -M2 + drift_nn(u,p);
#        return drift_nn(u,p);
    end

    function superInformedDrift(u,p,t)
        ρ = reshape(u[1:9],(3,3));
        M = reshape(u[10:18],(3,3));
        k = inv(ρ);
        Π = (k*k')/tr(k*k');
        placeHolder = [(M'*ρ); (-M^2 + tr(M^2)*Π)];
        return [reshape(placeHolder[1:3,1:3],9); reshape(placeHolder[4:6,1:3],9)] #du/dt = [ dρdt; dMdt]
    end

    function constructW(p)
        #p will be of length 18x18, but we'll throw away nearly half
        # when we enforce symmetry
        W = reshape(p,(18,18));
        return Symmetric(W);
    end
    function diff_dudt(u,p,t)
        W = constructW(p[drift_length+1:end]);
        return W*u;
    end

    u0_num = 1; # here to initialize and declare scope
    #predict_fd_sde is really here for posterity. Not used. Should clean
    # up eventually. TODO
    function predict_fd_ode(p,initialCondition)
        prob = ODEProblem{false}(superInformedDrift,
                                 initialCondition,
                                 tspan);
        return Array(solve(prob,
                           Tsit5(),#SOSRI(),
                           p=p,
                           saveat=t));#,
                           #sensealg=ForwardDiffSensitivity()));
    end

    function loss_fd_sde(p)
        loss = 0.0;
        Zygote.ignore() do
            u0_num = rand(1:size(u0)[2], miniBatchSize);
        end
        W = constructW(p[drift_length+1:end]);
        Π = inv(W); #only depends on p
        for i in 1:miniBatchSize
            pred = predict_fd_ode(p,sde_data[:,u0_num[i],1]);
            for j in 1:(datasize-1)
                f_θ = pred[:,j+1] - pred[:,j];
                δ = (sde_data[:,u0_num[i],j+1] - sde_data[:,u0_num[i],j]) - f_θ;
                loss += ((δ'*Π*δ)[1]);
            end
        end
        # for i in 1:(datasize-1)
        #     for j in 1:miniBatchSize
        #         pred = superInformedDrift(sde_data[:,u0_num[j],i], p, 0.0)#*dt;#I think this dt factor is messing the training up #using forward Euler discretization...might not be best solution
        #         δ = (sde_data[:,u0_num[j],i+1] - sde_data[:,u0_num[j],i]) - pred;
        #         loss += ((δ'*Π*δ)[1]);
        #     end
        # end
        loss *= 1000; #relative importance of optimizing log(det) vs path integral
        loss += (norm(W, 1))^2;
        # e = eigen(W);
        # for i in length(e.values)
        #     loss += exp(-(e.values[i]));
        # end
        loss -= log(det(W));
        return loss;
    end

    function cb(p,l) #callback function to observe training
        retval = false;
        display(l);
        if (l < 1.0e-6)
            retval = true;
        end
        return retval;
    end

    if (seed == nothing)
        p = rand(drift_length + diff_length);
    else
        p = seed;
    end

    #we require positive definiteness, so set W = Id
    W = Matrix(0.1I, 18, 18);
    p[drift_length+1:end] = reshape(W, length(p[drift_length+1:end]));

    res = DiffEqFlux.sciml_train(loss_fd_sde, p, cb=cb, ADAM(), maxiters = maxiters)

    #this seems like a dumb place to put this, and maybe it is. But, inside this function
    # is everything I need to define it, and I don't really want to expose any more info
    # to the caller function..
    prob = SDEProblem{false}(superInformedDrift,
                             diff_dudt,
                             u0[:,u0_num],
                             tspan);

    covMat = constructW(res.minimizer[drift_length+1:end]);
    return res,prob,covMat;

end
