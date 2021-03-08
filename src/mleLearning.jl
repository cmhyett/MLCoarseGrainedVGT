using DifferentialEquations;
using StochasticDiffEq;
using Zygote;
using Flux;
using LinearAlgebra;
using Statistics;

include("learningProb.jl");

#I expect x and y to be of size (18,numSamples), x[:,i] being the initial condition, y[:,i] being the ground-truth tangent
function Learn(prob::LearningProblem)

    driftStart = 1;
    if (prob.driftLength > 0)
        driftEnd = driftStart + (prob.driftLength-1);
        diffStart = driftEnd+1;
    else
        driftEnd = 0;
        diffStart = 1;
    end
    diffEnd = diffStart + (prob.diffLength-1);
    
    dt = (prob.tsteps[2] - prob.tsteps[1]);
    numSamples = size(prob.x)[end];

    #-------Begin training stuff------#

    function predict_(x, p)
        #assume autonomous
        return prob.drift_(x, p, 0.0);
    end

    function loss_(x, y, p)
        δ = predict_(x,p) .- y;

        loss = 0.0;
        for i in 1:prob.miniBatchSize
            tempPrecisionMat = prob.parameterizedPrecisionMat_(x[:,i],p[diffStart:diffEnd]);
            loss += dt*dot(δ[:,i],Diagonal(tempPrecisionMat),δ[:,i]) - log(prod(tempPrecisionMat));
        end
        return (1/prob.miniBatchSize)*loss;
    end

    #statically allocate for training
    driftGrad = zeros(prob.driftLength);
    driftGradHolder = zeros(prob.driftLength, prob.miniBatchSize);
    predictionErrorGrad = zeros(prob.diffLength);
    predictionErrorGradHolder = zeros(prob.diffLength, prob.miniBatchSize);
    detPiece = zeros(prob.diffLength);
    detPieceHolder = zeros(prob.diffLength, prob.miniBatchSize);
    
    precisionMat = zeros(18,prob.miniBatchSize);
    display(size(precisionMat));
    precisionMat[:,2];
    lossPerStep = zeros(prob.miniBatchSize);
    
    function lossAndGrads(x,y,p)
        δ = predict_(x,p) .- y;

        for i in 1:prob.miniBatchSize
            precisionMat[:,i] .= prob.parameterizedPrecisionMat_(x[:,i],p[diffStart:diffEnd]);
            lossPerStep[i] = dt*dot(δ[:,i],Diagonal(precisionMat[:,i]),δ[:,i]) - log(prod(precisionMat[:,i]));
        end
        sumLoss = (1/prob.miniBatchSize)*sum(lossPerStep);

        #-----------------------------drift grad-----------------------------#

        if (prob.driftLength > 0)
            #x is state variable, w is weight vector, pullback(p->f(x,p), p)[2](w)[1] performs the tensor contraction
            # (∂/∂p_i f_k) w^k
            ∂f(x,w) = pullback(p->prob.parameterizedDriftClosure_(x,p), p[driftStart:driftEnd])[2](w)[1];
            Threads.@threads for i in 1:prob.miniBatchSize
                weights = (precisionMat[10:18,i] .* δ[10:18,i])
                driftGradHolder[:,i] = ∂f(x[:,i], weights);
            end
            driftGrad = ((2*dt)/prob.miniBatchSize)*sum(driftGradHolder, dims=2);
        end

        #-----------------------------diff grad-----------------------------#
        diffGrad = zeros(prob.diffLength);
        
        if (prob.diffLength > 0)
            #x is state variable, w is weight vector, pullback(p->f(x,p), p)[2](w)[1] performs the tensor contraction
            # (∂/∂p_i f_k) w^k
            ∂Π(x,w) = pullback(p_->prob.parameterizedPrecisionMat_(x,p_), p[diffStart:diffEnd])[2](w)[1];
            
            Threads.@threads for i in 1:prob.miniBatchSize
                predictionErrorGradHolder[:,i] = ∂Π(x[:,i], abs2.(δ[:,i]));
                detPieceHolder[:,i] = ∂Π(x[:,i], precisionMat[:,i].^(-1));
            end
            predictionErrorGrad = dt*sum(predictionErrorGradHolder, dims=2);
            detPiece = sum(detPieceHolder, dims=2);
            
            diffGrad = (predictionErrorGrad - detPiece)/prob.miniBatchSize;
        end

        grads = vcat(driftGrad, diffGrad);
        grads = reshape(grads, size(grads)[1]);
        return sumLoss,grads;
    end

    function customTrain!(ps, x, y, opt)
        l,gs = lossAndGrads(x,y,ps);
        Flux.update!(opt, ps, gs);
        return l;
    end

    #--------------test gradient--------------#
    println("beginning test of gradients");
    temp = prob.miniBatchSize;
    prob.miniBatchSize = 5;
    sampleIdxs = rand(1:numSamples, prob.miniBatchSize);
    batchX = prob.x[:, sampleIdxs];
    batchY = prob.y[:, sampleIdxs];
    l, analyticGrad = lossAndGrads(batchX, batchY, prob.params);
    zgrad = Zygote.gradient(p->loss_(batchX, batchY ,p), prob.params)[1];
    prob.miniBatchSize = temp; #reset state of problem before we possibly prematurely exit

    if (zgrad ≈ analyticGrad) #passed
        println("gradient check passed!");        
    else
        println("gradient check failed!");
        println(" zgrad:    analytic grad:");
        display(hcat(zgrad, analyticGrad));
        @assert true == false
    end
    #-----------------------------------------#

    batchX = zeros(size(prob.x)[1], prob.miniBatchSize);
    batchY = zeros(size(prob.x)[1], prob.miniBatchSize);
    sampleIdxs = zeros(Int, prob.miniBatchSize);
    
    #training loop
    for i in 1:prob.maxiters
        sampleIdxs = rand(1:numSamples, prob.miniBatchSize);
        batchX = prob.x[:, sampleIdxs];
        batchY = prob.y[:, sampleIdxs];
        prob.lossArray[i] = customTrain!(prob.params, batchX, batchY, prob.optimizer);
        if (i % 50 == 0)
            display(mean(prob.lossArray[i-49:i]));
        end
    end

    return prob.params;
end

