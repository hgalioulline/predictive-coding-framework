function [epsilonData, phiData, priorMeanData] = multi_layered_pixels_advanced(numLayers, timeUnits, ...
    deltaT, numTrials, sigmaLR, thetaLR, priorMeanLR, lesion, priorMean, priorVariance, observedImages, ...
    observationFunction)

% create all layers
phiLayers = cell(1, numLayers);
epsilonLayers = cell(1, numLayers);
thetaLayers = cell(1, numLayers - 1);
sigmaLayers = cell(1, numLayers);

% phi_1's and sigma_1
numPixels = length(observedImages{1});
numImages = length(observedImages);
observationNoise = 1.0 * eye(numPixels);  % sigma_u

% initialize layers
phiLayers(1) = observedImages(1);
phiLayers(2:numLayers) = {priorMean};
epsilonLayers(:) = {zeros(numPixels, 1)};
thetaLayers(:) = {1 * eye(numPixels)};
sigmaLayers(1) = {observationNoise};
sigmaLayers(2:numLayers - 1) = {eye(numPixels)};

switch lesion
    case 'Perturb Sigma'
        if numLayers > 2
            randSigmaLayer = randi(numLayers);
            randSigmaElements = randi([1, numPixels], ceil(numPixels ^ 2 * 0.1), 2);
            for randSigmaElementIdx = 1:length(randSigmaElements)
                sigmaLayers{randSigmaLayer}(randSigmaElements(randSigmaElementIdx, 1), randSigmaElements(randSigmaElementIdx, 2)) = 10;
            end
        end
    case 'Loose Prior'
        priorVariance = priorVariance * 5.0;
end

sigmaLayers(numLayers) = {priorVariance};

% initialize cell arrays for gui
epsilonData = cell(1, numTrials);
phiData = cell(1, numTrials);
priorMeanData = cell(1, numTrials);

% pick theta element to zero pad
if strcmp(lesion, 'Zero Pad Theta')
    randThetaLayer = randi(numLayers - 1);
    randThetaElements = randi([1, numPixels], ceil(numPixels ^ 2 * 0.1), 2);
end

for imageIdx = 1:numImages
    for trialIdx = 2:numTrials
        for t = 2:timeUnits / deltaT
            % layer 1
            epsilonLayers{1}(:, t) = epsilonLayers{1}(:, t - 1) + deltaT * ...
                (observedImages{imageIdx} - thetaLayers{1}(:, :, trialIdx - 1) * h(phiLayers{2}(:, t - 1)) - ...
                                        sigmaLayers{1}(:, :, trialIdx - 1) * epsilonLayers{1}(:, t - 1));

            % layers 2 to N - 1
            for layerIdx = 2:numLayers - 1
                phiLayers{layerIdx}(:, t) = phiLayers{layerIdx}(:, t - 1) + deltaT * ...
                    (-epsilonLayers{layerIdx}(:, t - 1) + dh(phiLayers{layerIdx}(:, t - 1)) .* ...
                        thetaLayers{layerIdx - 1}(:, :, trialIdx - 1)' * epsilonLayers{layerIdx - 1}(:, t - 1));
                epsilonLayers{layerIdx}(:, t) = epsilonLayers{layerIdx}(:, t - 1) + deltaT * ...
                    (phiLayers{layerIdx}(:, t - 1) - thetaLayers{layerIdx}(:, :, trialIdx - 1) * ...
                        h(phiLayers{layerIdx + 1}(:, t - 1)) - sigmaLayers{layerIdx}(:, :, trialIdx - 1) * ...
                            epsilonLayers{layerIdx}(:, t - 1));
            end

            % layer N
            phiLayers{numLayers}(:, t) = phiLayers{numLayers}(:, t - 1) + deltaT * ...
                    (-epsilonLayers{numLayers}(:, t - 1) + dh(phiLayers{numLayers}(:, t - 1)) .* ...
                                 thetaLayers{numLayers - 1}(:, :, trialIdx - 1)' * epsilonLayers{numLayers - 1}(:, t - 1));
            epsilonLayers{numLayers}(:, t) = epsilonLayers{numLayers}(:, t - 1) + deltaT * ...
                        (phiLayers{numLayers}(:, t - 1) - priorMean(:, trialIdx - 1) - sigmaLayers{numLayers}(:, :, trialIdx - 1) * ...
                                                                epsilonLayers{numLayers}(:, t - 1));
        end

        % update sigma synaptic weights
        for sigmaLayerIdx = 1:length(sigmaLayers)
            sigmaLayers{sigmaLayerIdx}(:, :, trialIdx) = sigmaLayers{sigmaLayerIdx}(:, :, trialIdx - 1) + sigmaLR * ...
                (0.5 * (epsilonLayers{sigmaLayerIdx}(:, end) * epsilonLayers{sigmaLayerIdx}(:, end)' - ...
                    inv(sigmaLayers{sigmaLayerIdx}(:, :, trialIdx - 1))));
        end

        % update theta synaptic weights
        for thetaLayerIdx = 1:length(thetaLayers)
            thetaLayers{thetaLayerIdx}(:, :, trialIdx) = thetaLayers{thetaLayerIdx}(:, :, trialIdx - 1) + thetaLR * ...
                (epsilonLayers{thetaLayerIdx}(:, end) * h(phiLayers{thetaLayerIdx + 1}(:, end))');
        end
        
        if strcmp(lesion, 'Zero Pad Theta')
            for randThetaElementIdx = 1:length(randThetaElements(:, 1))
                thetaLayers{randThetaLayer}(randThetaElements(randThetaElementIdx, 1), ...
                    randThetaElements(randThetaElementIdx, 2)) = 0;
            end
        end

        % update prior mean synaptic weights
        priorMean(:, trialIdx) = priorMean(:, trialIdx - 1) + priorMeanLR * epsilonLayers{numLayers}(:, end);
        
        % save phi, epsilon, and priorMean at each trial
        phiData{trialIdx} = {deltaT:deltaT:timeUnits, phiLayers};
        epsilonData{trialIdx} = {deltaT:deltaT:timeUnits, epsilonLayers};
        priorMeanData{trialIdx} = {1:length(priorMean), priorMean};
    end

    % prepare layers for next iteration
    for layerIdx = 1:numLayers - 1
        % neurons
        epsilonLayers{layerIdx}(:, 1) = epsilonLayers{layerIdx}(:, end);
        phiLayers{layerIdx}(:, 1) = phiLayers{layerIdx}(:, end);

        % synapses
        sigmaLayers{layerIdx}(:, :, 1) = sigmaLayers{layerIdx}(:, :, end);
        thetaLayers{layerIdx}(:, :, 1) = thetaLayers{layerIdx}(:, :, end);
    end
    % neurons
    epsilonLayers{numLayers}(:, 1) = epsilonLayers{numLayers}(:, end);
    phiLayers{numLayers}(:, 1) = phiLayers{numLayers}(:, end);

    % synapses
    sigmaLayers{numLayers}(:, :, 1) = sigmaLayers{numLayers}(:, :, end);
    priorMean(:, 1) = priorMean(:, end);
end

function hOut = h(phi)
    switch observationFunction
        case 'Linear'
            hOut = phi;
        case 'Quadratic'
            hOut = phi .^ 2;
        case 'Logarithmic'
            hOut = log(phi);
    end
end

function dhOut = dh(phi)
    switch observationFunction
        case 'Linear'
            dhOut = ones(length(phi), 1);
        case 'Quadratic'
            dhOut = 2 .* phi;
        case 'Logarithmic'
            dhOut = 1 ./ phi;
    end
end

end

