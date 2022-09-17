%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% only need to run this block %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transamp: signal amplitude with
% rows: driving signal as in powervector
% columns: frequency of driving signal as in transfreq 

figure(); imagesc(transamp)
% figure(); plot(transfreq, transamp(20,:))
% row = transamp(20,:);

% lorentzian 
% maximum of curve is 1 / (pi * c) 

lAmp = 1e6;
lPeak = 5.957e9;
lGamma = 1e5;
lOffset = -50;

lorentzian = fittype('-a / (c * (1 + ((x - b)/c)^2)) + d', ...
    'dependent', {'y'}, 'independent', {'x'}, ...
    'coefficients', {'a', 'b', 'c', 'd'});

% gaussian

gAmp = 20; % amplitude
gAmpLow = 10;
gAmpHigh = 30;

gPeak = 5.957e9; % resonance point, in index units
gPeakLow = 5.957e9;
gPeakHigh = 5.958e9;

gVar = 1e11; % ~variance, in square index units
gVarLow = 1e10;
gVarHigh = 1e12;

gOffset = -50; % vertical offset
gOffsetLow = -50;
gOffsetHigh = -45;

gaussian = fittype('-a * exp(-(x-b)^2/c) + d', ...
    'dependent', {'y'}, 'independent', {'x'}, ...
    'coefficients', {'a', 'b', 'c', 'd'});


n = length(powervector);
peakFreq = zeros(n, 4); % to be populated with peak frequency
window = 30; % window size for moving mean
interpStep = 2e3;
interpDomain = transfreq(1):interpStep:transfreq(length(transfreq));

for i=1:n
    row = transamp(i,:);
    
    % absolute 
    [~, posAbs] = min(row);
    peakFreq(i, 1) = transfreq(posAbs);
    
    % gauss 
    gFit = fit(transpose(transfreq), transpose(row), gaussian, ...
    'StartPoint', [gAmp, gPeak, gVar, gOffset], ...
    'Lower', [gAmpLow, gPeakLow, gVarLow, gOffsetLow], ...
    'Upper', [gAmpHigh, gPeakHigh, gVarHigh, gOffsetHigh]);
    peakFreq(i,2) = gFit.b;
    
    % lorentzian
    lFit = fit(transpose(transfreq), transpose(row), lorentzian, ...
    'StartPoint', [lAmp, lPeak, lGamma, lOffset]);
    peakFreq(i,3) = lFit.b;
    
    % moving window mean with spline interpolation
    movMean = movmean(row, window);
    interpRes = interp1(transfreq, movMean, interpDomain,'spline');
    [~, posMov] = min(interpRes);
    peakFreq(i,4) = interpDomain(posMov);
end


% format output

PEAKFREQ_ARR = horzcat(transpose(powervector), peakFreq);
header = {'power','absolute','gaussian','lorentzian', 'moving mean'};
PEAKFREQ_CELL = [header; num2cell(horzcat(transpose(powervector), peakFreq))]


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% display results and plots for single power %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% absolute minimum

[minAbs, posAbs] = min(transpose(row));
sprintf('abslute minimum %e Hz at index %d', transfreq(posAbs), posAbs)

%% display gaussian fit for one power

gFit = fit(transpose(transfreq), transpose(row), gaussian, ...
    'StartPoint', [gAmp, gPeak, gVar, gOffset], ...
    'Lower', [gAmpLow, gPeakLow, gVarLow, gOffsetLow], ...
    'Upper', [gAmpHigh, gPeakHigh, gVarHigh, gOffsetHigh]);

figure()
gResults = -gFit.a .* exp(-(transfreq - gFit.b).^2 ./ gFit.c) + gFit.d;
hold on
plot(transfreq, gResults)
plot(transfreq, row)
title('Gaussian')
hold off

sprintf('gaussian minimum %e Hz', gFit.b)

%% display lorentzian fit for one power

lFit = fit(transpose(transfreq), transpose(row), lorentzian, ...
    'StartPoint', [lAmp, lPeak, lGamma, lOffset]);

lResult = -lFit.a ./ ...
    (lFit.c .* (1 + ((transfreq - lFit.b)./lFit.c).^2)) + lFit.d;

figure()
hold on
plot(transfreq, lResult)
plot(transfreq, row)
title('Lorentzian')
hold off

sprintf('lorentzian minimum %e Hz', lFit.b)

%% moving window mean with spline interpolation

movMean = movmean(row, window);
interpRes = interp1(transfreq, movMean, interpDomain,'spline');
[minMov, posMov] = min(interpRes)

figure(4)
hold on
plot(transfreq, row, 'r')
plot(transfreq, movMean, 'g', 'linewidth', 1.5)
plot(interpDomain, interpRes, 'b')
legend('data', 'moving mean', 'interpolated moving mean')
title(['Moving mean with window' ' ' num2str(window)])
hold off

sprintf('moving mean minimum %d at %e', posMov, interpDomain(posMov))
