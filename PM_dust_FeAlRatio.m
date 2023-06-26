% Specify the folder paths
PM25Folder = '/Users/renyuxuan/Desktop/Pie_dust/PM25';
PM10Folder = '/Users/renyuxuan/Desktop/Pie_dust/PM10';
direc_output = '/Users/renyuxuan/Desktop/Pie_dust/FeAl_ratio';

% Specify the site names and their corresponding labels
siteNames = {'NGIL', 'ILNZ', 'AEAZ', 'PRFJ'};
siteLabels = {'Ilorin', 'Rehovot', 'Abu Dhabi', 'Fajardo'};

% Initialize arrays to store Fe/Al ratios and standard deviations
FeAlRatioPM25 = zeros(1, numel(siteNames));
FeAlRatioPM10 = zeros(1, numel(siteNames));
sdFeAlRatioPM25 = zeros(1, numel(siteNames));
sdFeAlRatioPM10 = zeros(1, numel(siteNames));

% Open the text file for writing
fid = fopen('/Users/renyuxuan/Desktop/Pie_dust/FeAl_ratio/FeAlRatios.txt', 'w');

% Loop through the site names
for i = 1:numel(siteNames)
    siteName = siteNames{i};
    siteLabel = siteLabels{i};

    % Read PM2.5 files
    PM25Files = dir(fullfile(PM25Folder, [siteName, '*_PM25_speciation.csv']));
    PM25RatioValues = [];

    % Process PM2.5 files
    for j = 1:numel(PM25Files)
        PM25File = PM25Files(j).name;
        PM25Data = readtable(fullfile(PM25Folder, PM25File), 'Delimiter', ',');

        % Select rows from 2019 to 2023
        rows = PM25Data.Start_Year_local >= 2019 & PM25Data.Start_Year_local <= 2023;

        % Filter the data
        filteredData = PM25Data(rows, :);
        FeData = filteredData(strcmp(filteredData.Parameter_Name, 'Iron PM2.5'), :);
        AlData = filteredData(strcmp(filteredData.Parameter_Name, 'Aluminum PM2.5'), :);

        % Calculate Fe/Al ratio
        FeAlRatio = FeData.Value ./ AlData.Value;
        PM25RatioValues = [PM25RatioValues; FeAlRatio];
    end

    % Calculate average and standard deviation of Fe/Al ratio for PM2.5
    avgFeAlRatioPM25 = mean(PM25RatioValues);
    sdFeAlRatioPM25(i) = std(PM25RatioValues);

    % Read PM10 files
    PM10Files = dir(fullfile(PM10Folder, [siteName, '*_PM10_speciation.csv']));
    PM10RatioValues = [];

    % Process PM10 files
    for j = 1:numel(PM10Files)
        PM10File = PM10Files(j).name;
        PM10Data = readtable(fullfile(PM10Folder, PM10File), 'Delimiter', ',');

        % Select rows from 2019 to 2023
        rows = PM10Data.Start_Year_local >= 2019 & PM10Data.Start_Year_local <= 2023;

        % Filter the data
        filteredData = PM10Data(rows, :);
        FeData = filteredData(strcmp(filteredData.Parameter_Name, 'Iron PM10'), :);
        AlData = filteredData(strcmp(filteredData.Parameter_Name, 'Aluminum PM10'), :);

        % Calculate Fe/Al ratio
        FeAlRatio = FeData.Value ./ AlData.Value;
        PM10RatioValues = [PM10RatioValues; FeAlRatio];
    end

    % Calculate average and standard deviation of Fe/Al ratio for PM10
    avgFeAlRatioPM10 = mean(PM10RatioValues);
    sdFeAlRatioPM10(i) = std(PM10RatioValues);

    % Store the ratios
    FeAlRatioPM25(i) = avgFeAlRatioPM25;
    FeAlRatioPM10(i) = avgFeAlRatioPM10;

    % Write the results for the current site to the text file
    fprintf(fid, '%s\t%f\t%f\t%f\t%f\n', siteLabel, avgFeAlRatioPM25, sdFeAlRatioPM25(i), avgFeAlRatioPM10, sdFeAlRatioPM10(i));
end

% Close the text file
fclose(fid);

% Create a bar chart
figure;
h = bar(FeAlRatios', 'LineWidth', 1);
hold on;
numBars = size(FeAlRatios, 2);
barCenters = 1:numBars;
barWidth = 0.4;

errorbarY = [FeAlRatioPM25; sdFeAlRatioPM25; FeAlRatioPM10; sdFeAlRatioPM10];
errorbarX = h(1).XData + h(1).XOffset;

for i = 1:numBars
    errorbar(errorbarX(i), errorbarY(1, i), errorbarY(2, i), 'k', 'LineWidth', 1);
    errorbar(errorbarX(i) + 0.3, errorbarY(3, i), errorbarY(4, i), 'k', 'LineWidth', 1);
end

set(gca, 'XTick', 1:numBars, 'XTickLabel', siteLabels, 'FontSize', 18, 'LineWidth', 1);
title('Fe/Al Ratio for PM_{2.5} and PM_{10}', 'FontSize', 24, 'FontWeight', 'normal');
xlabel('Site', 'FontSize', 18);
ylabel('Fe/Al Ratio', 'FontSize', 18);

% Define the custom position for the legend
legendPosition = [0.7, 0.78, 0.2, 0.1]; % [x, y, width, height]

% Modify the legend position
legend('PM_{2.5}', 'PM_{10}', 'FontSize', 18, 'LineWidth', 0.5, 'EdgeColor', 'none');
legend('Location', 'none', 'Position', legendPosition);

% Remove ticks from the top, bottom, and right side
ax = gca;
% ax.TickDir = 'out'; % tick direction
ax.XAxis.TickLength = [0, 0];
ax.YAxis.TickLength = [0, 0]; % both left ticks and right ticks are regarded as ax.YAxis(1)
% ax.YAxis(1).TickLength = [0, 0];  % Hide left ticks

hold off;

% Set the figure size
figureSize = [10, 8]; % Width and height in inches
set(gcf, 'Units', 'inches', 'Position', [0, 0, figureSize]);

% Set the paper size
paperSize = [10, 8]; % Width and height in inches
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0, 0, paperSize]);

% Export the plot as JPEG
exportPath = '/Users/renyuxuan/Desktop/Pie_dust/FeAl_ratio/Fe_Al_ratio.jpg';
print(gcf, exportPath, '-djpeg', '-r300');