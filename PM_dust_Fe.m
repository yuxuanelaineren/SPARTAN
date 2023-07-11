% Specify the folder paths
PM25Folder = '/Users/renyuxuan/Desktop/Pie_dust/PM25';
PM10Folder = '/Users/renyuxuan/Desktop/Pie_dust/PM10';
direc_output = '/Users/renyuxuan/Desktop/Pie_dust/FeDust_ratio';

% Specify the site names and their corresponding labels
siteNames = {'AEAZ', 'NGIL', 'INKA', 'BDDU', 'CHTS', 'ILNZ', 'VNHN',...
             'ILHA', 'INDH', 'PRFJ', 'KRSE', 'TWKA', 'IDBD',...
             'ZAPR', 'KRUL', 'SGSU', 'ZAJB', 'MXMC', 'TWTA',...
             'USPA', 'CALE', 'CADO', 'CASH', 'AUMN'};
siteLabels = {'Masdar', 'Ilorin', 'Kanpur', 'Dhaka', 'Beijing', 'Rehovot', 'Hanoi',...
              'Haifa', 'Delhi', 'Fajardo', 'Seoul', 'Kaohsiung','Bandung',...
              'Pretoria', 'Ulsan', 'Singapore', 'Johannesburg', 'Mexico City', 'Taipei',...
              'Pasadena', 'Lethbridge', 'Downsview', 'Sherbrooke', 'Melbourne'};

% Initialize arrays to store Fe concentrations
FePM25avg = zeros(1, numel(siteNames));
FePM10avg = zeros(1, numel(siteNames));
FePM25se = zeros(1, numel(siteNames));
FePM10se = zeros(1, numel(siteNames));

% Open the text file for writing
fid = fopen('/Users/renyuxuan/Desktop/Pie_dust/FeDust_ratio/Fe.txt', 'w');

% Loop through the site names
for i = 1:numel(siteNames)
    siteName = siteNames{i};
    siteLabel = siteLabels{i};

    % Read PM2.5 files
    PM25Files = dir(fullfile(PM25Folder, [siteName, '*_PM25_speciation.csv']));
    PM25FeValues = [];
    
    % Process PM2.5 files
    for j = 1:numel(PM25Files)
        PM25File = PM25Files(j).name;
        PM25Data = readtable(fullfile(PM25Folder, PM25File), 'Delimiter', ',');
        
        % Apply transformations to the units
        ngIndices = ~strcmp(PM25Data.Units, 'Micrograms per cubic meter (ug/m3)');
        PM25Data.Value(ngIndices) = PM25Data.Value(ngIndices) * 10^-3;
        
        % Select rows from 2019 to 2023
        rows = PM25Data.Start_Year_local >= 2019 & PM25Data.Start_Year_local <= 2023;
        
        % Filter the data
        filteredData = PM25Data(rows, :);
        
        % Calculate Fe concentration
        FeValues = filteredData{strcmp(filteredData.Parameter_Name, 'Iron PM2.5'), 'Value'};
        
        % Store the Fe values in the array
        PM25FeValues = [PM25FeValues; FeValues];
    end
    
    % Calculate average and standard deviation of Fe for PM2.5 in each site
    FePM25avg(i) = mean(PM25FeValues, 'omitnan'); % any NaN values in the data arrays will be ignored
    FePM25se(i) = std(PM25FeValues, 'omitnan') / sqrt(sum(~isnan(PM25FeValues)));

    % Read PM10 files
    PM10Files = dir(fullfile(PM10Folder, [siteName, '*_PM10_speciation.csv']));
    PM10FeValues = [];
    
    % Process PM10 files
    for j = 1:numel(PM10Files)
        PM10File = PM10Files(j).name;
        PM10Data = readtable(fullfile(PM10Folder, PM10File), 'Delimiter', ',');
        
        % Apply transformations to the units
        ngIndices = ~strcmp(PM10Data.Units, 'Micrograms per cubic meter (ug/m3)');
        PM10Data.Value(ngIndices) = PM10Data.Value(ngIndices) * 10^-3;
        
        % Select rows from 2019 to 2023
        rows = PM10Data.Start_Year_local >= 2019 & PM10Data.Start_Year_local <= 2023;
        
        % Filter the data
        filteredData = PM10Data(rows, :);
        
        % Calculate Fe concentration
        FeValues = filteredData{strcmp(filteredData.Parameter_Name, 'Iron PM10'), 'Value'};
        
        % Store the Fe values in the array
        PM10FeValues = [PM10FeValues; FeValues];
    end
    
    % Calculate average and standard deviation of Fe for PM10 in each site
    FePM10avg(i) = mean(PM10FeValues, 'omitnan');
    FePM10se(i) = std(PM10FeValues, 'omitnan') / sqrt(sum(~isnan(PM10FeValues)));

    % Write the results for the current site to the text file
    fprintf(fid, '%s\t%f\t%f\t%f\t%f\t%d\n', siteLabel, FePM25avg(i), FePM25se(i), sum(~isnan(PM25FeValues)), FePM10avg(i), FePM10se(i), sum(~isnan(PM10FeValues)));

end

% Close the text file
fclose(fid);

% Create a bar chart
figure;
FeConcentrations = [FePM25avg; FePM10avg];
hold on;
numBars = size(FeConcentrations, 2);
barCenters = 1:numBars;
barWidth = 1.0;

% Compute the offset for each bar
barOffset = (1 - barWidth) / 2;

% Plot the bars
h = bar(barCenters, FeConcentrations', barWidth, 'LineWidth', 1);
hold on;

% Shift the error bars to the right by 0.1 units
errorbarY = [FePM25avg; FePM25se; FePM10avg; FePM10se];
capLength = 4; % Length of the error bar caps

% Plot the error bars
for i = 1:numBars
    % Adjust the error bar positions based on bar width and offset
    errorbarX = barCenters(i) + barOffset;
    errorbar(errorbarX - 0.15, errorbarY(1, i), errorbarY(2, i), 'k', 'LineWidth', 1, 'CapSize', capLength);
    errorbar(errorbarX + 0.15, errorbarY(3, i), errorbarY(4, i), 'k', 'LineWidth', 1, 'CapSize', capLength);
end

set(gca, 'XTick', 1:numBars, 'XTickLabel', siteLabels, 'FontSize', 18, 'LineWidth', 1);
title('Fe Concentration for PM_{2.5} and PM_{10}', 'FontSize', 24, 'FontWeight', 'normal');
xlabel('Site', 'FontSize', 18);
ylabel('Concentration (μg/m^3)', 'FontSize', 18);

% Define the custom position for the legend
legendPosition = [0.73, 0.802, 0.15, 0.03]; % [x, y, width, height]

% Modify the legend position
legend('PM_{2.5}', 'PM_{10}', 'FontSize', 18, 'LineWidth', 0.5, 'EdgeColor', 'none');
legend('Location', 'none', 'Position', legendPosition);

% Remove ticks from the top, bottom, and right side
ax = gca;
ax.Box = 'on'; % Display the border lines
% ax.TickDir = 'out'; % tick direction
ax.XAxis.TickLength = [0, 0];
ax.YAxis.TickLength = [0, 0]; % both left ticks and right ticks are regarded as ax.YAxis(1)
% ax.YAxis(1).TickLength = [0, 0];  % Hide left ticks

% Set a fixed length for the y-axis
ylim([0, 6.5]);

hold off;

% Set the figure size
figureSize = [15, 6]; % Width and height in inches
set(gcf, 'Units', 'inches', 'Position', [0, 0, figureSize]);

% Set the paper size
paperSize = [15, 6]; % Width and height in inches
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0, 0, paperSize]);

% Export the plot as JPEG
exportPath = '/Users/renyuxuan/Desktop/Pie_dust/FeDust_ratio/Fe.jpg';
print(gcf, exportPath, '-djpeg', '-r300');
