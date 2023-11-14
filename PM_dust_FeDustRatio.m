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

% Initialize arrays to store Fe/Al ratios and standard deviations
FeDustRatioPM25avg = zeros(1, numel(siteNames));
FeDustRatioPM10avg = zeros(1, numel(siteNames));
FeDustRatioPM25sd = zeros(1, numel(siteNames));
FeDustRatioPM10sd = zeros(1, numel(siteNames));

% Open the text file for writing
fid = fopen('/Users/renyuxuan/Desktop/Pie_dust/FeDust_ratio/FeDustRatios_updated.txt', 'w');

% Regional varying MAL and CF values:
MAL_CF={'AEAZ'	'AUMN'	'ARCB'	'BDDU'	'BIBU'	'CADO'	'CAHA'	'CAKE'	'CALE'	'CASH'	'CHTS'	'CLST'	'CODC'	'ETAD'	'IDBD'	'ILHA'	'ILNZ'	'INDH'	'INKA'	'KRSE'	'KRUL'	'MXMC'	'NGIL'	'PHMO'	'PRFJ'	'SGSU'	'TWKA'	'TWTA'	'USBA'	'USBO'	'USMC'	'USNO'	'USPA'	'VNHN'	'ZAJB'	'ZAPR';
          0.72 	0.24 	0.62 	0.62 	0.62 	0.62 	0.62 	0.62 	0.62 	0.62 	0.59 	0.62 	0.27 	0.62 	0.62 	0.72 	0.72 	0.62 	0.62 	0.59 	0.59 	0.27 	0.27 	0.62 	0.27 	0.62 	0.62 	0.62 	0.27 	0.27 	0.27 	0.27	0.66 	0.62 	0.62 	0.62 ;
          1.14 	1.05 	1.02 	1.02 	1.02 	1.02 	1.02 	1.02 	1.02 	1.02 	1.11 	1.02 	1.05 	1.02 	1.02 	1.14 	1.14 	1.02 	1.02 	1.11 	1.11 	1.05 	1.05 	1.02 	1.05 	1.02 	1.02 	1.02 	1.05 	1.05 	1.05 	1.05 	1.14 	1.02 	1.02 	1.02 };
    
% Loop through the site names
for i = 1:numel(siteNames)
    siteName = siteNames{i};
    siteLabel = siteLabels{i};

    % Read PM2.5 files
    PM25Files = dir(fullfile(PM25Folder, [siteName, '*_PM25_speciation.csv']));
    PM25RatioValues = {};
    PM25DustValues = {};

    % Process PM2.5 files
    for j = 1:numel(PM25Files)
        PM25File = PM25Files(j).name;
        PM25Data = readtable(fullfile(PM25Folder, PM25File), 'Delimiter', ',');

        % Apply transformations to the units
        ngIndices = ~strcmp(PM25Data.Units, 'Micrograms per cubic meter (ug/m3)');
        PM25Data.Value(ngIndices) = PM25Data.Value(ngIndices) * 10^-3;

        % Find the appropriate MAL and CF values for the current site
        siteIndex = find(strcmp(MAL_CF(1,:), siteName));
        MAL_value = MAL_CF{2, siteIndex};
        CF_value = MAL_CF{3, siteIndex};

        % Select rows from 2019 to 2023
        rows = PM25Data.Start_Year_local >= 2019 & PM25Data.Start_Year_local <= 2023;
        
        % Filter the data
        filteredData = PM25Data(rows, :);

        % Get unique dates (Year-Month-Date)
        uniqueDates = unique(datenum(filteredData{:, {'Start_Year_local', 'Start_Month_local', 'Start_Day_local'}}));

        % Count the number of unique dates
        numUniqueDates = numel(uniqueDates);
        disp(['Number of unique dates for PM2.5: ', num2str(numUniqueDates)]);

        % Calculate Dust and Fe/Dust ratio for each unique date
        for k = 1:numel(uniqueDates)
            currentDate = uniqueDates(k);
            dateRows = filteredData.Start_Year_local == year(currentDate) & ...
                filteredData.Start_Month_local == month(currentDate) & ...
                filteredData.Start_Day_local == day(currentDate);

        % Check if all required elements are available
        requiredElements = {'Aluminum PM2.5', 'Silicon PM2.5', 'Calcium PM2.5', 'Iron PM2.5', 'Titanium PM2.5'};
            if all(ismember(requiredElements, filteredData{dateRows, 'Parameter_Name'}))      
 
                % Increment the counter for each date that meets the requirements
                numUniqueDates = numUniqueDates + 1;

                % Calculate Dust using the provided equation
                AlValue = filteredData{dateRows & strcmp(filteredData.Parameter_Name, 'Aluminum PM2.5'), 'Value'};
                SiValue = filteredData{dateRows & strcmp(filteredData.Parameter_Name, 'Silicon PM2.5'), 'Value'};
                CaValue = filteredData{dateRows & strcmp(filteredData.Parameter_Name, 'Calcium PM2.5'), 'Value'};
                FeValue = filteredData{dateRows & strcmp(filteredData.Parameter_Name, 'Iron PM2.5'), 'Value'};
                TiValue = filteredData{dateRows & strcmp(filteredData.Parameter_Name, 'Titanium PM2.5'), 'Value'};
                   
                % Ensure scalar values for the variables
                AlValue = AlValue(1);
                SiValue = SiValue(1);
                CaValue = CaValue(1);
                FeValue = FeValue(1);
                TiValue = TiValue(1);

                Dust = ((1.98 * AlValue) * (1 + MAL_value) ...
                    + 2.14 * SiValue ...
                    + 1.40 * CaValue ...
                    + 1.36 * FeValue ...
                    + 1.67 * TiValue) * CF_value;
    
                % Calculate Fe/Dust ratio
                FeDustRatio = FeValue / Dust;
    
                % Store the Fe/Dust ratio in the array
                PM25RatioValues{end+1} = FeDustRatio;
                PM25DustValues{end+1} = Dust;
    
            else
                numDatesNotMeetingRequirements = numDatesNotMeetingRequirements + 1;
            end
        end

        % Print the number of unique dates that meet the requirements
        disp(['Number of unique dates with required elements: ', num2str(numUniqueDates)]);
        disp(['Number of unique dates not meeting requirements: ', num2str(numDatesNotMeetingRequirements)]);

    end

    % Calculate average and standard deviation of Fe/Dust ratio for PM2.5
    % Filter out empty cells and convert to a numeric array
    PM25RatioValuesNumeric = cellfun(@(x) double(x), PM25RatioValues, 'UniformOutput', false);
    PM25RatioValuesNumeric = cell2mat(PM25RatioValuesNumeric(~cellfun('isempty', PM25RatioValuesNumeric)));
    
    PM25DustValuesNumeric = cellfun(@(x) double(x), PM25DustValues, 'UniformOutput', false);
    PM25DustValuesNumeric = cell2mat(PM25DustValuesNumeric(~cellfun('isempty', PM25DustValuesNumeric)));
    
    avgFeDustRatioPM25 = mean(PM25RatioValuesNumeric);
    seFeDustRatioPM25 = std(PM25RatioValuesNumeric) / sqrt(length(PM25RatioValuesNumeric));
    avgDustPM25 = mean(PM25DustValuesNumeric);

    % Read PM10 files
    PM10Files = dir(fullfile(PM10Folder, [siteName, '*_PM10_speciation.csv']));
    PM10RatioValues = {};

    % Process PM10 files
    for j = 1:numel(PM10Files)
        PM10File = PM10Files(j).name;
        PM10Data = readtable(fullfile(PM10Folder, PM10File), 'Delimiter', ',');

        % Apply transformations to the units
        ngIndices = ~strcmp(PM10Data.Units, 'Micrograms per cubic meter (ug/m3)');
        PM10Data.Value(ngIndices) = PM10Data.Value(ngIndices) * 10^-3;

        % Find the appropriate MAL and CF values for the current site
        siteIndex = find(strcmp(MAL_CF(1,:), siteName));
        MAL_value = MAL_CF{2, siteIndex};
        CF_value = MAL_CF{3, siteIndex};

        % Select rows from 2019 to 2023
        rows = PM10Data.Start_Year_local >= 2019 & PM10Data.Start_Year_local <= 2023;
        
        % Filter the data
        filteredData = PM10Data(rows, :);

        % Get unique dates (Year-Month-Date)
        uniqueDates = unique(datenum(filteredData{:, {'Start_Year_local', 'Start_Month_local', 'Start_Day_local'}}));

        % Count the number of unique dates
        numUniqueDates = numel(uniqueDates);
        disp(['Number of unique dates for PM10: ', num2str(numUniqueDates)]);

        % Calculate Dust and Fe/Dust ratio for each unique date
        for k = 1:numel(uniqueDates)
            currentDate = uniqueDates(k);
            dateRows = filteredData.Start_Year_local == year(currentDate) & ...
                filteredData.Start_Month_local == month(currentDate) & ...
                filteredData.Start_Day_local == day(currentDate);

        % Check if all required elements are available
        requiredElements = {'Aluminum PM10', 'Silicon PM10', 'Calcium PM10', 'Iron PM10', 'Titanium PM10'};
            if all(ismember(requiredElements, filteredData{dateRows, 'Parameter_Name'}))      
 
                % Increment the counter for each date that meets the requirements
                numUniqueDates = numUniqueDates + 1;

                % Calculate Dust using the provided equation
                AlValue = filteredData{dateRows & strcmp(filteredData.Parameter_Name, 'Aluminum PM10'), 'Value'};
                SiValue = filteredData{dateRows & strcmp(filteredData.Parameter_Name, 'Silicon PM10'), 'Value'};
                CaValue = filteredData{dateRows & strcmp(filteredData.Parameter_Name, 'Calcium PM10'), 'Value'};
                FeValue = filteredData{dateRows & strcmp(filteredData.Parameter_Name, 'Iron PM10'), 'Value'};
                TiValue = filteredData{dateRows & strcmp(filteredData.Parameter_Name, 'Titanium PM10'), 'Value'};
                   
                % Ensure scalar values for the variables
                AlValue = AlValue(1);
                SiValue = SiValue(1);
                CaValue = CaValue(1);
                FeValue = FeValue(1);
                TiValue = TiValue(1);

                Dust = ((1.98 * AlValue) * (1 + MAL_value) ...
                    + 2.14 * SiValue ...
                    + 1.40 * CaValue ...
                    + 1.36 * FeValue ...
                    + 1.67 * TiValue) * CF_value;
    
                % Calculate Fe/Dust ratio
                FeDustRatio = FeValue / Dust;
    
                % Store the Fe/Dust ratio in the array
                PM10RatioValues{end+1} = FeDustRatio;
    
            else
                numDatesNotMeetingRequirements = numDatesNotMeetingRequirements + 1;
            end
        end

        % Print the number of unique dates that meet the requirements
        disp(['Number of unique dates with required elements: ', num2str(numUniqueDates)]);
        disp(['Number of unique dates not meeting requirements: ', num2str(numDatesNotMeetingRequirements)]);

    end

    % Calculate average and standard deviation of Fe/Dust ratio for PM10
    % Filter out empty cells and convert to a numeric array
    PM10RatioValuesNumeric = cellfun(@(x) double(x), PM10RatioValues, 'UniformOutput', false);
    PM10RatioValuesNumeric = cell2mat(PM10RatioValuesNumeric(~cellfun('isempty', PM10RatioValuesNumeric)));
    
    avgFeDustRatioPM10 = mean(PM10RatioValuesNumeric);
    seFeDustRatioPM10 = std(PM10RatioValuesNumeric) / sqrt(length(PM10RatioValuesNumeric));

    % Store the ratios
    FeDustRatioPM25avg(i) = avgFeDustRatioPM25;
    FeDustRatioPM25sd(i) = seFeDustRatioPM25;
    FeDustRatioPM10avg(i) = avgFeDustRatioPM10;
    FeDustRatioPM10sd(i) = seFeDustRatioPM10;

    % Write the results for the current site to the text file
    fprintf(fid, '%s\t%f\t%f\t%f\t%f\n', siteName, avgFeDustRatioPM25, seFeDustRatioPM25, avgFeDustRatioPM10, seFeDustRatioPM10, avgDustPM25);
end

% Close the text file
fclose(fid);

% Create a bar chart
figure;
FeDustRatios = [FeDustRatioPM25avg; FeDustRatioPM10avg]
hold on;
numBars = size(FeDustRatios, 2);
barCenters = 1:numBars;
barWidth = 1.0;

% Compute the offset for each bar
barOffset = (1 - barWidth) / 2;

% Plot the bars
h = bar(barCenters, FeDustRatios', barWidth, 'LineWidth', 1);
hold on;

% Shift the error bars to the right by 0.1 units
errorbarY = [FeDustRatioPM25avg; FeDustRatioPM25sd; FeDustRatioPM10avg; FeDustRatioPM10sd];
capLength = 3; % Length of the error bar caps

% Plot the error bars
for i = 1:numBars
    % Adjust the error bar positions based on bar width and offset
    errorbarX = barCenters(i) + barOffset;
    errorbar(errorbarX - 0.15, errorbarY(1, i), errorbarY(2, i), 'k', 'LineWidth', 1, 'CapSize', capLength);
    errorbar(errorbarX + 0.15, errorbarY(3, i), errorbarY(4, i), 'k', 'LineWidth', 1, 'CapSize', capLength);
end

set(gca, 'XTick', 1:numBars, 'XTickLabel', siteLabels, 'FontSize', 18, 'LineWidth', 1);
title('Fe Fraction of Dust for PM_{2.5} and PM_{10}', 'FontSize', 24, 'FontWeight', 'normal');
xlabel('Site', 'FontSize', 18);
ylabel('Fe Fraction of Dust', 'FontSize', 18);

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
ylim([0, 0.13]);

hold off;

% Set the figure size
figureSize = [15, 6]; % Width and height in inches
set(gcf, 'Units', 'inches', 'Position', [0, 0, figureSize]);

% Set the paper size
paperSize = [15, 6]; % Width and height in inches
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0, 0, paperSize]);

% Export the plot as JPEG
exportPath = '/Users/renyuxuan/Desktop/Pie_dust/FeDust_ratio/Fe_Dust_ratio_updated.jpg';
print(gcf, exportPath, '-djpeg', '-r300');
