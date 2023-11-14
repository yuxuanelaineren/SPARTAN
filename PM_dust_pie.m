% Set directories
direc_public_PM10 = '/Users/renyuxuan/Desktop/Pie_dust/PM10';
direc_output = '/Users/renyuxuan/Desktop/Pie_dust/PM10_plots';

% Get a list of all CSV files in the folder
fileList = dir(fullfile(direc_public_PM10, '*_PM10_speciation.csv'));
numFiles = numel(fileList);

% Define the colors for each parameter
Colors = [
    240,103,166;    % pink for Aluminum PM2.5 (Al)
    109,207,246;    % light blue for Silicon PM2.5 (Si)
    245,126,32;     % orange for Calcium PM2.5 (Ca)
    237,48,41;      % red for Iron PM2.5 (Fe)
    35,31,32;       % black for Lead PM2.5 (Pb)
    252,238,30;     % yellow for Potassium PM2.5 (K)
    128,130,133;    % grey for Magnesium PM2.5 (Mg)
    57,84,165;      % blue for Sodium PM2.5 (Na)
    80,184,72;      % green for Zinc PM2.5 (Zn)
    200,50,120      % unknown for Manganese PM2.5 (Mn)
] ./ 255;

% Loop through each CSV file
for fileIdx = 1:numFiles
    % Read the CSV file skipping the first row
    filePath = fullfile(direc_public_PM10, fileList(fileIdx).name);
    data = readtable(filePath, 'HeaderLines', 1);
    
    % Extract the site name
    [~, fileName, ~] = fileparts(filePath);
    siteName = fileName(1:4);
    
    % Select rows from 2019 to 2023
    rows = (data.Start_Year_local >= 2019 & data.Start_Year_local <= 2023);
    
    % Apply transformations to the units
    ngIndices = ~strcmp(data.Units, 'Micrograms per cubic meter (ug/m3)');
    data.Value(ngIndices) = data.Value(ngIndices) * 10^-3;

    % Group values based on Parameter_Name and calculate average
    parameterNames = {'Aluminum PM10', 'Silicon PM10', 'Calcium PM10', ...
        'Iron PM10', 'Lead PM10', 'Potassium PM10', 'Magnesium PM10', ...
        'Sodium PM10', 'Zinc PM10', 'Manganese PM10'};
    
    pieData = zeros(size(parameterNames));
    negativeCounts = zeros(size(parameterNames));  % Initialize array to store negative value counts
    
    for paramIdx = 1:numel(parameterNames)
        parameter = parameterNames{paramIdx};
        values = data.Value(strcmp(data.Parameter_Name, parameter) & rows);
        
        % Count negative values
        numNegative = sum(values < 0);
        negativeCounts(paramIdx) = numNegative;  % Store the count in the array
        
        % Exclude NaN values
        values = values(~isnan(values));
        
        % Skip parameter if all values are NaN
        if isempty(values)
            continue;
        end
        
        % Replace negative values with zero
        values(values < 0) = 0;
        
        % Calculate average of non-negative values
        avgValue = mean(values);
        
        pieData(paramIdx) = avgValue;
    end
    
    % Write the average values and negative value counts to the output file
    outputFilePath = fullfile(direc_output, 'dust_pie_output.txt');
    outputFile = fopen(outputFilePath, 'a');  % Open the output file in append mode
    fprintf(outputFile, 'Site: %s\n', siteName);
    
    for paramIdx = 1:numel(parameterNames)
        parameter = parameterNames{paramIdx};
        avgValue = pieData(paramIdx);
        count = negativeCounts(paramIdx);
        fprintf(outputFile, 'Parameter: %s, Average Value: %.2f, Number of Negative Values: %d\n', parameter, avgValue, count);
    end
    
    fprintf(outputFile, '\n');
    fclose(outputFile);  % Close the output file
    
    % Generate pie chart without percent text and title for positive parameters
    positivePieData = pieData(pieData > 0);
    positiveParameterNames = parameterNames(pieData > 0);
    
    % Normalize the data to ensure a complete circle
    if any(positivePieData)
        positivePieData = positivePieData / sum(positivePieData);
        
        % Generate pie chart for positive parameters
        figure('Visible', 'off');
        h = pie(positivePieData);
        colormap(Colors);
        % c = colorbar('Location', 'eastoutside'); % Display a color bar in each pie chart
        % caxis([1, numel(positiveParameterNames)]);
        % ticks = 1:numel(positiveParameterNames);
        % tickLabels = positiveParameterNames;
        % c.Ticks = ticks;
        % c.TickLabels = tickLabels;

        % Remove the percent text labels
        textObjs = findobj(h, 'Type', 'text');
        delete(textObjs);

        % Save the pie chart without title as PNG
        saveFilePath = fullfile(direc_output, [siteName '.png']);
        saveas(gcf, saveFilePath);
        
        % Save the pie chart without title as EPS
        saveFilePath = fullfile(direc_output, [siteName '.eps']);
        print(gcf, saveFilePath, '-depsc', '-vector');
        
        close(gcf);
    end

end
