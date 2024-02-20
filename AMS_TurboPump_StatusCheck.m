close all; clear; clc;

%% Code description
% Description: Reads turbo pump status and plots various parameters 
% (frequency, current, temperature) over time.
%
% Written by: Yuxuan Ren
% Created: February 03, 2024
%
% Version History:
% - February 03, 2024: Initial release.

%% User setting

% Set directories
in_dir = '/Volumes/rvmartin/Active/ren.yuxuan/AMS/Turbo_Pump_Status/'; % for Yuxuan
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/AMS/Turbo_Pump_Status/Issues/'; % for Yuxuan
% Specify date of the analyzed file and parameter to be plotted
date_of_file = '20240206';
parameter = 'all'; % 'all', 'frequency', 'current', or 'temperature'

%% Read data
% Read the data
file_path = fullfile(in_dir, ['PumpData_', date_of_file, '.txt']);
pump_data = readtable(file_path);

% Extract date and time
date_str = string(pump_data{:, 1}) + " " + string(pump_data{:, 2});
date_time = datetime(date_str, 'InputFormat', 'MM/dd/uuuu HH:mm:ss');

%% Plotting
if strcmp(parameter, 'all')
    % Plotting with three subplots for frequency, current, and temperature
    fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
    for i = 1:3
        subplot(3, 1, i);
        plot_single_parameter(i, date_time, pump_data);
    end
else
    % Plotting with a single subplot for the specified parameter
    fig = figure('Color', 'white', 'Position', [100, 100, 800, 400]);
    plot_single_parameter(find(strcmp(parameter, {'frequency', 'current', 'temperature'})), date_time, pump_data);
end

sgtitle('Turbo Pump Status', 'FontSize', 20, 'HorizontalAlignment', 'center');

%% Save plot
output_file_path = fullfile(out_dir, ['Turbo_Pump_Status_', date_of_file,'_' parameter, '.tiff']);
saveas(fig, output_file_path, 'tiffn');

%% Function to plot a single parameter
function plot_single_parameter(param_index, date_time, pump_data)
    % Define parameters and corresponding data subsets
    parameters = {'frequency', 'current', 'temperature'};
    data_columns = {9:5:29, 10:5:30, 11:5:31};
    
    parameter = parameters{param_index};
    data_subset = pump_data(:, data_columns{param_index});
    
    % Define axis labels and limits
    switch lower(parameter)
        case 'frequency'
            ylabel_text = 'Frequency (Hz)';
            ylim_range = [0, 1500];
        case 'current'
            ylabel_text = 'Current (mA)';
            ylim_range = [0, 500];
        case 'temperature'
            ylabel_text = 'Temperature (C)';
            ylim_range = [10, 40];
        otherwise
            error('Invalid parameter');
    end
    
    % Plot the data
    scatter(date_time, table2array(data_subset), 10, 'filled');
    xlabel('Date and Time');
    ylabel(ylabel_text);

    ax = gca;
    ax.Position(1) = ax.Position(1) * 0.7; % Decrease the position of the left edge of the axes
    ax.Position(4) = ax.Position(4) * 0.95; % Reduce the height of the axes by 5%
   
    % Add legend
    lgd = legend('Pump 2', 'Pump 3', 'Pump 4', 'Pump 5', 'Pump 6', 'Location', 'eastoutside');
    lgd.Position(1) = ax.Position(1) + ax.Position(3) + 0.02;
    
    % Set Y-axis limit based on parameter
    ylim(ylim_range);
    
    % Set axis properties
    set(gca, 'FontName', 'Arial', 'FontSize', 16, 'box', 'on', 'linewidth', 1);
    box on;
    ax.TickDir = 'none';
    ax.TickLength = [0.02 0.02];
end
