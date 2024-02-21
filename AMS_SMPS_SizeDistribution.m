close all; clear; clc;


% Set directories
in_dir = '/Volumes/rvmartin/Active/ren.yuxuan/AMS/SMPS/';
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/AMS/SMPS/';
date_of_file = '20240215';

% Read the data
file_path = fullfile(in_dir, ['SMPS_', date_of_file, '.xlsx']);
[num_data, txt_data, raw_data] = xlsread(file_path);

% Extract diameter data
diameter = num_data(:, 1);

%% plot individually
% Extract number concentration data for selected samples
start_sample = 25;
end_sample = 39;
selected_samples = num_data(:, start_sample+1:end_sample+1); % 6 and 21 for 2 g/L; 25 and 39 for 3 g/L; 44 and 46 for DIW

% Plot scatter plot for selected samples
figure('Color', 'white', 'Position', [100, 100, 800, 600]); % in pixels
hold on;
colors = colormap(jet(end_sample-start_sample+1));
marker_size = 10;
for i = 1:size(selected_samples, 2)
    fig = scatter(diameter, selected_samples(:, i), marker_size, 'filled', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', colors(i,:));
    % fig = plot(diameter, selected_samples(:, i), '-', 'Color', colors(i,:), 'LineWidth', 1.5);
end

% Customize plot
xlabel('Diameter (nm)');
ylabel('Number Concentration (#/cm^3)');
title('Size Distribution of 3 g/L NH_4NO_3', 'FontSize', 20, 'HorizontalAlignment', 'center', 'FontWeight', 'normal'); % NH_4NO_3
xlim([0, 550]);
xticks([0, 100, 200, 300, 400, 500]);
% yticks([1e6, 2e6, 3e6, 4e6]);
legend_str = cell(1, end_sample-start_sample+1);
for i = start_sample:end_sample
    legend_str{i-start_sample+1} = ['Sample ' num2str(i)];
end
ax = gca;
right_axis_position = ax.Position(1) + ax.Position(3);
legend(legend_str, 'Units', 'normalized', 'Position', [right_axis_position-0.2 0.5 0.1 0.1]); % -0.2 0.5 0.1 0.1
ax.Box = 'on';
ax.LineWidth = 1.5;
ax.TickLength = [0.010 0.010];
grid off;
hold off;

% Set all text
set(gca, 'FontName', 'Arial', 'FontSize', 20);


% Save plot
output_file_path = fullfile(out_dir, 'Size Distribution_NO4NO3.tiff');
saveas(fig, output_file_path, 'tiffn');

%% plot averages
% Extract number concentration data for each concentration
sample_indices = {[6, 21], [25, 39], [44, 46]};
concentrations = {'2 g/L', '3 g/L', 'DIW'};
colors = {'b', 'r', 'g'};

figure('Color', 'white', 'Position', [100, 100, 800, 600]);

for c = 1:numel(sample_indices)
    start_sample = sample_indices{c}(1);
    end_sample = sample_indices{c}(2);
    
    selected_samples = num_data(:, start_sample+1:end_sample+1);
    
    % Calculate average and standard error
    avg_concentration = mean(selected_samples, 2);
    stdev_concentration = std(selected_samples, 0, 2);
    n_concentration = size(selected_samples, 2);
    stderr_concentration = stdev_concentration / sqrt(n_concentration); % standard error (stdev / sqrt (n))
    
    % Plot average curve
    h(c) = plot(diameter, avg_concentration, 'Color', colors{c}, 'LineWidth', 1.5);
    hold on;
    
    % Plot shaded region for standard error
    x = [diameter; flipud(diameter)];
    y = [avg_concentration + stderr_concentration; flipud(avg_concentration - stderr_concentration)];
    patch(x, y, colors{c}, 'EdgeColor', 'none', 'FaceAlpha', 0.2, 'HandleVisibility', 'off');
    
end

% Customize plot
xlabel('Diameter (nm)');
ylabel('Number Concentration (#/cm^3)');
title('Size Distribution', 'FontSize', 20, 'FontWeight', 'normal');
legend(h);
xlim([0, 550]);
ylim([0, 14e5]);
xticks([0, 100, 200, 300, 400, 500]);
ax = gca;
right_axis_position = ax.Position(1) + ax.Position(3);
legend('2 g/L NH_4NO_3', '3 g/L NH_4NO_3', 'DIW', 'Units', 'normalized', 'Position', [right_axis_position-0.2 0.7 0.1 0.1]);
ax.Box = 'on';
ax.LineWidth = 1.5;
ax.TickLength = [0.010 0.010];
grid off;
hold off;

% Set all text properties
set(gca, 'FontName', 'Arial', 'FontSize', 20);

% Save plot
output_file_path = fullfile(out_dir, 'Size_Distribution_Averages_with_StdErr.tiff');
saveas(gcf, output_file_path, 'tiffn');

%% dN/dDp
% Clear workspace and command window
clear;
clc;

% Set directories
in_dir = '/Volumes/rvmartin/Active/ren.yuxuan/AMS/SMPS/';
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/AMS/SMPS/';
date_of_file = '20240215';

% Read the data
file_path = fullfile(in_dir, ['SMPS_', date_of_file, '.xlsx']);
[num_data, ~, ~] = xlsread(file_path);

% Extract diameter data
diameter = num_data(:, 1);

% Extract number concentration data for each concentration
sample_indices = {[6, 21], [25, 39], [44, 46]};
concentrations = {'2 g/L', '3 g/L', 'DIW'};
colors = {'b', 'r', 'g'};

figure('Color', 'white', 'Position', [100, 100, 800, 600]); % Set figure properties

for c = 1:numel(sample_indices)
    start_sample = sample_indices{c}(1);
    end_sample = sample_indices{c}(2);
    
    selected_samples = num_data(:, start_sample+1:end_sample+1);
    
    % Calculate average and standard error
    avg_concentration = mean(selected_samples, 2);
    
    % Divide concentration by log(diameter)
    conc_lgdiameter = avg_concentration ./ log10(diameter);
    
    % Plot conc/lgdiameter
    h(c) = plot(diameter, conc_lgdiameter, 'Color', colors{c}, 'LineWidth', 1.5, 'DisplayName', concentrations{c});
    hold on;
end

% Customize plot
xlabel('Diameter (nm)', 'FontSize', 16);
ylabel('dN / d(logDp) (#/cm^3)', 'FontSize', 16);
title('Size Distribution', 'FontSize', 20, 'FontWeight', 'normal');
legend(h);
ax = gca;
right_axis_position = ax.Position(1) + ax.Position(3);
legend('2 g/L NH_4NO_3', '3 g/L NH_4NO_3', 'DIW', 'Units', 'normalized', 'Position', [right_axis_position-0.2 0.7 0.1 0.1]);
ax.XScale = 'log'; % Set x-axis to log scale
ax.Box = 'on';
ax.LineWidth = 1.5;
ax.TickLength = [0.010 0.010];
grid off;

% Set x-axis tick labels
xticks([1, 10, 100, 1000]);
xticklabels({'1', '10', '100', '1000'});

% Set all text properties
set(gca, 'FontName', 'Arial', 'FontSize', 20);

% Save plot
output_file_path = fullfile(out_dir, 'Size_Distribution_dN_dDp_LogDiam.tiff');
saveas(gcf, output_file_path, 'tiffn');



