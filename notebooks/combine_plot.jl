using DataFrames, CSV, Plots, Statistics

function combine_and_plot_data(input_dir, output_dir)
    # Read all CSV files in the input directory
    all_data = vcat([CSV.read(joinpath(input_dir, f), DataFrame) for f in readdir(input_dir) if endswith(f, "tols.csv")]...)

    # Define standard error function
    sem(x) = std(x) / sqrt(length(x))

    println(all_data)
    # Group by Tolerance and Algorithm, then calculate the average time, standard errors, and maximum errors
    grouped_data = combine(
        groupby(all_data, [:Tolerance, :Algorithm]),
        :Time => mean => :AvgTime,
        :Time => sem => :SEMTime,
        :L2Error => maximum => :MaxL2Error,
        :MaxDist => maximum => :MaxMaxDist,
        :MaxTrajDist => maximum => :MaxMaxTrajDist,
        :MaxTrajDist => mean => :MeanMaxTrajDist,
        :MaxTrajDist => sem => :SEMMeanMaxTrajDist
    )

    if !isdir(output_dir)
        mkdir(output_dir)
    end
    # Save combined data
    CSV.write(joinpath(output_dir, "solver_comparison_data_avg_time_max_error.csv"), grouped_data)
    println("combined data")
    # Create and save plots
    # Set plot attributes for publication quality
    default(; 
        legendtitle = nothing,
        tickfont = font(12),
        guidefont = font(14),
        legendfont = font(12),
        titlefont = font(16),
        markersize = 8,
        markerstrokewidth = 0.5,
        linewidth = 1,
        dpi = 300,
        size = (900, 600)
    )

    p1 = plot(
        grouped_data.MaxL2Error,
        grouped_data.AvgTime,
        group = grouped_data.Algorithm,
        xscale = :log10,
        yscale = :log10,
        seriestype=:path,
        xlabel = "Max L2 Error",
        ylabel = "Average Time (s)",
        #title = "Max L2 Error vs Average Time",
        legend = :topright,
        markershape = :auto
    )
    savefig(p1, joinpath(output_dir, "max_l2error_vs_avg_time.png"))

    p2 = plot(
        grouped_data.MaxMaxDist,
        grouped_data.AvgTime,
        group = grouped_data.Algorithm,
        xscale = :log10,
        yscale = :log10,
        seriestype=:path,
        xlabel = "Max Distance",
        ylabel = "Average Time (s)",
        #title = "Max Distance vs Average Time",
        legend = :topright,
        markershape = :auto,
        line = :solid
    )
    savefig(p2, joinpath(output_dir, "max_maxdist_vs_avg_time.png"))

    p3 = plot(
        grouped_data.MaxMaxTrajDist,
        grouped_data.AvgTime,
        group = grouped_data.Algorithm,
        xscale = :log10,
        yscale = :log10,
        seriestype=:path,
        xlabel = "Max Trajectory Distance",
        ylabel = "Average Time (s)",
        #title = "Max Trajectory Distance vs Average Time",
        legend = :topright,
        markershape = :auto,
        line = :solid
    )
    savefig(p3, joinpath(output_dir, "max_maxtrajdist_vs_avg_time.png"))

    # Plot with error bars for MeanMaxTrajDist vs AvgTime
    p4 = plot(
        grouped_data.MeanMaxTrajDist,
        grouped_data.AvgTime,
        xerror = grouped_data.SEMMeanMaxTrajDist,
        yerror = grouped_data.SEMTime,
        group = grouped_data.Algorithm,
        xscale = :log10,
        yscale = :log10,
        seriestype=:path,
        xlabel = "Mean Trajectory Distance",
        ylabel = "Average Time (s)",
        #title = "Mean Trajectory Distance vs Average Time",
        legend = :topright,
        line = :solid,
        markershape = :auto,
        grid = :on,
        framestyle = :box,
        markersize = 3  # Make points very small
    )
    savefig(p4, joinpath(output_dir, "mean_maxtrajdist_vs_avg_time.png"))

    println("Data has been combined and plots have been created with error bars.")
end

# Parse command-line arguments
input_dir = ARGS[1]
output_dir = ARGS[2]

# Run the function
combine_and_plot_data(input_dir, output_dir)