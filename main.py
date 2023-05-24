import argparse
import time

import matplotlib.pyplot as plt

import pca
import fdfs as sa
import file_handler as fr
import helper_funcs as hf
import numpy as np

import warnings

# import test_funcs as tf

default_filters = ["uniq", "flat", "spike", "fft"]
print_modes = ["print", "file", "none"]

default_time_window = (0.210, 0.50)


def arg_parser():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="analyse a meg dataset and"
                                                 "find faulty and/or unphysical"
                                                 "measurements")

    parser.add_argument("--filename", required=True, type=str, help="filename of the dataset to analyse")
    parser.add_argument("-m", "--mode", required=True, type=int,
                        help="the mode for the program. 1 -> analyse entire time window, "
                             "show bad segments; 2 -> analyse partial window , "
                             "show good segments (requires -t/--time)")
    parser.add_argument("-t", "--time", type=float, nargs=2, default=default_time_window,
                        help="time window for mode 2 in "
                             "seconds, default (0.210, 0.50)")
    parser.add_argument('--filters', nargs='+', choices=default_filters,
                        default=default_filters, help="the basic FDFs to use")
    parser.add_argument("-p", "--physicality", action="store_true", default=False,
                        help="do consistency analysis")
    parser.add_argument("-o", "--output", type=str, default="output_test.txt", help="file to output results to")
    parser.add_argument("--plot", action="store_true", default=False, help="plot signals with results")
    parser.add_argument("-prnt", "--print_mode", default="print", choices=print_modes)
    parser.add_argument("-log", "--log_filename", default="")

    args = parser.parse_args()

    return args


def thirdver(fname, filters, phys, printer, channels=["MEG*1", "MEG*4"]):
    """run the default mode of the program (mode 1). outputs bad and suspicious segments as well as results of CA.

    parameters:
    fname: name of file to analyse
    filters: list of all FDFs to use. strings
    phys: boolean. if true, CA is run
    channels: names of channels to analyse. allows wildcard characters.
    printer: printer object. see file_handler.py

    returns:
    col_names: column names for the results.
    write_data: list of lists containing all the results. for example, write_data[i] contains the list with results for column with the name in col_names[i]
    plot_data: list of lists with data for plot_in_order_ver3"""

    printer.extended_write("analysing " + fname, additional_mode="print")
    printer.extended_write("", additional_mode="print")

    signals, names, t, n_chan = fr.get_signals(fname, channels=channels)
    printer.extended_write("filtering with the following FDFs:", filters, additional_mode="print")
    printer.extended_write("", additional_mode="print")
    start_time = time.time()
    signal_statuses, bad_segs, suspicious_segs, exec_times = sa.analyse_all_neo(signals, names, n_chan, printer,
                                                                                filters=filters, fft_goertzel=False)

    num_bads, frac_bads = hf.frac_of_sigs(bad_segs)

    end_time = time.time()
    filt_time = (end_time - start_time)
    printer.extended_write("time elapsed in filtering: " + str(filt_time) + " secs", additional_mode="print")
    printer.extended_write("", additional_mode="print")
    printer.extended_write("bad segments present in ", 100 * frac_bads, "% of the signals", additional_mode="print")

    if phys:
        printer.extended_write("-----------------------------------------------------", additional_mode="print")
        printer.extended_write("consistency analysis", additional_mode="print")
        printer.extended_write("", additional_mode="print")
        detecs = np.load("array120_trans_newnames.npz")
        start_time = time.time()
        # ave_sens = 10**(-12)
        all_diffs, all_rel_diffs, chan_dict = pca.check_all_phys(signals, detecs, names, n_chan, bad_segs,
                                                                 suspicious_segs, printer,
                                                                 ave_window=100, ave_sens=5 * 10 ** (-13))

        phys_stat, phys_conf = pca.analyse_phys_dat_alt(all_diffs, names, all_rel_diffs, chan_dict)
        num_const = phys_stat.count(0)
        num_unconst = phys_stat.count(1)
        num_und = phys_stat.count(2)
        num_unus = phys_stat.count(3)
        # print(phys_stat)

        frac_const_all = num_const / n_chan
        frac_unconst_all = num_unconst / n_chan
        frac_und_all = num_und / n_chan
        frac_unus_all = num_unus / n_chan

        n_analysed = n_chan - num_bads

        frac_const_anal = num_const / n_analysed
        frac_unconst_anal = num_unconst / n_analysed
        frac_und_anal = num_und / n_analysed
        frac_unus_anal = (num_unus - num_bads) / n_analysed

        end_time = time.time()
        phys_time = (end_time - start_time)
        printer.extended_write("time elapsed in physicality analysis: " + str(phys_time) + " secs",
                               additional_mode="print")
        printer.extended_write("", additional_mode="print")
        printer.extended_write("per cent of all signals", additional_mode="print")
        printer.extended_write("consistent:", 100 * frac_const_all, "%", additional_mode="print")
        printer.extended_write("inconsistent:", 100 * frac_unconst_all, "%", additional_mode="print")
        printer.extended_write("undetermined:", 100 * frac_und_all, "%", additional_mode="print")
        printer.extended_write("unused:", 100 * frac_unus_all, "%", additional_mode="print")
        printer.extended_write("", additional_mode="print")
        printer.extended_write("per cent of all analysed signals", additional_mode="print")
        printer.extended_write("consistent:", 100 * frac_const_anal, "%", additional_mode="print")
        printer.extended_write("inconsistent:", 100 * frac_unconst_anal, "%", additional_mode="print")
        printer.extended_write("undetermined:", 100 * frac_und_anal, "%", additional_mode="print")
        printer.extended_write("unused:", 100 * frac_unus_anal, "%", additional_mode="print")
        printer.extended_write("", additional_mode="print")
    else:
        phys_stat = []
        phys_conf = []
        all_rel_diffs = []
        chan_dict = []
        phys_time = 0

    tot_time = phys_time + filt_time
    printer.extended_write("-----------------------------------------------------", additional_mode="print")
    printer.extended_write("", additional_mode="print")

    i_x = list(range(len(t)))
    bad_segs_time = hf.segs_from_i_to_time(i_x, t, bad_segs)
    sus_segs_time = hf.segs_from_i_to_time(i_x, t, suspicious_segs)
    col_names = ["name", "bad segments", "suspicious segments"]
    write_data = [names, bad_segs_time, sus_segs_time]

    if phys:
        write_data.append(phys_stat)
        write_data.append(phys_conf)
        col_names.append("pca status")
        col_names.append("pca fraction")

    printer.extended_write("total time elapsed: " + str(tot_time) + " secs", additional_mode="print")

    printer.close()

    plot_dat = [signals, names, n_chan, signal_statuses, bad_segs, suspicious_segs, phys_stat, t]

    return col_names, write_data, plot_dat


def partial_analysis(time_seg, fname, printer, output="output_test.txt",
                     channels=["MEG*1", "MEG*4"],
                     filters=default_filters, seg_extend=400, phys=False, plot=False):
    """run partial analysis (mode 2). analyses a segment of all signals slightly larger than the inputted time segment,
     returns nothing but writes results to file.

    parameters:
    time_seg: tuple of floats. time segment (in seconds) to analyse
    fname: filename to analyse
    printer: printer object, see file_handler.py
    output: string. the output filename
    channels: the channels to analyse. allows wildcard characters
    filters: the FDFs to use
    seg_extend: int. determines how much the time segment is extended at each end (in data points)
    phys: boolean. if true, CA is run
    plot: boolean. if true, results are plotted"""
    signals, names, t, n_chan = fr.get_signals(fname, channels=channels)

    printer.extended_write("analysing time window " + str(time_seg) + " secs from file " + fname,
                           additional_mode="print")
    printer.extended_write("", additional_mode="print")

    printer.extended_write("filtering with the following FDFs:", filters, additional_mode="print")
    printer.extended_write("", additional_mode="print")
    start_time = time.time()
    cropped_signals, cropped_ix, seg_i = hf.crop_signals_time(time_seg, t, signals, seg_extend)
    signal_statuses, bad_segs, suspicious_segs, exec_times = sa.analyse_all_neo(cropped_signals, names, n_chan, printer,
                                                                                filters=filters, filter_beginning=False)
    end_time = time.time()
    filt_time = end_time - start_time
    printer.extended_write("time elapsed in filtering: " + str(filt_time) + " secs", additional_mode="print")
    printer.extended_write("", additional_mode="print")

    good_seg_list = hf.find_good_segs(seg_i, bad_segs, cropped_ix)

    num_bads, frac_bads = hf.frac_of_sigs(bad_segs)
    num_goods, frac_goods = hf.frac_of_sigs(good_seg_list)
    printer.extended_write("bad segments present in ", 100 * frac_bads, "% of the signals", additional_mode="print")
    printer.extended_write("good segments present in ", 100 * frac_goods, "% of the signals", additional_mode="print")

    if phys:
        detecs = np.load("array120_trans_newnames.npz")
        printer.extended_write("-----------------------------------------------------", additional_mode="print")
        printer.extended_write("beginning physicality analysis", additional_mode="print")
        printer.extended_write("", additional_mode="print")
        start_time = time.time()
        all_diffs, all_rel_diffs, chan_dict = pca.check_all_phys(cropped_signals, detecs, names, n_chan, bad_segs,
                                                                 suspicious_segs, printer,
                                                                 ave_window=100, ave_sens=5 * 10 ** (-13))
        end_time = time.time()
        phys_time = end_time - start_time
        printer.extended_write()
        printer.extended_write("time elapsed in physicality analysis: " + str(phys_time) + " secs",
                               additional_mode="print")

        phys_stat, phys_conf = pca.analyse_phys_dat_alt(all_diffs, names, all_rel_diffs, chan_dict)
    else:
        phys_time = 0

    tot_time = phys_time + filt_time
    printer.extended_write("-----------------------------------------------------", additional_mode="print")
    printer.extended_write("", additional_mode="print")

    good_segs_time = hf.segs_from_i_to_time(cropped_ix, t, good_seg_list)

    col_names = ["name", "good segments"]
    write_data = [names, good_segs_time]

    if phys:
        write_data.append(phys_stat)
        write_data.append(phys_conf)
        col_names.append("pca status")
        col_names.append("pca fraction")

    if output is not None:
        fr.write_data_compact(output, write_data, col_names)

    printer.extended_write("total time elapsed: " + str(tot_time) + " secs", additional_mode="print")
    printer.close()

    if plot:
        bad_segs_time = hf.segs_from_i_to_time(cropped_ix, t, bad_segs)
        sus_segs_time = hf.segs_from_i_to_time(cropped_ix, t, suspicious_segs)
        for i in range(n_chan):
            i_x = cropped_ix[i]
            t_x = t[i_x]
            name = names[i]
            bad_seg_plot = bad_segs_time[i]
            sus_segs_plot = sus_segs_time[i]
            good_segs_plot = good_segs_time[i]
            cropped_sig = cropped_signals[i]
            p_stat = phys_stat[i]
            p_conf = phys_conf[i]
            figure, ax = plt.subplots()
            plt.plot(t_x, cropped_sig)
            hf.plot_spans(ax, bad_seg_plot, color="red")
            hf.plot_spans(ax, sus_segs_plot, color="yellow")
            hf.plot_spans(ax, good_segs_plot, color="green")

            ax.axvline(t[seg_i[0]], linestyle="--", color="black")
            ax.axvline(t[seg_i[-1]], linestyle="--", color="black")

            status = name + ", " + str(good_segs_plot)

            if phys:
                status += ", " + str(p_stat) + ", " + str(p_conf)

            ax.set_title(status)
            plt.show()


def orig_main(args, printer):
    """wrapper for thirdver. runs thirdver, writes data to file and plots data if required.

    parameters:
    args: command line arguments
    printer: printer object. see file_handler.py"""
    col_names, data, plot_dat = thirdver(args.filename, args.filters, args.physicality, printer)

    signals, names, n_chan, signal_statuses, bad_segs, suspicious_segs, phys_stat, t = plot_dat

    fr.write_data_compact("output_test.txt", data, col_names)

    if args.plot:
        hf.plot_in_order_ver3(signals, names, n_chan, signal_statuses, bad_segs, suspicious_segs, physicality=phys_stat,
                              time_x=t)


def main():
    """main program. reads command line arguments and runs the required mode."""
    args = arg_parser()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if args.print_mode == "file":
        if args.log_filename == "":
            logfname = "log.log"
        else:
            logfname = args.log_filename

    else:
        logfname = ""

    if args.print_mode == "file":
        printer = fr.Printer("file", open(logfname, "w"))
    else:
        printer = fr.Printer(args.print_mode)

    printer.extended_write("mode:", args.mode, additional_mode="print")

    if args.mode == 1:
        orig_main(args, printer)

    if args.mode == 2:
        if args.time == default_time_window:
            printer.extended_write("no time window given, analysing default window", default_time_window,
                                   additional_mode="print")

        partial_analysis(args.time, args.filename, printer, args.output, filters=args.filters, phys=args.physicality,
                         plot=args.plot)


if __name__ == '__main__':
    main()
    # tf.test_fft()
    # tf.show_helmet()
    # tf.test_fft_emergency()
    # tf.show()
    # tf.test_new_excluder()
    # tf.test_magn2()
    # tf.test_seg_finder()
    # tf.test_crop()
    # tf.test_ffft()
    # tf.show_pca()
    # tf.test_flat_new()
    # datadir = "example_data_for_patrik/"
    # partial_analysis([0.46, 0.46], datadir + "sample_data38.npz", "print", phys=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
