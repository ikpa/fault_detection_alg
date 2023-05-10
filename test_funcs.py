import time

#import helmet_vis as vis
import numpy as np

import fdfs
import helper_funcs
import helper_funcs as hf
import file_handler as fr
import pca
import fdfs as sa
import matplotlib.pyplot as plt
import helmet_vis as vis

# from sklearn import preprocessing

datadir = "example_data_for_patrik/"

brokens = ["MEG0111", "MEG0221", "MEG0234", "MEG0241", "MEG1244", "MEG2131", "MEG2244"]


# various test functions. not commented

def animate_vectors():
    fname = "many_successful.npz"
    # fname = "sample_data38.npz"
    signals, names, time, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    names, signals = hf.order_lists(detecs, names, signals)
    print(type(signals))

    ffts = hf.calc_fft_all(signals)
    # print(np.shape(signals))
    # print(np.shape(ffts))
    # print(type(ffts))
    # vis.helmet_animation(names, ffts, 1000, cmap="Purples", vlims=[0, 1 * 10 ** (-7)]

def animate_fft():
    def animate(i):
        if i + window > max_i:
            end_i = max_i
        else:
            end_i = i + window

        signal_windowed = signal[i: end_i]
        ftrans = sa.get_fft(signal_windowed)
        i1.append(ftrans[1])
        i2.append(ftrans[2])
        i6.append(ftrans[6])

        if i % 20 == 0:
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax2.set_ylim(-1 * 10 ** (-7), 1.5 * 10 ** (-6))
            ax1.plot(plot_time, signal_windowed)
            ax2.plot(ftrans[0:10], ".-")
            ax3.plot(i1, label="index 1")
            ax3.plot(i2, label="index 2")
            ax3.plot(i6, label="index 6")
            ax3.legend()

    from matplotlib.animation import FuncAnimation
    fname = "sample_data30.npz"
    channels = ["MEG0121", "MEG1114"]
    # channels = ["MEG1114"]
    # fname = "many_failed.npz"
    # channels = ["MEG0131"]
    signals, names, time, n_chan = fr.get_signals(fname, channels=channels)
    signal = signals[0]

    window = 400
    plot_time = time[:window]
    time_0 = plot_time[0]
    plot_time = [x - time_0 for x in plot_time]
    max_i = len(signals[0]) - 1

    i1 = []
    i2 = []
    i6 = []

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ani = FuncAnimation(fig, animate, frames=len(signal) - window, interval=.001, repeat=False)
    plt.show()


def test_magn2():
    smooth_window = 401
    offset = int(smooth_window / 2)
    printer = fr.Printer("print")
    #plt.rcParams.update({'font.size': 42})
    crop = False

    fname = datadir + "many_many_successful.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)

    if crop:
        time_window = [0.4, 0.45]
        precropped_signals, precropped_ix = hf.crop_signals_time(time_window, timex, signals, 200)

    detecs = np.load("array120_trans_newnames.npz")

    ave_window = 100
    ave_sens = 10 ** (-13)

    def seg_lens(sig, segs):
        length = 0
        for seg in segs:
            length += seg[1] - seg[0]

        return length / len(sig)

    if crop:
        signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(precropped_signals, names,
                                                                                                len(signals), filter_beginning=False)
    else:
        signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals,
                                                                                                    names,
                                                                                                    len(signals), printer,
                                                                                                    filter_beginning=True)

    for i in range(len(signals)):
        comp_detec = names[i]
        nearby_names = helper_funcs.find_nearby_detectors(comp_detec, detecs, names)
        nearby_names.append(comp_detec)

        new_names = []
        for name in nearby_names:
            index = names.index(name)
            bad_segs = bad_segment_list[index]
            bad = len(bad_segs) != 0

            if bad:
                print("excluding " + name)
                continue

            new_names.append(name)

        #print(len(new_names))

        if len(new_names) == 0:
            print()
            continue

        near_vs = []
        near_rs = []

        for name in new_names:
            near_vs.append(detecs[name][:3, 2])
            near_rs.append(detecs[name][:3, 3])

        # nearby_names.append(comp_detec)
        if crop:
            near_sigs = hf.find_signals(new_names, precropped_signals, names)
        else:
            near_sigs = hf.find_signals(new_names, signals, names)

        near_sus = hf.find_signals(new_names, suspicious_segment_list, names)
        near_bads = hf.find_signals(new_names, bad_segment_list, names)

        smooth_sigs = []
        detrended_sigs = []
        xs = []

        for k in range(len(near_sigs)):
            signal = near_sigs[k]
            # filter_i = sa.filter_start(signal)
            # filt_sig = signal[filter_i:]
            filtered_signal, x, smooth_signal, smooth_x, new_smooth = hf.filter_and_smooth(signal, offset,
                                                                                           smooth_window, smooth_only=crop)

            smooth_sigs.append(np.gradient(new_smooth))
            xs.append(x)

            if False:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
                plt.tight_layout(rect=(0.045, 0.02, 0.99, 0.98))
                linewidth = 4
                ax1.plot(timex[x], filtered_signal, linewidth=linewidth)
                ax2.plot(timex[x], new_smooth, linewidth=linewidth)
                ax3.plot(timex[x], np.gradient(new_smooth), linewidth=linewidth)
                ax3.set_xlabel("Time [s]")
                ax1.set_ylabel("Magn. F. [T]")
                ax2.set_ylabel("Magn. F. [T]")
                ax3.set_ylabel("Magn. F. \nDeriv. [T/s]")
                ax1.grid()
                ax2.grid()
                ax3.grid()

            # i_arr, ix, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(filt_sig, [2])
            # smooth_sigs.append(i_arr[0])
            # xs.append(ix)
            # detrended_sigs.append(detrended_signal)

            # smooth_sigs.append(smooth_signal)
            # xs.append(x)

        # for signal in smooth_sigs:
        # print(len(signal))

        cropped_signals, o_new_x = helper_funcs.crop_all_sigs(smooth_sigs, xs, [])

        # print(len(new_x))

        # print(len(cropped_signals))
        ave_of_aves, aves, all_diffs, rec_sigs, mag_is, new_cropped_signals, crop_x = pca.rec_and_diff(cropped_signals,
                                                                                                       [o_new_x],
                                                                                                       near_vs, printer,
                                                                                                       ave_window=ave_window)

        print(type(mag_is[0]))

        if False:
            fig, ax = plt.subplots(figsize=(12, 10))
            plt.tight_layout(rect=(0.045, 0.02, 0.99, 0.98))
            linewidth = 4
            ax.plot(timex[xs[0]], smooth_sigs[0], linewidth=linewidth, label="Preprocessed signal")
            ax.plot(timex[mag_is], rec_sigs[0], linewidth=linewidth, label="Reconstructed signal")
            ax.set_ylabel("Magn. F. \nDeriv. [T/s]")
            ax.set_xlabel("Time [s]")
            ax.legend()
            ax.grid()

        excludes, new_x, ave_diffs, rel_diffs = pca.filter_unphysical_sigs(smooth_sigs, new_names, xs, near_vs,
                                                                           near_bads, near_sus, printer,
                                                                           ave_window=ave_window, ave_sens=ave_sens)

        # print(o_new_x, new_x)

        if cropped_signals is None or rec_sigs is None:
            print()
            continue

        good_sigs = []
        good_names = []
        good_vs = []
        good_xs = []

        for k in range(len(smooth_sigs)):
            nam = new_names[k]

            if nam in excludes:
                continue

            good_sigs.append(cropped_signals[k])
            good_names.append(nam)
            good_vs.append(near_vs[k])
            # good_xs.append(xs[k])
            good_xs.append(o_new_x)

        good_ave_of_aves, good_aves, good_all_diffs, good_rec_sigs, good_mag_is, good_cropped_signals, good_crop_x = pca.rec_and_diff(
            good_sigs, [o_new_x], good_vs, printer, ave_window=ave_window)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(new_names)))
        good_colors = []
        for k in range(len(colors)):
            nam = new_names[k]
            if nam in excludes:
                continue
            good_colors.append(colors[k])

        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        #fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
        #plt.tight_layout(rect=(0.045, 0.02, 0.99, 0.98))
        linewidth = 1.5

        for k in range(len(cropped_signals)):
            color = colors[k]
            plot_name = new_names[k]
            ax1.plot(timex[xs[k]], smooth_sigs[k], color=color, label=plot_name, linewidth=linewidth)
            ax2.plot(timex[mag_is], rec_sigs[k], color=color, label=plot_name, linewidth=linewidth)
            ax3.plot(timex[mag_is], all_diffs[k], color=color, label=aves[k])

        #ax1.set_title(ave_of_aves)
        #ax2.legend()
        ax1.set_ylabel("Magn. F. \nDeriv. [T/s]")
        ax2.set_ylabel("Magn. F. \nDeriv. [T/s]")
        ax2.set_xlabel("Time [s]")
        ax1.grid()
        ax2.grid()
        #ax3.legend()

        fig2, (ax11, ax22, ax33) = plt.subplots(3, 1, sharex=True)
        #fig12, (ax11, ax22) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
        #plt.tight_layout(rect=(0.045, 0.02, 0.99, 0.98))

        for k in range(len(good_sigs)):
            color = good_colors[k]
            plot_name = good_names[k]
            ax11.plot(timex[good_xs[k]], good_sigs[k], color=color, label=plot_name, linewidth=linewidth)
            ax22.plot(timex[good_mag_is], good_rec_sigs[k], color=color, label=plot_name, linewidth=linewidth)
            ax33.plot(timex[good_mag_is], good_all_diffs[k], color=color, label=good_aves[k])

        #ax11.set_title(good_ave_of_aves)
        #ax22.legend()
        #ax33.legend()
        ax11.set_ylabel("Magn. F. \nDeriv. [T/s]")
        ax22.set_ylabel("Magn. F. \nDeriv. [T/s]")
        ax22.set_xlabel("Time [s]")
        ax11.grid()
        ax22.grid()

        if len(new_x) != 0:
            print(new_x[0], new_x[-1])
            #hf.plot_spans(ax11, [[new_x[0], new_x[-1]]])
            #hf.plot_spans(ax1, [[new_x[0], new_x[-1]]])
        else:
            print("no span")

        fig3, ax = plt.subplots()
        for k in range(len(near_sigs)):
            plot_name = new_names[k]
            color = colors[k]
            ax.plot(near_sigs[k], label=plot_name, color=color)

        ax.legend()

        print()

        plt.show()


def test_new_excluder():
    smooth_window = 401
    offset = int(smooth_window / 2)

    printer = fr.Printer("print")
    crop = False

    fname = datadir + "sample_data34.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)

    if crop:
        time_window = [0.22, 0.25]
        signals, ix, i_seg = hf.crop_signals_time(time_window, timex, signals, 200)
        signals, ix = helper_funcs.crop_all_sigs(signals, ix, [])

    detecs = np.load("array120_trans_newnames.npz")

    # times_excluded = np.zeros(n_chan)
    # times_in_calc = np.zeros(n_chan)

    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan, printer,
                                                                                                filter_beginning=(not crop))

    start_time = time.time()
    all_diffs, all_rel_diffs, chan_dict = pca.check_all_phys(signals, detecs, names, n_chan, bad_segment_list, suspicious_segment_list,
                                                             printer, ave_window=100, ave_sens=10 ** (-13), smooth_only=crop)
    end_time = time.time()

    status, confidence = pca.analyse_phys_dat_alt(all_diffs, names, all_rel_diffs, chan_dict)

    ex_time = (end_time - start_time) / 60
    print("execution time:", ex_time, "mins")

    titles = []

    for i in range(len(chan_dict)):
        stat = status[i]

        if stat == 0:
            st_string = "physical"
        elif stat == 1:
            st_string = "unphysical"
        elif stat == 2:
            st_string = "undetermined"
        elif stat == 3:
            st_string = "unused"
        # num_exc = times_excluded[i]
        # num_tot = times_in_calc[i]
        nam = names[i]

        diffs = all_diffs[nam]
        rel_diffs = all_rel_diffs[nam]
        chan_dat = chan_dict[nam]

        num_tot = len(chan_dat)
        num_exc = len([x for x in chan_dat if chan_dat[x] == 1])
        # print(num_tot, num_exc)
        num_exc = np.float64(num_exc)
        num_tot = np.float64(num_tot)

        tit = nam + " " + str(num_exc) + "/" + str(num_tot) + "=" + \
              str(num_exc / num_tot) + ": " + st_string + ", " + str(confidence[i])

        titles.append(tit)

        print(nam, num_exc, num_tot, np.float64(num_exc / num_tot), st_string, confidence[i], np.mean(rel_diffs))

    for i in range(len(chan_dict)):
        nam = names[i]
        chan_dat = chan_dict[nam]
        num_tot = len(chan_dat)
        num_exc = len([x for x in chan_dat if chan_dat[x] == 1])
        num_exc = np.float64(num_exc)
        num_tot = np.float64(num_tot)
        signal = signals[i]
        segs = bad_segment_list[i]

        nearby_names = hf.find_nearby_detectors(nam, detecs, names)
        sigs = hf.find_signals(nearby_names, signals, names)

        # if num_exc == 0:
        #    continue

        colors = plt.cm.rainbow(np.linspace(0, 1, len(nearby_names) + 1))
        fig, ax = plt.subplots()
        ax.plot(signal, label=nam, color=colors[0])
        hf.plot_spans(ax, segs)
        ax.set_title(titles[i])

        for j in range(len(nearby_names)):
            name = nearby_names[j]
            sig = sigs[j]
            ax.plot(sig, label=name, color=colors[j + 1])

        ax.legend()

        plt.show()


def test_fft_full():
    fname = datadir + "many_many_successful2.npz"
    channels = ["MEG2*1"]
    signals, names, timex, n_chan = fr.get_signals(fname)

    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan,
                                                                                                badness_sensitivity=.5)

    hf.plot_in_order_ver3(signals, names, n_chan, signal_statuses, bad_segment_list, suspicious_segment_list)


def test_fft():
    # SUS ONES: sd37: 2214
    # sd 32 is weird
    fname = datadir + "sample_data37.npz"
    #channels = ["MEG0624"]
    channels = ["MEG*1", "MEG*4"]
    signals, names, timex, n_chan = fr.get_signals(fname, channels=channels)
    printer = fr.Printer("print")

    detecs = np.load("array120_trans_newnames.npz")

    #time_window = [0.4, 0.5]
    #cropped_signals, cropped_ix = hf.crop_signals_time(time_window, timex, signals, 200)

    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan,
                                                                                                printer,
                                                                                                filters=["uniq", "flat", "spike"],
                                                                                                filter_beginning=True)
    lol = False
    #plt.rcParams.update({'font.size': 41})
    for i in range(n_chan):
        name = names[i]
        print(name)
        signal = signals[i]
        # signal = signals[i]
        bad_segs = bad_segment_list[i]
        filter_i = sa.filter_start(signal)

        filter_i_list = sa.filter_start(signal, debug=True)[0]

        #normal_sig = signal[filter_i:]
        #normal_x = list(range(filter_i, len(signal)))
        full_x = list(range(len(signal)))

        indices = [2]
        fft_window = 400

        nu_i_x, nu_i_arr, u_filter_i_i, u_i_arr_ave, u_i_arr_sdev, u_cut_grad, u_grad_ave, u_grad_x, status, sus_score, rms_x, fft_sdev, error_start, sdev_thresh, sdev_span, detrended_sig = sa.fft_filter(
            signal, filter_i, bad_segs, printer, indices=indices, fft_window=fft_window, debug=True, goertzel=False, lol=lol)

        if status == 0:
            s = "GOOD"

        if status == 1:
            s = "BAD"

        if status == 2:
            s = "UNDETERMINED"

        print("SIGNAL FFT STATUS: " + s, status)
        print("SUS SCORE:", sus_score)

        if nu_i_x is None:
            print("skip")
            print()
            continue


        #u_filter_i_i = u_filter_i_i + filter_i[0]
        #u_grad_x = [x + filter_i for x in u_grad_x]

        #nu_i_x = [x for x in nu_i_x]
        #u_filter_i_i = u_filter_i_i
        #u_grad_x = [x for x in u_grad_x]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

        #fig, (ax1) = plt.subplots(1, 1, figsize=(12, 10))
        #plt.tight_layout(rect=(0.045, 0.02, 0.99, 0.98))
        linewidth = 1.5

        #error_start = None
        #status = -1
        ax1.plot(timex[full_x], signal, linewidth=linewidth)
        ax1.axvline(timex[filter_i], linestyle="--", color="blue")
        #[ax1.axvline(timex[x], linestyle="--", color="black", linewidth=0.5) for x in filter_i_list]

        #filter_i = filter_i[0]
        nu_i_x = [x + filter_i for x in nu_i_x]
        #print(nu_i_arr)
        #print("lol")
        offset = int(len(nu_i_arr) * 0.0875)
        filter_i_i_list, largest = sa.filter_start(nu_i_arr, offset=offset, max_rel=.175, debug=True)
        figlol, (axlol1, axlol2) = plt.subplots(2, 1, sharex=True)
        axlol1.plot(np.gradient(filter_i_i_list))
        axlol2.plot(largest)
        axlol2.set_ylim(0, 10*10**(-10))
        axlol2.grid()
        axlol1.grid()
        ax3.plot(timex[nu_i_x], np.gradient(nu_i_arr))
        #print(filter_i_i_list)
        filter_i_i_list = [x + filter_i for x in filter_i_i_list]
        #u_filter_i_i = u_filter_i_i + filter_i
        # [ax2.axvline(timex[x], linestyle="--", color="black", linewidth=0.5) for x in filter_i_i_list]
        #u_filter_i_i = u_filter_i_i[0]

        #ax2.plot(timex[normal_x], detrended_sig, linewidth=linewidth)
        #ax2.set_yticklabels([])
        ax2.grid()
        #ax1.plot(full_x, signal)
        hf.plot_spans(ax1, hf.seg_to_time(timex, bad_segs), color="darkred")
        #hf.plot_spans(ax1, bad_segs)

        if error_start is not None:
            hf.plot_spans(ax1, hf.seg_to_time(timex, [[error_start, len(signal) - 1]]), color="red")
            #hf.plot_spans(ax1, [[error_start, len(signal) - 1]], color="red")
        elif status == 1:
            hf.plot_spans(ax1, hf.seg_to_time(timex, [[u_filter_i_i, len(signal) - 1]]), color="red")
            #hf.plot_spans(ax1, [[filter_i, len(signal) - 1]], color="red")
        elif status == 2:
            hf.plot_spans(ax1, hf.seg_to_time(timex, [[u_filter_i_i, len(signal) - 1]]), color="yellow")
            #hf.plot_spans(ax1, [[filter_i, len(signal) - 1]], color="yellow")

        #ax1.grid()
        ax1.set_ylabel("Magnetic Field [T]")
        #ax2.set_ylabel("Mag. F. [T]")

        if nu_i_x is not None:
             ax2.plot(timex[nu_i_x], nu_i_arr, linewidth=linewidth)
             #ax2.plot(timex[nu_i_x], nu_i_arr)
             pass

        ax2.set_ylim(-0.1 * 10 ** (-7), 1.2 * 10 ** (-7))
        ax3.grid()
        ax1.set_xlabel("Time [s]")
        #ax3.set_ylabel("FT Amp.")

        ax1.grid()

        # ax2.legend()

        ax2.axvline(timex[u_filter_i_i], linestyle="--", color="blue", linewidth=0.5)
        #ax3.set_ylim(-.25 * 10 ** (-9), .25 * 10 ** (-9))

        #ax3.plot(u_grad_x, u_cut_grad, label="orig")

        # fig2, ax11 = plt.subplots()

        # rms_x, fft_sdev, error_start, sdev_thresh, sdev_span = sa.find_saturation_point_from_fft(nu_i_x, nu_i_arr, u_filter_i_i, fft_window)

        if error_start is not None:
            #pass
            ax1.axvline(timex[error_start], linestyle="--", color="red")

        if sdev_span is not None:

            #pass
            hf.plot_spans(ax4, hf.seg_to_time(timex, [sdev_span + filter_i]))
            rms_x = [x + filter_i for x in rms_x]
            ax4.plot(timex[rms_x], fft_sdev, label="sdev")
            ax4.axhline(sdev_thresh, linestyle="--", color="black")
        #
        # # ax4.plot(rms_x, fft_rms, label="rms")
        # ax4.legend()
        # ax4.set_ylim(0, 2 * 10 ** (-9))

        #ax1.legend()
        #ax2.set_ylim(0, 10 ** (-7))

        # fig, (ax1) = plt.subplots(1, 1, sharex=True)
        #
        # ax4.set_ylim(-.5 * 10 ** (-9), .5 * 10 ** (-9))
        # ax3.axhline(thresh, linestyle="--", color="black")
        # ax3.axhline(-thresh, linestyle="--", color="black")
        # ax3.legend()

        plt.show()

        print()


def show():
    channels = ["MEG0511", "MEG0541"]
    fname = datadir + "many_successful.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)

    #print(1/0.0001)

    #signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan,
    #                                                                                            badness_sensitivity=.5,
    #filter_beginning=True)

    plt.rcParams.update({'font.size': 42})
    for i in range(n_chan):
        signal = signals[i]
        name = names[i]
        #bad_segs = bad_segment_list[i]
        #sus_segs = suspicious_segment_list[i]

        print(name)
        print()

        # plt.plot(timex, signal)
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
        linewidth = 4
        plt.plot(timex, signal, linewidth=linewidth)
        #plt.title(name)
        plt.ylabel("Magnetic Field [T]")
        plt.xlabel("Time [s]")
        plt.grid()

        #hf.plot_spans(ax, bad_segs, color="red")
        #hf.plot_spans(ax, sus_segs, color="yellow")

        plt.show()


def test_ffft():
    fname = datadir + "sample_data40.npz"
    # channels = ["MEG1041"]
    signals, names, timex, n_chan = fr.get_signals(fname)

    printer = fr.Printer("print")
    detecs = np.load("array120_trans_newnames.npz")

    indices = [2]

    for i in range(n_chan):
        signal = signals[i]
        name = names[i]
        print(name)
        filter_i = sa.filter_start(signal)
        signal = signal[filter_i:]

        start_time = time.time()
        i_arr, nu_x, smooth_signal, smooth_x, filtered_signal = sa.calc_fft_indices(signal, printer, indices=indices)
        end_time = time.time()
        orig_time = end_time - start_time

        start_time = time.time()
        new_i_arr, nu_x, smooth_signal, smooth_x, filtered_signal = sa.calc_fft_index_fast(signal, printer)
        end_time = time.time()
        new_time = end_time - start_time

        plt.plot(i_arr[0], label="orig, " + str(orig_time))
        plt.plot(new_i_arr, label="new, " + str(new_time))
        plt.legend()
        plt.show()


def show_pca():
    # mms 1241 -> bad exc
    skp = "MEG1044"
    fname = datadir + "sample_data02.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    printer = fr.Printer("print")
    skip = False

    start_t = time.time()
    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan, printer)
    all_diffs, all_rel_diffs, chan_dict = pca.check_all_phys(signals, detecs, names, n_chan, bad_segment_list, suspicious_segment_list,
                                                             printer, ave_window=100, ave_sens=5 * 10 ** (-13))
    phys_stat, phys_conf = pca.analyse_phys_dat_alt(all_diffs, names, all_rel_diffs, chan_dict)

    end_t = time.time()

    print(end_t - start_t)

    # plt.rcParams.update({'font.size': 21})
    remove_bads = True

    for i in range(n_chan):
        name = names[i]
        print(name)
        if skip and not name == skp:
            continue
        near_names = helper_funcs.find_nearby_detectors(name, detecs, names)
        near_names.append(name)
        sigs = hf.find_signals(near_names, signals, names)
        phys_stats = hf.find_signals(near_names, phys_stat, names)
        fracs = hf.find_signals(near_names, phys_conf, names)

        if remove_bads:
            bads = hf.find_signals(near_names, bad_segment_list, names)
            index_list = []
            for ind in range(len(bads)):
                b = bads[ind]
                if len(b) != 0:
                    index_list.append(ind)
                    print("removing " + near_names[ind])

            if len(index_list) > 0:
                for indy in sorted(index_list, reverse=True):
                    del sigs[indy]
                    del near_names[indy]
                    del phys_stats[indy]
                    del fracs[indy]

        if len(sigs) == 0:
            continue

        phys_stats_string = []
        for stat in phys_stats:
            if stat == 0:
                strng = "consistent"

            if stat == 1:
                strng = "inconsistent"

            if stat == 2:
                strng = "undetermined"

            if stat == 3:
                strng = "not used"

            phys_stats_string.append(strng)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(near_names)))
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))

        for j in range(len(near_names)):
            nam = near_names[j]
            sig = sigs[j]
            status = phys_stats_string[j]
            frac = fracs[j]

            lbl = nam + ": " + status
            if not status == "not used":
                lbl += ", exc. frac.:" + str(round(frac, 2))


            if nam == name:
                c = "black"
            else:
                c = colors[j]

            ax.plot(timex, sig, color=c, label=lbl, linewidth=4)

        ax.grid()
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Magnetic Field [T]")
        plt.show()

        print()

def show_helmet():
    fname = datadir + "many_many_successful.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    names, signals = hf.order_lists(detecs, names, signals)

    new_dat = []

    for i in range(len(names)):
        name = names[i]

        if name[-1] == "4":
            new_dat.append(1)
        else:
            new_dat.append(0)

    vis.plot_all(names, new_dat)


def test_flat_new():
    fname = datadir + "sample_data23.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    printer = fr.Printer("print")

    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan,
                                                                                                printer)


    for i in range(n_chan):
        signal = signals[i]
        name = names[i]
        bad_segs = bad_segment_list[i]
        sus_segs = suspicious_segment_list[i]

        print(name)
        #segs, scores = fdfs.flat_filter(signal, printer)

        fig, ax = plt.subplots()
        plt.plot(signal)
        hf.plot_spans(ax, bad_segs, color="red")
        hf.plot_spans(ax, sus_segs, color="yellow")
        plt.title(name)

        print()
        plt.show()
