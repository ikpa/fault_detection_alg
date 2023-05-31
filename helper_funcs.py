import fdfs as sa
import matplotlib.pyplot as plt
import numpy as np
import math


"""this file contains certain helper functions used in various places.
not all of them are used or commented"""

orig_time_window = (0.210, 0.50)
r_sens_sq = 0.06**2

def plot_spans(ax, segments, color="blue"):
    """plot/color all segments in list onto a figure

    parameters:
        ax: the axis to plot the spans to.
        segments: list containing the segments to plot.
        color: string. color of the segments.

    returns:
        void"""
    if len(segments) == 0:
        return

    for segment in segments:
        ax.axvspan(segment[0], segment[1], color=color, alpha=.5)

    return


def seg_to_time(x, segs):
    """converts all segments (in indices) to another unit, such as time.

    parameters:
        x: list of the x values in the same units the segments need to be converted to.
        segs: list of the segments to convert (in indices).

    returns:
        new_segs: list containing converted segments in units of x."""
    new_segs = []
    for seg in segs:
        new_segs.append(x[seg])

    return new_segs

def plot_in_order_ver3(signals, names, n_chan, statuses,
                       bad_seg_list, suspicious_seg_list,
                       physicality=[], time_x=None, ylims=None, showtitle=True):
    """plot all signals as well as show data calculated by the program. shows one signal at a time.
    clicking x moves on to the next signal.

    parameters:
        signals: list containing all signals.
        names: list containing names of all detectors/signals.
        n_chan: integer. number of signals.
        statuses: list of booleans. statuses[i] is True if signals[i] is bad.
        bad_seg_list: list containing bad segments.
        suspicious_seg_list: list containing suspicious segments.
        physicality: list of integers. physicality statuses of signals. see analyse_phys_dat and analyse_phys_dat_alt in pca.py.
        time_x: list of the x values of the plots. must be the same length as all of the lists in signals.
        ylims: y axis limits of the plots. see set_ylims in the matplotlib documentation.
        showtitle: boolean. if True, the titles are shown."""
    print_phys = not len(physicality) == 0

    #plt.rcParams.update({'font.size': 42})
    for i in range(n_chan):
        name = names[i]
        print(name)
        signal = signals[i]
        bad = statuses[i]
        bad_segs = bad_seg_list[i]
        suspicious_segs = suspicious_seg_list[i]

        if time_x is not None:
            bad_segs = seg_to_time(time_x, bad_segs)
            suspicious_segs = seg_to_time(time_x, suspicious_segs)

        if print_phys:
            phys = physicality[i]

            if phys == 0:
                phys_stat = ", physical"
            if phys == 1:
                phys_stat = ", unphysical"
            if phys == 2:
                phys_stat = ", physicality could not be determined"
            if phys == 3:
                phys_stat = ", not used in physicality calculation"

        else:
            phys_stat = ""

        if bad:
            status = "bad segments present"
        else:
            status = "no bad segments found"

        fig, ax = plt.subplots(figsize=(12,10))
        plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
        linewidth = 4

        if time_x is None:
            ax.plot(signal, linewidth=linewidth)
        else:
            ax.plot(time_x, signal, linewidth=linewidth)

        plot_spans(ax, bad_segs, color="red")
        plot_spans(ax, suspicious_segs, color="yellow")

        ax.grid()
        ax.set_ylabel("Magnetic Field [T]")
        ax.set_xlabel("Time [s]")
        if ylims is not None:
            ax.set_ylim(ylims)

        if showtitle:
            title = name + ": " + status + phys_stat
            ax.set_title(title)

        plt.show()
        print()

def bad_list_for_anim(names, bads):
    """USED FOR TESTING
    reformat list for animation function"""
    bad_names = []
    for i in range(len(names)):

        if bads[i]:
            bad_names.append(names[i])

    return bad_names

def order_lists(pos_list, dat_names, signals):
    """USED FOR TESTING"""
    new_signals = []
    new_names = []

    n = 0
    for name in pos_list:
        i = dat_names.index(name)
        new_names.append(dat_names[i])
        new_signals.append(signals[i])
        n += 1

    return new_names, new_signals


def filter_and_smooth(signal, offset, smooth_window, smooth_only=False):
    """filter the beginning spike from a signal and smooth it. filtering the beginning
    can be turned off if necessary.

    parameters:
        signal: list or array containing signal.
        offset: integer. offset for smooth_x. only used if the beginning is not filtered.
        smooth_window: integer. window length for the smoothing algorithm.
        smooth_only: boolean. if true, the signal is only smoothed and the beginning is not filtered.

    returns:
        filtered_signal: list containing the signal with the beginning spike cropped out.
        x: x values of the signal (in indices).
        smooth_signal: list containing the FULL smoothed signal, i.e. containing values outside the original signal span.
        smooth_x: list containing the x values of smooth_signal.
        new_smooth: the smoothed signal."""
    if not smooth_only:
        filter_i = sa.filter_start(signal)
    else:
        filter_i = 0

    filtered_signal = signal[filter_i:]
    x = list(range(filter_i, len(signal)))

    smooth_signal = smooth(filtered_signal, window_len=smooth_window)
    smooth_x = [x - offset + filter_i for x in list(range(len(smooth_signal)))]
    new_smooth = []
    for i in range(len(filtered_signal)):
        new_smooth.append(smooth_signal[i + offset])

    return filtered_signal, x, smooth_signal, smooth_x, new_smooth


def fix_segs(segs, offset):
    """offsets all segments by offset.

    parameters:
        segs: list of segments.
        offset: integer. number with which to offset.

    returns:
        new_segs: list of offset segments."""
    new_segs = []
    for seg in segs:
        new_segs.append([seg[0] + offset, seg[1] + offset])

    return new_segs


def split_into_lists(original_list):
    """split a single list of integers into several lists so that each new list
    contains no gaps between each integer.

    parameters:
        original_list: list of integers.

    returns:
        new_lists: list of lists. contains lists with no gaps between integers."""
    n = len(original_list)

    if n == 0:
        return original_list

    original_list.sort()
    new_lists = []
    lst = [original_list[0]]
    for i in range(1, n):
        integer = original_list[i]
        prev_int = integer - 1

        if prev_int not in lst:
            new_lists.append(lst)
            lst = [integer]
        elif i == n - 1:
            lst.append(integer)
            new_lists.append(lst)
        else:
            lst.append(integer)

    return new_lists

def i_seg_from_time_seg(time_seg, t_x):
    """converts a segment in units of time to segment in units of indices.

    parameters:
        time_seg: original segment (in seconds).
        t_x: list containing all x_values (in seconds.

    returns:
        segment in indices."""
    start_i = np.where(abs(time_seg[0] - t_x) == min(abs(time_seg[0] - t_x)))[0][0]
    end_i = np.where(abs(time_seg[1] - t_x) == min(abs(time_seg[1] - t_x)))[0][0]
    return [start_i, end_i]


def crop_signals_time(time_seg, t, signals, seg_extend):
    """takes in a list of signals and a time segment, and returns a list of signals cropped within that time segment.
    the crop is extended at each end of the crop window.

    parameters:
        time_seg: time segment/span to crop all signals to.
        t: list containing the x values of the signals.
        signals: list of lists containing all signals.
        seg_extend: integer. how much to extend the crop at each end of the time window in indices.

    returns:
        cropped_signals: list of lists containing cropped signals.
        cropped_ix: the x values of the signals in cropped_signals in indices.
        i_seg: time_seg in indices."""
    if time_seg[0] > time_seg[1]:
        raise Exception("start of segment cannot be greater than end of segment")

    final_i = len(t) - 1
    i_seg = i_seg_from_time_seg(time_seg, t)
    i_seg_extend = [i_seg[0] - seg_extend, i_seg[-1] + seg_extend]

    if i_seg_extend[0] < 0:
        i_seg_extend[0] = 0

    if i_seg_extend[-1] > final_i:
        i_seg_extend[-1] = final_i

    if time_seg[0] < orig_time_window[0] or time_seg[1] > orig_time_window[1]:
        print("the window you have requested is out of bounds for the data window."
              " outputting the following window instead:", t[i_seg_extend])

    cropped_signals = []
    cropped_ix = []

    # print(i_seg_extend)

    for signal in signals:
        filter_i = sa.filter_start(signal)

        if filter_i > i_seg_extend[0]:
            start_i = filter_i
        else:
            start_i = i_seg_extend[0]

        cropped_signals.append(signal[start_i:i_seg_extend[-1]])
        cropped_ix.append(list(range(start_i, i_seg_extend[-1])))

    return cropped_signals, cropped_ix, i_seg


def segs_from_i_to_time(ix_list, t_x, bad_segs):
    """convert list of segments in indices to a segment in seconds.

    parameters:
        ix_list: list containing all x values in indices.
        t_x: list containing x values in seconds.
        bad_segs: list containing all segments (in indices) to convert:

    returns:
        bad_segs_time: converted segments."""
    ix_is_list = not isinstance(ix_list[0], int)
    bad_segs_time = []
    for i in range(len(bad_segs)):
        bad_seg = bad_segs[i]
        if ix_is_list:
            i_x = ix_list[i]
        else:
            i_x = ix_list
        offset = i_x[0]

        fixed_bads = fix_segs(bad_seg, offset)

        bad_segs_time_1 = []
        for bad_seg_1 in fixed_bads:
            #print(bad_seg_1)
            start_t = t_x[bad_seg_1[0]]

            if bad_seg_1[-1] > len(t_x) + 1:
                end_t = t_x[-1]
            else:
                end_t = t_x[bad_seg_1[-1]]
            bad_segs_time_1.append([start_t, end_t])

        bad_segs_time.append(bad_segs_time_1)

    return bad_segs_time


def find_good_segs(i_x_target, bad_seg_list, i_x_tot):
    """finds all good segments from a set of signals based on the found bad segments. works for signals cropped by
    crop_signals_time.

    parameters:
        i_x_target: list containing x values. only contains the cropped x values.
        bad_seg_list: list of bad segments.
        i_x_tot: list containing x values. contains extended values.

    returns:
        good_seg_is: list containing good segments."""
    #ix_set = set(list(range(i_x_target[0] - i_x_tot[0], i_x_target[1] - i_x_tot[0] + 1)))
    good_seg_is = []

    #print(i_x_tot)

    for i in range(len(bad_seg_list)):
        bad_segs = bad_seg_list[i]
        ix = i_x_tot[0]
        ix_set = set(list(range(i_x_target[0] - ix[0], i_x_target[1] - ix[0] + 1)))
        #print(bad_segs)

        bad_i_set = set()
        for bad_seg in bad_segs:
            bad_list = list(range(bad_seg[0], bad_seg[-1] + 1))
            bad_i_set.update(bad_list)

        good_list_all = [x for x in list(ix_set - bad_i_set) if x >= 0]

        split_good_list = split_into_lists(good_list_all)

        good_is = []
        for good_list in split_good_list:
            good_is.append([good_list[0], good_list[-1]])

        good_seg_is.append(good_is)

        #print(good_list)

        # if len(good_is) > 1:
        #     print(good_list_all)
        #     print(good_is)

    return good_seg_is


def find_signals(channels, data_arr, names):
    """finds and returns data corresponding to channel names. data_arr and
    names must be ordered and have the same length.

    parameters:
        channels: list of channel names to find.
        data_arr: list containing all data to search.
        names: list of all channel names.

    returns:
        signals_to_return: list containing the data corresponding to each name in channels."""
    indices = []

    for channel in channels:
        i = names.index(channel)
        indices.append(i)

    signals_to_return = []

    for index in indices:
        signals_to_return.append(data_arr[index])

    return signals_to_return


def reorganize_signals(signals, n):
    """reformats signals so that a single array contains one signal instead of
    one array containing a single data point from all channels.

    parameters:
        signals: list containing all signals.
        n: integer. number of signals.

    returns:
        new_signals: list containing reorganized signals."""
    new_signals = []
    for i in range(n):
        signal = signals[:, i]
        new_signals.append(signal)

    return new_signals

def filter_and_smooth_and_gradient_all(signals, offset, smooth_window, smooth_only=False):
    """filters beginning spike, smooths and calculates the time derivative of all signals.
    filtering the beginning spike can be turned on or off.

    parameters:
        signals: list containing all signals.
        offset: integer. offset for the x values if smooth_only is chosen.
        smooth_window: integer. window length for smoothing algorithm.
        smooth_only: boolean. if True, the beginning spike is not filtered.

    returns:
        filt_signs: list containing all processed signals.
        xs: list containing the x values for each processed signal."""
    filt_sigs = []
    xs = []

    for signal in signals:
        filtered_signal, x, smooth_signal, smooth_x, new_smooth = filter_and_smooth(signal, offset, smooth_window,
                                                                                    smooth_only=smooth_only)

        filt_sigs.append(np.gradient(filtered_signal))
        xs.append(x)

    return filt_sigs, xs


def smooth(x, window_len=21, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def crop_all_sigs(signals, xs, bad_segs):
    """takes several signals and crops them on the x axis so that all signals
    are of the same length. if bad segments are present, only the part
    before these segments are included. x-values must be in indices.

    parameters:
        signals: list of lists containing all signals.
        xs: list of lists containing all x values for each signal.
        bad_segs: list containing all bad segments

    returns:
        new_signals: list containing all cropped signals.
        new_x: list of x values of the cropped signals."""
    highest_min_x = 0
    lowest_max_x = 10 ** 100

    # find the min and max x values that are shared by all signals
    for x in xs:
        min_x = np.amin(x)
        max_x = np.amax(x)

        if min_x > highest_min_x:
            highest_min_x = min_x

        if max_x < lowest_max_x:
            lowest_max_x = max_x

    # remove all x values appearing in or after all bad segments
    for seg_list in bad_segs:
        for seg in seg_list:
            if lowest_max_x > seg[0]:
                lowest_max_x = seg[0]

    new_x = list(range(highest_min_x, lowest_max_x))
    new_signals = []

    # get parts of all signals that appear within the new x values
    for i in range(len(signals)):
        signal = signals[i]
        x = xs[i]
        max_i = x.index(lowest_max_x)
        min_i = x.index(highest_min_x)
        new_signals.append(signal[min_i:max_i])

    return new_signals, new_x


def averaged_signal(signal, ave_window, x=[], mode=0):
    """calculate rolling operation to signal. returns a signal that is
    len(signal)/ave_window data points long.
    modes:
    0 = average.
    1 = rms.
    2 = sdev.

    parameters:
        signal: list containing signal.
        ave_window: integer. window length for operation:
        x: list containing x values of signal.
        mode: integer. determines which operation to run

    returns:
        new_sig: list containing processed signal.
        new_x: list containing x values of processed signal. returned only if x was inputted."""
    new_sig = []
    new_x = []

    start_i = 0
    end_i = ave_window
    max_i = len(signal) - 1

    cont = True
    while cont:
        seg = signal[start_i:end_i]
        if mode == 0:
            ave = np.mean(seg)

        if mode == 1:
            ave = np.sqrt(np.mean([x ** 2 for x in seg]))

        if mode == 2:
            ave = np.std(seg)

        new_sig.append(ave)

        if len(x) != 0:
            new_x.append(int(np.mean([x[start_i], x[end_i]])))

        start_i = end_i
        end_i = end_i + ave_window

        if end_i > max_i:
            end_i = max_i

        if start_i >= max_i:
            cont = False

    if len(x) != 0:
        return new_sig, new_x

    return new_sig


def calc_diff(signal1, signal2, x):
    """calculate absolute difference between points in two different signals. signals do not need to be the same length.
    x-values must be in indices.

    parameters:
        signal1: list containing the first signal.
        signal2: list containing the second signal. can be longer than signal1.
        x: list containing the x values of signal1.

    returns:
        diffs: list containing the differences between the signals.
        new_x: list containing the new x values. currently the same as x"""
    new_x = []
    diffs = []
    for i in range(len(signal1)):
        new_x.append(x[i])
        point1 = signal1[i]
        point2 = signal2[i]
        diffs.append(abs(point1 - point2))

    return diffs, new_x


# TODO make faster
def find_nearby_detectors(d_name, detectors, good_names):
    """find detectors within a radius of 6 cm from a given detector. channels
    not in good_names are not counted.

    parameters:
        d_name: string. name of detector.
        detectors: list of strings containing all detector names:
        good_names: list of strings containing the names of the detectors to exclude.

    returns:
        nears: list of strings containing the names of the nearby detectors."""
    dut = detectors[d_name]
    r_dut = dut[:3, 3]

    nears = []

    for name in detectors:
        if name == d_name or name not in good_names:
            continue

        detector = detectors[name]
        r = detector[:3, 3]
        delta_r = (r_dut[0] - r[0]) ** 2 + (r_dut[1] - r[1]) ** 2 + (r_dut[2] - r[2]) ** 2

        if delta_r < r_sens_sq:
            nears.append(name)

    return nears


def length_of_segments(segments):
    """calculate total length of all segments.

    parameters:
        segments: list of segments.

    returns:
        tot_len: integer. length of segments"""
    tot_len = 0
    for segment in segments:
        length = segment[1] - segment[0]
        tot_len += length

    return tot_len


def frac_of_sigs(segs):
    """calculate the fraction of signals that have segments.

    parameters:
        segs: list of segments.

    returns:
        n_segs: integer. number of signals that have segments.
        frac: float. fraction of signals that have segments."""
    n_tot = len(segs)

    n_segs = 0
    for seg in segs:
        if seg:
            n_segs += 1

    frac = n_segs/n_tot
    return n_segs, frac
