import numpy as np
import time
import helper_funcs as hf
from operator import itemgetter

"""all sensitivity and weight values in this file have been determined
experimentally and changing them will affect the accuracy of the program
significantly"""

sample_freq = 10000  # sampling frequency of the squids


def filter_start(signal, offset=50, max_rel=0.05, debug=False):
    """filter the jump in the beginning of the signal. returns the index
    where the jump has ended.

    parameters:
        signal: array or list containing the signal.
        offset: how much to add to the index of the actual jump.
        max_rel: the relative length of the signal to analyse (from the beginning).
        debug: debug mode.

    returns:
        result: the index where to cut the signal."""
    max_i = int(max_rel * len(signal))
    grad = np.gradient(signal)
    new_grad = grad[:max_i]
    new_grad = abs(np.array(new_grad))
    largest = sorted(new_grad, reverse=True)[0:20]
    locations = sorted([np.where(new_grad == x)[0][0] for x in largest])

    if largest[0] < 2.5 * 10 ** (-10) or any(x >= 50 for x in np.gradient(locations)):
        # print("result:", 0)
        # result = np.int64(0)
        result = min(locations)
    else:
        # print("result:", locations[0] + offset)
        result = locations[0] + offset

    # farther_i = np.amax([max_grad_i, min_grad_i])
    if debug:
        return locations, largest

    return result


def averages_are_close(signal, start_is, end_is, averages=None, std_sensitivity=0.015):
    """check if the values of segments are close to eachother. previously calculated
    averages may also be included. difference in values is determined by
    calculating the standard deviation of all values.

    parameters:
        signal: list or array containing signal.
        start_is: list containing the start indices of all segments.
        end_is: list containing the end indices of all segments.
        averages: additional averages to evaluate.
        std_sensitivity: the tolerance of the standard deviation of all evaluated averages.

    returns:
        boolean. true if averages are close."""
    if averages is None:
        averages = []

    if len(start_is) == 0:
        return False

    if len(start_is) == 1 and len(averages) == 0:
        return True

    for i in range(len(start_is)):
        segment = signal[start_is[i]: end_is[i]]
        av = np.mean(segment)
        averages.append(av)

    av_of_avs = sum(averages) / len(averages)
    std = np.std(averages) / abs(av_of_avs)
    return std <= std_sensitivity


def average_of_gradient(signal, start_i, end_i, rel_offset=0.05):
    """calculate the average value of the gradient of a signal between
    start_i and end_i. start_i may be offset if the start of a segment
    needs to be excluded (due to how find_flat_segments works the start of
    a given segment may be slightly before the signal truly flattens)

    parameters:
        signal: list or array containing signal.
        start_i: start index of segment to analyse.
        end_i: end index of segment to analyse.
        rel_offset: how much start_i is offset by.

    returns:
        float. mean of the gradient of the segment."""
    length = end_i - start_i
    offset = int(rel_offset * length)
    segment = signal[start_i + offset: end_i]
    grad = np.gradient(segment)
    return np.mean(grad)


def uniq_filter_neo(signal, filter_i):
    """find segments where a certain value repeats. this filter ignores parts
    where the signal deviates from the unique value momentarily.

    parameters:
        signal: list or array containing signal.
        filter_i: index returned by filter_start().

    returns:
        list of lists. list containing all found segments (contains only one segment).
        list of int. scores of all segments (contains only one score)."""
    uniqs, indices, counts = np.unique(signal[:], return_index=True, return_counts=True)
    max_repeat = np.amax(counts)
    if max_repeat <= 10:
        return [], []
    uniq_is = np.where(counts == max_repeat)

    max_vals = uniqs[uniq_is]
    where_repeat = np.where(signal == max_vals[0])
    where_repeat = list(where_repeat[0])
    where_repeat = [x for x in where_repeat if x > filter_i]

    if len(where_repeat) == 0:
        return [], []

    seg_start = np.amin(where_repeat)
    seg_end = np.amax(where_repeat)

    return [[seg_start, seg_end]], [2]


def reformat_segments(start_is, end_is):
    """reformat segment start and end indices into a list of tuples

    parameters:
        start_is: list of ints. all start indices.
        end_is: list of ints. all end indices.

    returns:
        lst: list of tuples. contains all segments."""
    lst = []
    for i in range(len(start_is)):
        lst.append([start_is[i], end_is[i]])

    return lst


def find_flat_segments(signal):
    """finds segments in the signal where the value stays approximately the same for long periods.
    returns lengths of segments, as well as their start and end indices.
    rel_sensitive_length determines how long a segment needs to be marked
    and relative_sensitivity determines how close the values need to be.

    parameters:
        signal: list or array containing signal.

    returns:
        lengths: list containing lengths of segments.
        start_is: list containing start indices of segments.
        end_is: list containing end indices of segments."""
    lengths = []
    start_is = []
    end_is = []
    lock_val = None  # subsequent values are compared to this value

    # sensitive_length = len(signal) * rel_sensitive_length
    sensitive_length = 200
    length = 1
    for i in range(len(signal)):
        val = signal[i]

        if lock_val is None:
            is_close = False
        else:
            # print("rel", abs(abs(val - lock_val) / lock_val), "abs", abs(val - lock_val))
            # is_close = abs(abs(val - lock_val) / lock_val) < relative_sensitivity
            is_close = abs(val - lock_val) <= 2 * 10 ** (-10)

        if not is_close or (is_close and i == len(signal) - 1):
            # print("cut")
            if length > sensitive_length:
                start_is.append(start_i)
                end_is.append(i)
                lengths.append(length)
            start_i = i
            length = 1
            lock_val = val

        if is_close:
            length += 1

    return lengths, start_is, end_is


def cal_seg_score_flat(signal, start_i, end_i, printer,
                       uniq_w=1.5, grad_sensitivity=0.5 * 10 ** (-13),
                       grad_w=10 ** 12, len_w=1, max_len=2900):
    """calculate a goodness value or score (a value determining how likely a flat segment
    has been wrongly detected by find_flat_segments) for a segment in a
    signal. a goodness value is increased if a segment has no clear trend,
    has a small fraction of unique values and is long.
    goodness > 1 -> bad.
    goodness < 1 -> good/suspicious.
    goodness < 0 -> very good.

    parameters:
        signal: list or array containing signal.
        start_i: integer. start index of segment to score.
        end_i: integer. end index of segment to score.
        printer: printer object. see file_handler.py.
        uniq_w, grad_w, len_w: floats. weights for scoring.
        grad_sensitivity: float. if the absolute value of the gradient is below this, grad score is 0.
        max_len: integer. maximum length of signal. NOT CHANGED.

    returns:
        tot_conf: float. score for segment."""
    segment = signal[start_i: end_i]
    uniqs = np.unique(segment)

    uniquevals = len(uniqs)
    totvals = len(segment)
    frac_of_uniq = 1 - uniquevals / totvals

    uniq_conf = uniq_w * frac_of_uniq

    grad_average = abs(average_of_gradient(signal, start_i, end_i))
    printer.extended_write("grad_average: ", grad_average)

    if grad_average < grad_sensitivity:
        grad_conf = 0
    else:
        grad_conf = - grad_w * grad_average

    rel_len = (end_i - start_i) / max_len
    sig_len = len(signal)
    # print(len(signal))

    if sig_len >= max_len / 2:
        len_w = 1.5 * len_w
        grad_conf *= 1.5

    len_conf = rel_len * len_w

    printer.extended_write("uniq_conf:", uniq_conf, "grad_conf:", grad_conf, "len_conf:", len_conf)

    tot_conf = uniq_conf + grad_conf + len_conf
    printer.extended_write("tot_conf:", tot_conf)
    return tot_conf


def flat_filter(signal, printer, grad_sens=0.5 * 10 ** (-13)):
    """find segments where the value stays the same value for a long period.
    also recalculates the tail of the signal and calculates
    a confidence value for the segment.

    parameters:
        signal: list or array containing signal.
        printer: printer object. see file_handler.py.
        grad_sens: float. sensitivity threshold for the absolute value of the gradient.

    returns:
        comb_segs: list of tuples. contains all suspicious and bad segments.
        confidences: list of floats. scores for all returned segments."""
    lengths, start_is, end_is = find_flat_segments(signal)

    if len(start_is) == 0:
        return [], []

    final_i = end_is[len(end_is) - 1]
    seg_is = reformat_segments(start_is, end_is)

    printer.extended_write("number of segments found ", len(seg_is))

    # recheck tail
    if final_i != len(signal) - 1:
        tail_ave = [np.mean(signal[final_i:])]
    else:
        tail_ave = []

    close = averages_are_close(signal, start_is, end_is, averages=tail_ave)

    # if the averages of all segments are close to each other, they are combined
    # into one segment
    if close:
        if not tail_ave:
            seg_is = [[start_is[0], end_is[len(end_is) - 1]]]
        else:
            seg_is = [[start_is[0], len(signal) - 1]]

    comb_segs = combine_segments(seg_is)

    printer.extended_write("number of segments outputted", len(comb_segs))
    # print(comb_segs)

    confidences = []
    for segment in comb_segs:
        confidences.append(cal_seg_score_flat(signal, segment[0], segment[1], printer, grad_sensitivity=grad_sens))

    return comb_segs, confidences


def cal_seg_score_spike(gradient, spikes, all_diffs, printer, max_sensitivities=None,
                        n_sensitivities=None,
                        grad_sensitivity=2 * 10 ** (-13),
                        sdens_sensitivity=0.1):
    """calculate a goodness value for a segment found by find_spikes. the confidence
    depends on the steepness of the spikes and their number,
    average gradient of the segment and the density of spikes.
    returns both the confidence and a segment that starts
    at the first spike and ends at the last.
    goodness > 1 -> bad.
    goodness < 1 -> good.

    parameters:
        gradient: list or array containing the derivative of the signal.
        spikes: list of lists. each individual list contains the indices where the derivative is considered a spike.
        all_diffs: list of lists. contains differences between grad_sensitivity and the actual value of the derivative for
        all spike indices.
        printer: printer object. see file_handler.py.
        max_sensitivities, n_sensitivities, grad_sensitivity, sdens_sensitivity: floats. sensitivity/threshold values
        for diffs, number of spikes, average gradient value and spike density.

    returns:
        list: start and end indices of the marked segment.
        score: float. score of segment."""
    if n_sensitivities is None:
        n_sensitivities = [20, 100]

    if max_sensitivities is None:
        max_sensitivities = [1.5, 1, .5]

    n = len(spikes)

    if n <= 1:
        return [], None

    score = .5

    first_spike = spikes[0]
    seg_start = first_spike[0]
    last_spike = spikes[len(spikes) - 1]
    seg_end = last_spike[len(last_spike) - 1]
    seg_len = seg_end - seg_start

    max_diffs = []
    for i in range(n):
        diffs = all_diffs[i]
        max_diffs.append(np.amax(diffs))

    av_max = np.mean(max_diffs)

    # TEST DIFFS----------------------------------------
    if av_max >= max_sensitivities[0]:
        score += 2
    elif av_max >= max_sensitivities[1]:
        score += 1
    elif av_max >= max_sensitivities[2]:
        score += .5
    # --------------------------------------------------

    if n == 1:
        return [seg_start, seg_end], score

    spike_density = n / seg_len

    grad_ave = abs(np.mean(gradient[seg_start:seg_end]))

    # TEST NUMBER OF SPIKES-----------------------------
    if n >= n_sensitivities[1]:
        score += 1
    elif n >= n_sensitivities[0]:
        score += .5
    # --------------------------------------------------

    # TEST GRADIENT-------------------------------------
    if grad_ave >= grad_sensitivity:
        score -= .25
    else:
        score += .5
    # --------------------------------------------------

    # TEST SPIKE DENSITY--------------------------------
    if spike_density >= sdens_sensitivity:
        score += 1

    score = score / 1.5

    printer.extended_write("num_spikes", n, "av_diff", av_max, "grad_ave", grad_ave,
                           "spike_density", spike_density, "badness", score)

    return [seg_start, seg_end], score


def find_spikes(gradient, filter_i, grad_sensitivity, len_sensitivity=6, start_sens=150):
    """find spikes in the signal by checking where the absolute gradient of the signal
    abruptly goes above grad_sensitivity. returns the spikes and their
    difference between the gradient and the grad_sensitivity.

    parameters:
        gradient: list containing derivative of signal.
        filter_i: integer. index returned by filter_start().
        grad_sensitivity, len_sensitivity, start_sens: floats. sensitivities for value of gradient, spike length and
        length of unanalysed signal at the start.

    returns:
        spikes: list of lists. each individual list contains the indices where the derivative is considered a spike.
        all_diffs: list of lists. contains differences between grad_sensitivity and the actual value of the derivative for
        all spike indices."""
    spikes = []
    all_diffs = []

    diffs = []
    spike = []
    # print("filter i", filter_i)
    for i in range(filter_i, len(gradient)):
        val = abs(gradient[i])

        if val > grad_sensitivity:
            spike.append(i)
            diffs.append((val - grad_sensitivity) / grad_sensitivity)
            continue

        if i - 1 in spike:
            if len(spike) < len_sensitivity and abs(filter_i - spike[0]) > start_sens:
                spikes.append(spike)
                all_diffs.append(diffs)

            spike = []
            diffs = []

    return spikes, all_diffs


def spike_filter_neo(signal, filter_i, printer, grad_sensitivity=10 ** (-10)):
    """finds segments with steep spikes in the signal and calculates their score.

    parameters:
        signal: list or array containing signal.
        filter_i: integer. index returned by filter_start().
        printer: printer object. see file_handler.py.
        grad_sensitivity: float. the value above which the gradient needs to be in order to be concidered a spike.

    returns:
        list containing the marked segment.
        list containing the score of the marked segment."""
    gradient = np.gradient(signal)
    spikes, all_diffs = find_spikes(gradient, filter_i, grad_sensitivity)
    seg_is, confidence = cal_seg_score_spike(gradient, spikes, all_diffs, printer)

    if len(seg_is) == 0:
        return [], []

    final_i = len(signal) - 1

    if seg_is[1] != final_i:
        tail = signal[seg_is[1]:]
        tail_ave = np.mean(tail)
        close = averages_are_close(signal, [seg_is[0]], [seg_is[1]], averages=[tail_ave])

        if close:
            seg_is[1] = final_i

    return [seg_is], [confidence]


def get_fft(signal, filter_i=0):
    """NOT USED
    calculate the fft (absolute value) of the signal
    """
    from scipy.fft import fft

    if len(signal) == 0:
        return [0]

    ftrans = fft(signal[filter_i:])
    # ftrans_abs = [abs(x) for x in ftrans]
    ftrans_abs = abs(ftrans)
    # ftrans_abs = ftrans
    return ftrans_abs, ftrans


def calc_fft_indices(signal, printer, indices=None, window=400, smooth_window=401, filter_offset=0, goertzel=False):
    """NOT USED
    calculate the fft for a windowed part of the signal. the window is scanned across
    the entire signal so that the fft values are as a function of time.
    returns the specified indices of the fft as a function of time.
    before calculating the ffts the trend of the signal is removed by first
    calculating a rolling average with a very large smoothing window and then
    getting the difference between the smoothed signal and the original."""
    if indices is None:
        indices = [1, 2, 6]

    if len(signal) < window:
        printer.extended_write("stopping fft, window larger than signal")
        return None, None, None, None, None

    if goertzel:
        goertzel_freqs = []

        for i in indices:
            freq_start = i * sample_freq / window
            goertzel_freqs.append((freq_start, freq_start + 1))

    sig_len = len(signal)
    ftrans_points = sig_len - window
    i_arr = np.zeros((len(indices), ftrans_points))

    # calculate smoothed signal
    offset = int(smooth_window / 2)
    smooth_signal = hf.smooth(signal, window_len=smooth_window)
    # CHECK AGAIN IF THIS IS USED SOMEWHERE
    smooth_x = [x - offset + filter_offset for x in list(range(len(smooth_signal)))]

    # remove trailing values for the smooth signal
    new_smooth = []
    for i in range(sig_len):
        new_smooth.append(smooth_signal[i + offset])

    # remove trend from original signal
    filtered_signal = [a - b for a, b in zip(signal, new_smooth)]

    # calculate ffts
    for i in range(ftrans_points):
        end_i = i + window
        signal_windowed = filtered_signal[i: end_i]

        if not goertzel:
            ftrans, ftrans_comp = get_fft(signal_windowed)

            for j in range(len(indices)):
                index = indices[j]
                # print("orig", ftrans_comp[index])
                i_arr[j][i] = ftrans[index]
                # i_arr[j][i] = ftrans_comp[index]

    nu_x = list(range(ftrans_points))
    # nu_x = [x + filter_offset for x in nu_x]

    return i_arr, nu_x, smooth_signal, smooth_x, filtered_signal


def calc_dft_constants(k, N):
    """calculate all constant elements (e^(-i*2*pi*k*n/N)) for dft calculation.

    parameters:
        k: integer. index of the dft component to calculate.
        N: integer. number of points in signal.

    returns:
        dft_factors: list of floats. contains dft factors."""
    const = -2 * np.pi / N
    dft_factors = []
    for n in range(N):
        dft_factors.append(complex(np.cos(const * k * n), np.sin(const * k * n)))

    return dft_factors


def calc_fft_index_fast(signal, printer, dft_consts, window=400, smooth_window=401, filter_offset=0):
    """calculates a single element of the dft of a signal as a function of time.

    parameters:
        signal: list or array containing signal.
        printer: printer object. see file_handler.py.
        dft_consts: list of floats. contains all constant elements of the dft.
        window: integer. window length of fft series.
        smooth_window: integer. window length of signal smoothing.
        filter_offset: integer. constant used in creating smooth_x. MOST LIKELY NOT USED ANYWHERE.

    returns:
        fft_tseries: list of floats. the required dft index as a function of time.
        nu_x: list of integers. x values (in indices) of fft_tseries.
        smooth_signal: list containing smoothed signal.
        smooth_x: x values (in indices) of smooth_signal.
        filtered_signal: list containing signal with the trend removed."""
    sig_len = len(signal)
    ftrans_points = sig_len - window

    if sig_len < window:
        printer.extended_write("stopping dft, window larger than signal")
        return None, None, None, None, None

    # calculate smoothed signal
    offset = int(smooth_window / 2)
    smooth_signal = hf.smooth(signal, window_len=smooth_window)
    # CHECK AGAIN IF THIS IS USED SOMEWHERE
    smooth_x = [x - offset + filter_offset for x in list(range(len(smooth_signal)))]

    # remove trailing values for the smooth signal
    new_smooth = []
    for i in range(sig_len):
        new_smooth.append(smooth_signal[i + offset])

    # remove trend from original signal
    filtered_signal = [a - b for a, b in zip(signal, new_smooth)]

    def make_window(i, signal):
        return signal[i:i + window]

    def calc_dft_point(segment, dft_consts):
        dft_list = np.dot(segment, dft_consts)
        dft_sum = np.sum(dft_list)
        return dft_sum

    # dft_consts = calc_dft_constants(2, window)

    window_func = np.vectorize(make_window, signature="(),(m)->(k)")
    windows = window_func(list(range(ftrans_points)), filtered_signal)

    dft_func = np.vectorize(calc_dft_point, signature="(n),(n)->()")
    fft_tseries = dft_func(windows, dft_consts)

    fft_tseries = abs(fft_tseries)

    nu_x = list(range(ftrans_points))

    return fft_tseries, nu_x, smooth_signal, smooth_x, filtered_signal


def stats_from_i(i_arr, i_x, bad_segs, fft_window, printer, cut_length=70, max_sig_len=800):
    """post_process fft time series, calculate statistics and score.
    status:
    0 = good.
    1 = bad.
    2 = undetermined.

    parameters:
        i_arr: array containing fft time series.
        i_x: x values (in indices) of time series.
        bad_segs: list containing bad segments.
        fft_window: integer. length of rolling window.
        printer: printer object. see file_handler.py.
        cut_length: integer. length (in indices) of signal to remove from the end.
        max_sig_len: integer. maximum length of signal.

    returns:
        nu_i_arr: post processed (portions cut from start and end) fft time series.
        nu_i_x: x values (in indices) of post processed fft time series.
        filter_i_i: integer. index returned by filter_start when run on fft time series.
        i_arr_ave: float. average of fft time series.
        i_arr_sdev: float. standard deviation of fft time series.
        cut_grad: derivative of post processed fft time series.
        grad_ave: float. average of derivative.
        grad_x: x values (in indices) of derivative.
        status: integer. status of fft analysis (0, 1, 2).
        sus_score: float. score of fft time series:
        short: boolean. if true, fft time series is short."""
    printer.extended_write("filtering fft:")
    offset = int(len(i_arr) * 0.0875)
    filter_i_i = filter_start(i_arr, offset=offset, max_rel=.175)
    # filter_i_i = filter_i_list[0]

    last_i = len(i_arr) - 1

    if len(bad_segs) == 0:
        minus_i = cut_length
    else:
        final_i = bad_segs[0][0] - 2 * cut_length
        minus_i = last_i - final_i + fft_window

    nu_i_arr = i_arr[:-minus_i]
    cut_i_arr = nu_i_arr[filter_i_i:]
    nu_i_x = i_x[:-minus_i]

    i_arr_ave = np.mean(cut_i_arr)
    i_arr_sdev = np.std(cut_i_arr)

    grad = np.gradient(i_arr)
    cut_grad = grad[filter_i_i:-minus_i]
    grad_ave = np.mean(cut_grad)
    grad_x = list(range(filter_i_i, i_x[-1] + 1 - minus_i))

    grad_rmsd = np.sqrt(np.mean([(grad_ave - x) ** 2 for x in cut_grad]))

    printer.extended_write("i ave:", i_arr_ave, " i sdev:",
                           i_arr_sdev)  # ave < 10e-09 - 5e-09 => SUS, sdev > 3e-09 => SUS
    printer.extended_write("grad rmsd", grad_rmsd)  # sus thresh: 2.5e-11, bad tresh: 7e-11 (raise this possibly)
    printer.extended_write("grad ave", grad_ave)  # > 1e-12 => SUS
    printer.extended_write("filtered fft length ", len(cut_i_arr))

    def change_status(new_stat, old_stat):
        if old_stat > 0:
            return old_stat

        return new_stat

    short = False
    status = 0
    sus_score = 0

    if len(cut_i_arr) < 400:
        printer.extended_write("SIGNAL TOO SHORT")
        status = change_status(3, status)
        short = True
        # return i_arr, nu_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short
        return nu_i_arr, nu_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short

    if len(cut_i_arr) < max_sig_len:
        printer.extended_write("NOT ENOUGH SIGNAL FOR ERROR LOCALIZATION")
        # status = change_status(3, status)
        short = True

    grad_ave_thresh = 2 * 10 ** (-12)
    # maybe 8.5e-12
    if grad_ave > 10 ** (-11):
        printer.extended_write("EXTREMELY HIGH GRADIENT AVERAGE")
        status = change_status(1, status)
    elif grad_ave > 3.5 * 10 ** (-12):
        printer.extended_write("INCREASING 50HZ")
        sus_score += 1

    if 3.5 * 10 ** (-11) < grad_rmsd < 2 * 10 ** (-10):
        printer.extended_write("SUSPICIOUS RMS")
        sus_score += 1
    elif grad_rmsd > 2 * 10 ** (-10):
        printer.extended_write("EXTREMELY HIGH RMS")
        status = change_status(1, status)

    if 3.5 * 10 ** (-9) < i_arr_sdev < 10 ** (-8):
        printer.extended_write("SUSPICIOUS SDEV")
        sus_score += 1
    elif i_arr_sdev > 10 ** (-8):  # TODO increase maybe??
        printer.extended_write("EXTREMELY HIGH SDEV")
        status = change_status(1, status)

    # sus: 1e-08 - 5e-09, bad < 5e-09
    if i_arr_ave < 1 * 10 ** (-14):  # 1.5e-09
        printer.extended_write("NO 50HZ DETECTED")
        status = change_status(1, status)
    elif i_arr_ave < 6 * 10 ** (-9):
        printer.extended_write("NOT ENOUGH 50HZ")
        status = change_status(2, status)
    elif i_arr_ave < 1.5 * 10 ** (-8):
        printer.extended_write("LOW 50HZ")
        sus_score += 1

    # TODO maybe dont do this?
    if sus_score >= 3:
        printer.extended_write("BAD SIGNAL")
        status = change_status(1, status)
    elif sus_score >= 2:
        printer.extended_write("SUSPICIOUS SIGNAL")
        status = change_status(2, status)

    # return i_arr, nu_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short
    return nu_i_arr, nu_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short


# TODO check that this works properly
def find_saturation_point_from_fft(i_x, i_arr, filter_i, fft_window, printer, sdev_window=10, rel_sdev_thresh=1.75,
                                   abs_sdev_thresh=1.35 * 10 ** (-10)):
    """analyses fft time series and looks for clear saturation points. does this by calculating a rolling standard deviation

    parameters:
        i_x: x values (in indices) of time series.
        i_arr: array containing fft time series.
        filter_i: integer. index returned by filter_start when run on original signal.
        fft_window: integer. window length for fft time series.
        printer: printer object. see file_handler.py
        sdev_window: integer. length of rolling sdev calculation.
        rel_sdev_thresh, abs_sdev_thresh: floats. relative threshold values for the sdev of the sdev time series.

    returns:
        rms_x: x values (in indices) of sdev time series.
        fft_sdev: sdev time series of fft time series.
        error_start: integer. start index of saturation. None if no saturation point found.
        sdev_thresh: float. absolute threshold value of sdev of sdev.
        sdev_span: span where sdev time series is above threshold."""
    if len(i_arr) == 0:
        return None, None, None, None, None

    x_start_i = np.where(i_x == filter_i)[0][0]
    fft_sdev, rms_x = hf.averaged_signal(i_arr[filter_i:], sdev_window, i_x[x_start_i:], mode=2)

    sdev_mean = np.mean(fft_sdev)
    sdev_sdev = np.std(fft_sdev)
    sdev_thresh = sdev_mean + rel_sdev_thresh * sdev_sdev
    where_above_sdev = np.where(fft_sdev > sdev_thresh)[0]
    if len(where_above_sdev) != 0:
        sdev_span = [rms_x[where_above_sdev[0]], rms_x[where_above_sdev[-1]]]
        span_sdev_ave = np.mean(fft_sdev[where_above_sdev[0]:where_above_sdev[-1]])
        seg_len = hf.length_of_segments([sdev_span])
        highsdev = span_sdev_ave > abs_sdev_thresh
        # ave_diff = span_sdev_ave - sdev_mean
        # highsdev = ave_diff > 2.5*10**(-11)
        span_start_i = where_above_sdev[0]
        printer.extended_write("span_sdev_ave", span_sdev_ave, "seg_len", seg_len, "span_start_i", span_start_i)
        # printer.extended_write("ave diff", ave_diff)
        local_err = 200 <= seg_len <= 500  # TODO increase upper bound?

        # ave diff 2.5-6e-11
        if highsdev and local_err and span_start_i > 6:
            printer.extended_write("saturation point found")
            error_start = sdev_span[0] + fft_window + sdev_window
        else:
            printer.extended_write("no saturation point found")
            error_start = None

        return rms_x, fft_sdev, error_start, sdev_thresh, sdev_span

    return rms_x, fft_sdev, None, sdev_thresh, None


def fft_filter(signal, filter_i, bad_segs, printer, dft_consts, fft_window=400, badness_sens=.5,
               debug=False, fft_cut=70, min_length=400):
    """calculates fft time series and returns segments containing errors. the segment
    is the entire analysed part of the signal unless a specific saturation point is found.

    parameters:
        signal: list or array containing signal.
        filter_i: integer. index returned by filter_start().
        bad_segs: list of bad segments in signal.
        printer: printer object. see file_handler.py.
        dft_consts: list of all constant elements of dft calculation.
        fft_window: window for fft time series calculation.
        badness_sens: float. relative length of signal that needs to be bad for the signal to be concidered too short.
        debug: boolean. if true, the function will return more data.
        fft_cut: integer. how many data points to remove from the end of the fft time series.
        min_length: integer. minimum required length of signal

    returns:
        list containing the marked segment.
        list containing the score for the segment.
        see docstrings for calc_fft_index_fast, stats_from_i and find_saturation_point_from_fft for details about variables
        returned in debug mode."""
    normal_sig = signal[filter_i:]
    sig_len = len(normal_sig)
    final_i_filtsignal = sig_len - 1
    final_i_fullsignal = len(signal) - 1

    if len(bad_segs) == 0:
        bad_len = 0
        good_len = sig_len - fft_window - fft_cut
    else:
        bad_start_i = bad_segs[0][0] - 2 * fft_cut
        bad_len = final_i_filtsignal - bad_start_i
        minus_i = final_i_filtsignal - bad_start_i + fft_window
        good_len = len(normal_sig[:-minus_i])

    rel_bad_len = bad_len / sig_len

    if rel_bad_len >= badness_sens or good_len < min_length:
        printer.extended_write("NOT ENOUGH SIGNAL FOR FFT")
        if debug:
            return None, None, None, None, None, None, None, None, 2, 0, None, None, None, None, None, None
        else:
            return [], []

    # i_arr, i_x, smooth_signal, smooth_x, detrended_sig = calc_fft_indices(normal_sig, printer, indices=indices, window=fft_window,
    # filter_offset=filter_i, goertzel=goertzel)

    i_arr, i_x, smooth_signal, smooth_x, detrended_sig = calc_fft_index_fast(normal_sig, printer, dft_consts,
                                                                             filter_offset=filter_i)

    if i_arr is None:
        if debug:
            return None, None, None, None, None, None, None, None, 2, 0, None, None, None, None, None, None
        else:
            return [], []

    # cut_i_arr, cut_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short = stats_from_i(
    #     i_arr[0], i_x, bad_segs, fft_window, printer, cut_length=fft_cut, lol=lol)

    cut_i_arr, cut_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short = stats_from_i(
        i_arr, i_x, bad_segs, fft_window, printer, cut_length=fft_cut)

    filter_i_i += filter_i

    if not short:
        rms_x, fft_sdev, error_start, sdev_thresh, sdev_span = find_saturation_point_from_fft(cut_i_x, cut_i_arr,
                                                                                              filter_i_i,
                                                                                              fft_window, printer)
    else:
        rms_x, fft_sdev, error_start, sdev_thresh, sdev_span = None, None, None, None, None

    if debug:
        return cut_i_x, cut_i_arr, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, rms_x, fft_sdev, error_start, sdev_thresh, sdev_span, detrended_sig

    return score_fft_segment(status, error_start, final_i_fullsignal, filter_i_i)


def score_fft_segment(status, error_start, final_i_fullsignal, filter_i_i):
    """translate status from stats_from_i to a score readable by separate_segments. also generate a readable
    segment.

    parameters:
        status: integer. status from stats_from_i.
        error_start: integer. error_start from find_saturation_point_from_fft.
        final_i_full_signal: integer. last index of original signal.
        filter_i_i: integer. index returned by filter_start when run on fft time series.

    returns:
        list containing marked segment.
        list containing score for marked segment."""
    if status == 0:
        score = -.5

    if status == 3:
        score = 0.0

    if status == 1:
        score = 1.5

    if status == 2:
        score = .5

    if error_start is not None:
        segments = [[error_start, final_i_fullsignal]]
        score = 1.5
    elif score > 0:
        segments = [[filter_i_i, final_i_fullsignal]]
    else:
        segments = []

    return segments, [score]


def combine_segments(segments):
    """combine several segments so that there is no overlap between them.

    parameters:
        segments: list of all segments to combine.

    returns:
        combined_segs: list of all combined segments."""
    n = len(segments)

    if n == 0:
        return []

    if n == 1:
        return segments

    segments_sorted = sorted(segments, key=itemgetter(0))

    combined_segs = []
    anchor_seg = segments_sorted[0]

    for i in range(1, n):
        segment = segments_sorted[i]

        if anchor_seg[1] < segment[0]:
            combined_segs.append(anchor_seg)
            anchor_seg = segment

        new_start = anchor_seg[0]
        new_end = max(anchor_seg[1], segment[1])

        anchor_seg = [new_start, new_end]

        if i == n - 1:
            combined_segs.append(anchor_seg)

    return combined_segs


def separate_segments(segments, confidences, conf_threshold=1):
    """sort segments into bad and suspicious based on their confidence values.

    parameters:
        segments: list of all segments to separate.
        confidences: list of confidences of segments.
        conf_threshold: float. threshold value for confidence values.

    returns:
        bad_segs: list of all bad segments.
        suspicious_segs: list of all suspicious segments."""
    n = len(segments)

    bad_segs = []
    suspicious_segs = []

    for i in range(n):
        conf = confidences[i]
        segment = segments[i]

        if conf >= conf_threshold:
            bad_segs.append(segment)
        elif conf >= 0:
            suspicious_segs.append(segment)

    return bad_segs, suspicious_segs


def fix_overlap(bad_segs, suspicious_segs):
    """fix overlap between suspicious and bad segments. bad segments take priority
    over suspicious ones. only returns the suspicious segments since bad segments stay unchanged.

    parameters:
        bad_segs: list containing bad segments.
        suspicious_segs: list containing suspicious segments.

    returns:
        new_suspicious_segs: list of fixed suspicious segments."""
    if len(bad_segs) == 0 or len(suspicious_segs) == 0:
        return suspicious_segs

    new_suspicious_segs = []
    for sus_seg in suspicious_segs:
        sus_list = list(range(sus_seg[0], sus_seg[1] + 1))
        for bad_seg in bad_segs:
            bad_list = list(range(bad_seg[0], bad_seg[1] + 1))
            sus_list = list(set(sus_list) - set(bad_list))

        split_lists = hf.split_into_lists(sus_list)
        split_segs = []

        for lst in split_lists:
            split_segs.append([np.amin(lst), np.amax(lst)])

        new_suspicious_segs += split_segs

    return new_suspicious_segs


def final_analysis(segments, confidences):
    """take all segments and their confidences and separate segments into good,
    suspicious and bad, as well as fix the overlap between them.

    parameters:
        segments: list of segments.
        confidences: list of scores for segments.

    returns:
        bad_segs: list of bad segments.
        suspicious_segs: list of suspicious segments."""
    bad_segs, suspicious_segs = separate_segments(segments, confidences)

    bad_segs = combine_segments(bad_segs)
    suspicious_segs = combine_segments(suspicious_segs)

    suspicious_segs = fix_overlap(bad_segs, suspicious_segs)

    return bad_segs, suspicious_segs


def analyse_all_neo(signals, names, chan_num, printer=None,
                    filters=None,
                    filter_beginning=True):
    """go through all signals and determine suspicious and bad segments within them.
    this is done by running the signal through three different filters
    (spike_filter_neo, flat_filter_neo, uniq_filter_neo and fft_filter).
    the function returns lists containing all bad and suspicious segments
    as well as ones containing whether the signal is bad (boolean value) and
    the time it took to analyse each signal.

    parameters:
        signals: list containing all signals.
        names: list containing names of signals/detectors.
        chan_num: integer. total number of analysed signals/detectors.
        printer: printer object. see file_handler.py.
        filters: list of strings. contains all FDFs to use.
        filter_beginning: boolean. if true, filters the start of each signal with filter_start.

    returns:
        signal_statuses: list of booleans. signal_statuses[i] is True if signals[i] is bad.
        bad_segment_list: list containing bad segments.
        suspicious_segment_list: list containing suspicious segments.
        exec_times: list containing the execution times of each signal.
    """
    if filters is None:
        filters = ["uniq", "flat", "spike", "fft"]

    if printer is None:
        import file_handler as fh
        printer = fh.Printer("none")

    # move fft filter to last place (requires other bad segments as input)
    if "fft" in filters:
        dft_consts = calc_dft_constants(2, 400)
        filters.append(filters.pop(filters.index("fft")))

    # exec_times = np.empty(chan_num)
    # signal_statuses = np.empty(chan_num)
    # bad_segment_list = np.empty(chan_num)
    # suspicious_segment_list = np.empty(chan_num)
    # bad_segment_list = [None]*chan_num
    # suspicious_segment_list = [None]*chan_num

    exec_times = []
    signal_statuses = []
    bad_segment_list = []
    suspicious_segment_list = []

    for i in range(chan_num):
        printer.extended_write(names[i])
        signal = signals[i]
        signal_length = len(signal)
        segments = []
        confidences = []
        # bad = False

        start_time = time.time()

        if filter_beginning:
            filter_i = filter_start(signal)
        else:
            filter_i = 0

        for fltr in filters:
            printer.extended_write("beginning analysis with " + fltr + " filter")

            if fltr == "uniq":
                seg_is, confs = uniq_filter_neo(signal, filter_i)

            if fltr == "flat":
                seg_is, confs = flat_filter(signal, printer)

            if fltr == "spike":
                seg_is, confs = spike_filter_neo(signal, filter_i, printer)

            if fltr == "fft":
                temp_bad_segs, temp_suspicious_segs = final_analysis(segments, confidences)
                seg_is, confs = fft_filter(signal, filter_i, temp_bad_segs, printer, dft_consts)

            new_segs = len(seg_is)

            if new_segs == 0:
                printer.extended_write("no segments found")
            else:
                printer.extended_write(new_segs, "segment(s) found:")

                for seg in seg_is:
                    printer.extended_write(seg)

            segments += seg_is
            confidences += confs

        bad_segs, suspicious_segs = final_analysis(segments, confidences)
        num_bad = len(bad_segs)
        bad = num_bad > 0
        num_sus = len(suspicious_segs)
        printer.extended_write(num_sus, "suspicious and", num_bad, " bad segment(s) found in total")

        if not bad:
            printer.extended_write("no bad segments found")

        signal_statuses.append(bad)
        bad_segment_list.append(bad_segs)
        suspicious_segment_list.append(suspicious_segs)

        # signal_statuses[i] = bad
        # bad_segment_list[i] = bad_segs
        # suspicious_segment_list[i] = suspicious_segs

        end_time = time.time()
        exec_time = end_time - start_time
        printer.extended_write("execution time:", exec_time)
        exec_times.append(exec_time)
        # exec_times[i] = exec_time

        printer.extended_write()

    return signal_statuses, bad_segment_list, suspicious_segment_list, exec_times
