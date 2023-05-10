import math
import os
import time
#import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np

import helper_funcs as hf

def magn_from_point(a_pinv, point):
    """calculates the lengths of the magnetic field vector from a set of signal magnitudes (point)
    and detectors direction vectors (a) using pseudoinverse"""
    # a_pinv = np.linalg.pinv(a)
    magn = a_pinv.dot(point)
    return magn


# cpus = os.cpu_count()
# pool = Pool(cpus)

def calc_magn_field_from_signals(signals, xs, vectors, printer, ave_window=400):
    """calculates a magnetic field vector as a function of time from a set of signals
    and their vectors. the magnetic fields are calculated from averaged
    signals. averaging window can be changed using ave_window"""
    # crop signals if needed
    if len(xs) == 1:
        cropped_signals = signals
        new_x = xs[0]
    else:
        cropped_signals, new_x = hf.crop_all_sigs(signals, xs, [])

    parallel = False

    averaged_sigs = []
    new_is = []

    for signal in cropped_signals:
        ave_sig, new_i = hf.averaged_signal(signal, ave_window, x=new_x)
        averaged_sigs.append(ave_sig)
        new_is.append(new_i)

    # if there are less than 3 signals, the calculation is not performed
    if len(signals) < 3:
        printer.extended_write("not enough signals to calculate magnetic field vector")
        return [], [], cropped_signals, new_x, averaged_sigs

    def mag_and_i(i):
        points = []
        for j in range(len(averaged_sigs)):
            points.append(averaged_sigs[j][i])

        vector_pinv = np.linalg.pinv(vectors)
        mag = magn_from_point(vector_pinv, points)
        # magn_vectors.append(mag)
        # magn_vectors[i] = mag
        current_index = new_is[0][i]
        # mag_is.append(current_index)
       #  mag_is[i] = current_index

        return mag, current_index

    n = len(averaged_sigs[0])
    # magn_vectors = []
    magn_vectors = np.zeros((n, 3))
    # mag_is = []
    mag_is = np.zeros(n, dtype=int)
    for i in range(len(averaged_sigs[0])):
        mg, ind = mag_and_i(i)
        magn_vectors[i] = mg
        mag_is[i] = ind


    #print(magn_vectors)

    return magn_vectors, mag_is, cropped_signals, new_x, averaged_sigs


def reconstruct(mag, v):
    """reconstruct a signal using a magnetic vector (as a function of time) mag
    and a detector direction vector v"""
    # rec_sig = []
    # for mag_point in mag:
    #     rec_sig.append(np.dot(mag_point, v))

    rec_sig = np.dot(mag, v)

    return rec_sig


def rec_and_diff(signals, xs, vs, printer, ave_window=1):
    """using pseudoinverse, reconstruct a set of signals and calculate their difference
    to the original signals. returns average total difference for each signal,
    average total difference for all signals in total, the reconstructed and cropped
    signals and their x values. increasing ave_window decreases calculation time
    and accuracy"""
    if len(signals) == 0:
        return None, None, None, None, None, None, None

    # calculate magnetic field vector
    magn_vectors, mag_is, cropped_signals, new_x, averaged_signals = calc_magn_field_from_signals(signals, xs, vs,
                                                                                                  printer,
                                                                                                  ave_window=ave_window)

    if len(magn_vectors) == 0:
        return None, None, None, None, None, None, None

    rec_sigs = []
    aves = []
    all_diffs = []
    for i in range(len(cropped_signals)):
        # calculate reconstructed signal
        rec_sig = reconstruct(magn_vectors, vs[i])
        rec_sigs.append(rec_sig)

        # calculate difference between original and reconstructed signals
        diffs, diff_x = hf.calc_diff(averaged_signals[i], rec_sig, mag_is, mag_is)

        # calculate averages
        ave = np.mean(diffs)
        aves.append(ave)
        all_diffs.append(diffs)

    ave_of_aves = np.mean(aves)

    return ave_of_aves, aves, all_diffs, rec_sigs, mag_is, averaged_signals, mag_is  # cropped_signals, new_x


def filter_unphysical_sigs(signals, names, xs, vs, bad_segs, sus_segs, printer, ave_sens=10 ** (-13), ave_window=1,
                           min_sigs=4):
    """from a set of signals, systematically  remove the most unphysical ones
    until a certain accuracy is reached. this is done by reconstructing
    optimal magnetic field vectors using pseudoinverse and calculating the total
    average difference between original signals and the reconstructed signals.
    when the total average goes below ave_sens, the calculation is stopped.
    reconstucted magnetic fields can be averaged for faster performance, but
    ave_sens MUST ALSO BE CHANGED for accurate calculation. returns the names
    of removed signals, the new x-values of the cropped signals (in indices)
    as well as the absolute and relative change in the average total diference
    at each removal
    the following ave_sens ave_window pairs are confirmed to produce good results:
    10**(-13) 1
    10**(-12) 100
    5*10**(-13) 100 CHECK THIS"""
    if len(signals) <= min_sigs:
        printer.extended_write("too few signals, stopping")
        return [], [], [], []

    # find the index of the channel to exclude. favors channels with suspicious
    # channels present that have an average below ave_sens
    # TODO make this more general possibly
    def exclude(averages, sus_list):
        sus_present = False

        for sus in sus_list:

            if len(sus) != 0:
                sus_present = True
                break

        if not sus_present:
            best = np.amin(averages)
            return best, averages.index(best)

        best_aves = []

        for j in range(len(sus_list)):
            sus = sus_list[j]
            ave = averages[j]

            if len(sus) != 0 and ave < ave_sens:
                best_aves.append(ave)

        if len(best_aves) == 0:
            best = np.amin(averages)
        else:
            best = np.amin(best_aves)

        return best, averages.index(best)

    # crop signals
    cropped_signals, new_x = hf.crop_all_sigs(signals, xs, bad_segs)

    printer.extended_write("analysing " + str(len(cropped_signals)) + " signals")
    temp_sigs = cropped_signals[:]
    temp_names = names[:]
    temp_vs = vs[:]
    temp_sus = sus_segs[:]

    # calculate initial reconstruction
    ave_of_aves, aves, diffs, rec_sigs, magis, cropped_signals, new_new_x = rec_and_diff(cropped_signals, [new_x], vs,
                                                                                         printer,
                                                                                         ave_window=ave_window)
    printer.extended_write("average at start:", ave_of_aves)

    if ave_of_aves < ave_sens:
        printer.extended_write("no optimisation needed, stopping")
        return [], new_x, [], []

    excludes = []
    ave_diffs = []
    rel_diffs = []
    while ave_of_aves > ave_sens:

        printer.extended_write(len(temp_sigs), "signals left")

        if len(temp_sigs) <= min_sigs:
            printer.extended_write("no optimal magnetic field found")
            return [], new_x, [], []

        new_aves = []
        # calculate a new reconstruction excluding each signal one at a time
        for i in range(len(temp_sigs)):
            excl_name = temp_names[i]

            if excl_name in excludes:
                new_aves.append(100000)
                printer.extended_write("skipping", i)
                continue

            sigs_without = temp_sigs[:i] + temp_sigs[i + 1:]
            vs_without = temp_vs[:i] + temp_vs[i + 1:]

            # calculate reconstruction and average difference
            new_ave_of_aves, temp_aves, temp_diffs, temp_rec_sigs, temp_magis, temp_crop_sigs, temp_new_x = rec_and_diff(
                sigs_without, [new_x], vs_without, printer, ave_window=ave_window)
            new_aves.append(new_ave_of_aves)

        # choose the lowest average difference and permanently exclude this signal
        # from the rest of the calculation
        best_ave, best_exclusion_i = exclude(new_aves, temp_sus)
        diff = ave_of_aves - best_ave
        rel_diff = diff / ave_of_aves
        printer.extended_write("average", best_ave)
        ave_diffs.append(diff)
        rel_diffs.append(rel_diff)
        ex_nam = temp_names[best_exclusion_i]
        printer.extended_write(ex_nam + " excluded")
        excludes.append(ex_nam)
        temp_vs.pop(best_exclusion_i)
        temp_sigs.pop(best_exclusion_i)
        temp_names.pop(best_exclusion_i)
        temp_sus.pop(best_exclusion_i)
        ave_of_aves = best_ave

    printer.extended_write(len(temp_names), "signals left at the end of calculation")
    return excludes, new_x, ave_diffs, rel_diffs


def check_all_phys_alt(signals, detecs, names, n_chan, bad_seg_list, sus_seg_list, printer, smooth_window=401,
                   ave_window=1, ave_sens=10 ** (-13), smooth_only=False):
    """goes through all detectors, does the filter_unphysical_sigs calculation for a
    signal cluster containing a given signal and all neighbouring signals and
    logs how many times each signal has been excluded and how many times it
    has been used in a filter_unphysical_sigs calculation. segments of the signals
    previously determined to be bad must also be included; signals with bad segments
    are removed.
    the calculation is done on the gradients of the smoothed and filtered signals.
    returns a dictionary containing the names of the detectors as well as
    a dictionary of all the calculations it was included in and whether
    it was excluded. also returns the absolute and relative improvement
    each signal's exclusion caused to the average total difference."""

    # import helper_funcs_cython as hfc

    offset = int(smooth_window / 2)
    alt = True

    # initalise dictionaries
    all_diffs = {}
    all_rel_diffs = {}
    chan_dict = {}

    print(detecs)

    for name in names:
        all_diffs[name] = []
        all_rel_diffs[name] = []
        chan_dict[name] = {}

    # good_sigs, good_names, good_sus, good_bad = hf.list_good_sigs(names, signals, bad_seg_list)

    if alt:
        signals, all_xs = hf.filter_and_smooth_and_gradient_all(signals, offset, smooth_window, smooth_only=smooth_only)

    def loop_func(k, all_diffs, all_rel_diffs, chan_dict, detecs):
        # choose central detector and find nearby detectors

        print(detecs)
        comp_detec = names[k]
        printer.extended_write(comp_detec)

        # start_time = time.time()
        nearby_names = hf.find_nearby_detectors(comp_detec, detecs, names)
        # nearby_names = []
        # nearby_names = hfc.find_nearby_detectors(comp_detec, nearby_names, detecs, names)
        # end_time = time.time()
        nearby_names.append(comp_detec)

        # exclude signals whose bad segments are too long
        # start_time = time.time()
        new_near = []
        for nam in nearby_names:
            index = names.index(nam)
            sig_len = len(signals[index])
            bad_segs = bad_seg_list[index]
            bad = len(bad_segs) != 0

            if bad:
                printer.extended_write("excluding " + nam + " from calculation due to presence of bad segments")
                continue

            if sig_len < smooth_window:
                printer.extended_write("excluding " + nam + " from calculation due to shortness")
                continue

            new_near.append(nam)

        if len(new_near) == 0:
            printer.extended_write("no signals in group\n")
            return


        # end_time = time.time()

        near_vs = []
        near_rs = []

        for name in new_near:
            near_vs.append(detecs[name][:3, 2])
            near_rs.append(detecs[name][:3, 3])

        # start_time = time.time()
        near_sigs = hf.find_signals(new_near, signals, names)
        near_bad_segs = hf.find_signals(new_near, bad_seg_list, names)
        near_sus_segs = hf.find_signals(new_near, sus_seg_list, names)
        # end_time = time.time()
        # print("time to ahihfsahj", end_time - start_time)

        # start_time = time.time()
        smooth_sigs = []
        xs = []

        # filter each signal, smooth and calculate gradient of smoothed and
        # filtered signals
        if not alt:
            for i in range(len(near_sigs)):
                signal = near_sigs[i]
                filtered_signal, x, smooth_signal, smooth_x, new_smooth = hf.filter_and_smooth(signal, offset,
                                                                                               smooth_window,
                                                                                               smooth_only=smooth_only)
                smooth_sigs.append(np.gradient(new_smooth))
                xs.append(x)
        else:
            smooth_sigs = near_sigs
            xs = hf.find_signals(new_near, all_xs, names)

        # end_time = time.time()
        # print("time to filter and smooth", end_time - start_time)

        # start_time = time.time()
        # calculate which signals in the cluster to exclude
        exclude_chans, new_x, diffs, rel_diffs = filter_unphysical_sigs(smooth_sigs, new_near, xs, near_vs,
                                                                        near_bad_segs, near_sus_segs, printer,
                                                                        ave_window=ave_window,
                                                                        ave_sens=ave_sens)

        # end_time = time.time()
        # print("time to exclude", end_time - start_time)

        # start_time = time.time()
        if len(new_x) != 0:
            for nam in new_near:
                chan_dict[nam][comp_detec] = 0

        if len(new_x) > 1:
            printer.extended_write("tested: ", new_near)
            printer.extended_write("analysed segment between", new_x[0], new_x[len(new_x) - 1])

        printer.extended_write("excluded", exclude_chans)

        # log and print data
        for j in range(len(exclude_chans)):
            chan = exclude_chans[j]
            diff = diffs[j]
            rel_diff = rel_diffs[j]
            all_diffs[chan].append(diff)
            all_rel_diffs[chan].append(rel_diff)
            chan_dict[chan][comp_detec] += 1

            chan_dat = chan_dict[chan]
            tot = np.float64(len(chan_dat))
            ex = np.float64(len([x for x in chan_dat if chan_dat[x] == 1]))

            printer.extended_write(chan, "times excluded:", ex, ", times in calculation:", tot,
                                   ", fraction excluded:", float(ex / tot),
                                   "average relative difference:", np.mean(all_rel_diffs[chan]))

        # end_time = time.time()
        # print("time to get stats", end_time - start_time)
        printer.extended_write()

    # pool.map(loop_func, list(range(n_chan)), all_diffs, all_rel_diffs, chan_dict, np.full(n_chan, detecs))

    return all_diffs, all_rel_diffs, chan_dict


# TODO lower ave_sens
def check_all_phys(signals, detecs, names, n_chan, bad_seg_list, sus_seg_list, printer, smooth_window=401,
                   ave_window=1, ave_sens=10 ** (-13), smooth_only=False):
    """goes through all detectors, does the filter_unphysical_sigs calculation for a
    signal cluster containing a given signal and all neighbouring signals and
    logs how many times each signal has been excluded and how many times it
    has been used in a filter_unphysical_sigs calculation. segments of the signals
    previously determined to be bad must also be included; signals with bad segments
    are removed.
    the calculation is done on the gradients of the smoothed and filtered signals.
    returns a dictionary containing the names of the detectors as well as
    a dictionary of all the calculations it was included in and whether
    it was excluded. also returns the absolute and relative improvement
    each signal's exclusion caused to the average total difference."""

    # import helper_funcs_cython as hfc

    offset = int(smooth_window / 2)
    alt = True

    # initalise dictionaries
    all_diffs = {}
    all_rel_diffs = {}
    chan_dict = {}

    tested_groups = []

    for name in names:
        all_diffs[name] = []
        all_rel_diffs[name] = []
        chan_dict[name] = {}

    # good_sigs, good_names, good_sus, good_bad = hf.list_good_sigs(names, signals, bad_seg_list)

    if alt:
        signals, all_xs = hf.filter_and_smooth_and_gradient_all(signals, offset, smooth_window, smooth_only=smooth_only)

    for k in range(n_chan):
        # choose central detector and find nearby detectors

        comp_detec = names[k]
        printer.extended_write(comp_detec)

        # start_time = time.time()
        nearby_names = hf.find_nearby_detectors(comp_detec, detecs, names)
        # nearby_names = []
        # nearby_names = hfc.find_nearby_detectors(comp_detec, nearby_names, detecs, names)
        # end_time = time.time()
        nearby_names.append(comp_detec)

        # exclude signals whose bad segments are too long
        # start_time = time.time()
        new_near = []
        for nam in nearby_names:
            index = names.index(nam)
            sig_len = len(signals[index])
            bad_segs = bad_seg_list[index]
            bad = len(bad_segs) != 0

            if bad:
                printer.extended_write("excluding " + nam + " from calculation due to presence of bad segments")
                continue

            if sig_len < smooth_window:
                printer.extended_write("excluding " + nam + " from calculation due to shortness")
                continue

            new_near.append(nam)

        if len(new_near) == 0:
            printer.extended_write("no signals in group\n")
            continue

        if new_near in tested_groups:
            printer.extended_write("already calculated, skipping\n")
            continue

        # end_time = time.time()

        tested_groups.append(new_near)

        near_vs = []
        near_rs = []

        for name in new_near:
            near_vs.append(detecs[name][:3, 2])
            near_rs.append(detecs[name][:3, 3])

        # start_time = time.time()
        near_sigs = hf.find_signals(new_near, signals, names)
        near_bad_segs = hf.find_signals(new_near, bad_seg_list, names)
        near_sus_segs = hf.find_signals(new_near, sus_seg_list, names)
        # end_time = time.time()
        # print("time to ahihfsahj", end_time - start_time)

        # start_time = time.time()
        smooth_sigs = []
        xs = []

        # filter each signal, smooth and calculate gradient of smoothed and
        # filtered signals
        if not alt:
            for i in range(len(near_sigs)):
                signal = near_sigs[i]
                filtered_signal, x, smooth_signal, smooth_x, new_smooth = hf.filter_and_smooth(signal, offset,
                                                                                               smooth_window,
                                                                                               smooth_only=smooth_only)
                smooth_sigs.append(np.gradient(new_smooth))
                xs.append(x)
        else:
            smooth_sigs = near_sigs
            xs = hf.find_signals(new_near, all_xs, names)

        # end_time = time.time()
        # print("time to filter and smooth", end_time - start_time)

        # start_time = time.time()
        # calculate which signals in the cluster to exclude
        exclude_chans, new_x, diffs, rel_diffs = filter_unphysical_sigs(smooth_sigs, new_near, xs, near_vs,
                                                                        near_bad_segs, near_sus_segs, printer,
                                                                        ave_window=ave_window,
                                                                        ave_sens=ave_sens)

        # end_time = time.time()
        # print("time to exclude", end_time - start_time)

        # start_time = time.time()
        if len(new_x) != 0:
            for nam in new_near:
                chan_dict[nam][comp_detec] = 0

        if len(new_x) > 1:
            printer.extended_write("tested: ", new_near)
            printer.extended_write("analysed segment between", new_x[0], new_x[len(new_x) - 1])

        printer.extended_write("excluded", exclude_chans)

        # log and print data
        for j in range(len(exclude_chans)):
            chan = exclude_chans[j]
            diff = diffs[j]
            rel_diff = rel_diffs[j]
            all_diffs[chan].append(diff)
            all_rel_diffs[chan].append(rel_diff)
            chan_dict[chan][comp_detec] += 1

            chan_dat = chan_dict[chan]
            tot = np.float64(len(chan_dat))
            ex = np.float64(len([x for x in chan_dat if chan_dat[x] == 1]))

            printer.extended_write(chan, "times excluded:", ex, ", times in calculation:", tot,
                                   ", fraction excluded:", float(ex / tot),
                                   "average relative difference:", np.mean(all_rel_diffs[chan]))

        # end_time = time.time()
        # print("time to get stats", end_time - start_time)
        printer.extended_write()

    return all_diffs, all_rel_diffs, chan_dict


def analyse_phys_dat(all_diffs, names, all_rel_diffs, chan_dict, frac_w=2.5,
                     diff_w=.5, num_w=.5, unphys_sensitivity=.3, conf_sens=.55,
                     min_chans=5, uncert_chans=7):
    """NOT USED
    analyse the data calculated by check_all_phys. if a signal has been excluded
    from a significant enough fraction (unphys_sensitivity),
    of all filter_unphysical_sigs calculations, it is marked as unphysical.
    if it has been included in too few calculations, it is marked as
    undetermined. if it has been included in no calculations,
    it is marked as unused. a confidence value is calculated for each marking;
    a higher value means a higher chance of the marking being correct.
    if a signal is marked as physical or unphysical but the confidence is below
    conf_sens, the signal is marked as undetermined.
    the confidence value is determined by the fraction of exclusions, number
    of times it has been included in a calculation and the differences caused
    by a signal's exclusion
    0 = physical
    1 = unphysical
    2 = undetermined
    3 = unused"""
    status = []
    confidence = []

    for i in range(len(names)):
        name = names[i]
        rel_diffs = all_rel_diffs[name]
        # print(name)
        diffs = all_diffs[name]
        chan_dat = chan_dict[name]

        tot = np.float64(len(chan_dat))

        # mark signal that has been in a calculation too few times
        if tot == 0:
            status.append(3)
            confidence.append(3)
            continue

        if tot < min_chans:
            status.append(2)
            confidence.append(3)
            continue

        # determine weight of the total number of times the signal has been in calculation
        if tot < uncert_chans:
            num_w2 = - uncert_chans / tot
        else:
            num_w2 = tot / 14

        num_conf = num_w * num_w2

        ex = np.float64(len([x for x in chan_dat if chan_dat[x] == 1]))
        frac_excluded = ex / tot

        ave_diff = np.mean(rel_diffs)

        # calculate confidence and mark
        if frac_excluded > unphys_sensitivity:
            stat = 1
            diff_conf = diff_w * ave_diff
            frac_conf = 1.25 * frac_w * frac_excluded
        else:
            stat = 0

            if math.isnan(ave_diff):
                diff_conf = diff_w
            else:
                diff_conf = diff_w * (1 - ave_diff)

            frac_conf = frac_w * (1 - frac_excluded / (2.5 * unphys_sensitivity))

        conf = (diff_conf + frac_conf + num_conf) / (diff_w + frac_w + num_w)

        if conf < conf_sens:
            stat = 2

        confidence.append(conf)
        status.append(stat)

    return status, confidence


def analyse_phys_dat_alt(all_diffs, names, all_rel_diffs, chan_dict):
    """same as analyse_phys_dat but less confusing
    0 = physical
    1 = unphysical
    2 = undetermined
    3 = unused"""
    status = []
    confidence = []

    for i in range(len(names)):
        name = names[i]
        rel_diffs = all_rel_diffs[name]
        # print(name)
        diffs = all_diffs[name]
        chan_dat = chan_dict[name]

        tot = np.float64(len(chan_dat))

        if tot == 0:
            status.append(3)
            confidence.append(3)
            continue

        ex = np.float64(len([x for x in chan_dat if chan_dat[x] == 1]))
        frac_excluded = ex / tot

        if .6 <= frac_excluded:
            status.append(1)
        elif frac_excluded <= .4:
            status.append(0)
        else:
            status.append(2)

        # confidence.append(tot/14)
        confidence.append(frac_excluded)

    return status, confidence
