import dataio as dat
import helper_funcs as hf

time_window = (0.210, 0.50)

def write_data_compact(fname, data, col_names, divider=";\t"):
    """write data into a file separated by the string in divider. data must
    be a 2d array, the first axis containing the different columns you want to write,
    and the second containing the data value for each detector. the names in
    col_names are written in the header.

    parameters:
        fname: string. name of file to write to.
        data: list of lists containing data to write.
        col_names: list of strings. names of each column.
        divider: string. divider between data entries."""
    f = open(fname, "w")
    header = ""

    for name in col_names:
        if header == "":
            header += name
        else:
            header += divider + name

    f.write(header + "\n")

    n_points = len(data[0])

    for i in range(n_points):
        line = ""

        for data_arr in data:
            data_value = data_arr[i]
            if line == "":
                line += str(data_value)
            else:
                line += divider + str(data_value)

        f.write(line + "\n")

    f.close()

def load_all(fname):
    """load all data from npz file.

    parameters:
        fname: string. name of npz file to read.

    returns:
        array with all data in file."""
    return dat.MultiChannelData.load_npz(fname)

def get_signals(fname, channels=["MEG*1", "MEG*4"]):
    """load data from file and reformat to individual arrays.

    parameters:
        fname: string. name of npz file to read.
        channels: list of strings. channels to return. allows wildcard characters.

    returns:
        signals: list containing signals.
        names: list containing names of channels.
        time: list containing x values (in seconds) for all signals.
        n_chans: integer. number of channels read."""
    fname_full = fname
    data = load_all(fname_full).subpool(channels).clip(time_window)
    unorganized_signals = data.data
    names = data.names
    n_chan = data.n_channels
    time = data.time
    signals = hf.reorganize_signals(unorganized_signals, n_chan)

    return signals, names, time, n_chan

class Printer:
    """an alternate to the intrinsic print function. prints to stdout, file
    or nowhere depending on mode."""
    def __init__(self, write_mode, file=None):
        """initialize object.

        parameters:
            write_mode: string. determines where to write. print if to stdout, file if to file and none if to nowhere.
            file: string. name of file to write to"""
        self.mode = write_mode
        self.f = file

    def extended_write(self, *args, additional_mode=""):
        """replaces standard print function, and is called similarly. writes to output determined by mode.

        parameters:
            args: list of strings. objects to write.
            additional_mode: string. the input of args will be written to another output defined by this parameter.
            allows same arguments as self.mode."""
        mode = self.mode + additional_mode
        f = self.f

        if mode == "none":
            return

        txt = ""

        for i in range(len(args)):
            arg = args[i]
            if not isinstance(arg, str):
                arg = str(arg)

            if i == 0:
                txt += arg
            else:
                txt += " " + arg

        if "print" in mode:
            print(txt)

        if "file" in mode and f is not None:
            f.write(txt + "\n")

    def close(self):
        """close file if file exists"""
        if "file" in self.mode:
            self.f.close()
