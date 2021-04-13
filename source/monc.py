"""
Utility specific to working with MONC data
"""
import numpy as np
from netCDF4 import Dataset


def _file_key(file):
    f1 = file.split("_")[-1]
    f2 = f1.split(".")[0]
    return float(f2)


def find_time_in_files(files, ref_time, nodt=False):
    r"""
    Function to find file containing data at required time.
        Assumes file names are of form \*_tt.0\* where tt is model output time.

    Args:
        files: ordered list of files
        ref_time: required time.
        nodt: if True do not look for next time to get delta_t

    Returns:
        Variables defining location of data in file list::

            ref_file: Index of file containing required time in files.
            it: Index of time in dataset.
            delta_t: Interval between data.

    @author: Peter Clark

    """

    file_times = np.zeros(len(files))
    for i, file in enumerate(files):
        file_times[i] = _file_key(file)

    def get_file_times(dataset):
        theta = dataset.variables["th"]
        t = dataset.variables[theta.dimensions[0]][...]
        return t

    delta_t = 0.0
    it = -1
    ref_file = np.where(file_times >= ref_time)[0][0]
    while True:
        if ref_file >= len(files) or ref_file < 0:
            ref_file = None
            break
        dataset = Dataset(files[ref_file])
        print("Dataset opened ", files[ref_file])
        times = get_file_times(dataset)

        if len(times) == 1:
            dataset.close()
            print("dataset closed")
            it = 0
            if times[it] != ref_time:
                print(
                    "Could not find exact time {} in file {}".format(
                        ref_time, files[ref_file]
                    )
                )
                ref_file = None
            else:
                if nodt:
                    delta_t = 0.0
                else:
                    print("Looking in next file to get dt.")
                    dataset_next = Dataset(files[ref_file + 1])
                    print("Dataset_next opened ", files[ref_file + 1])
                    times_next = get_file_times(dataset_next)
                    delta_t = times_next[0] - times[0]
                    dataset_next.close()
                    print("dataset_next closed")
            break

        else:  # len(times) > 1
            it = np.where(times == ref_time)[0]
            if len(it) == 0:
                print(
                    "Could not find exact time {} in file {}".format(ref_time, ref_file)
                )
                it = np.where(times >= ref_time)[0]
                #                print("it={}".format(it))
                if len(it) == 0:
                    print(
                        "Could not find time >= {} in file {}, looking in next.".format(
                            ref_time, ref_file
                        )
                    )
                    ref_file += 1
                    continue
            #            else :
            it = it[0]
            if it == (len(times) - 1):
                delta_t = times[it] - times[it - 1]
            else:
                delta_t = times[it + 1] - times[it]
            break
    print(
        "Looking for time {}, returning file #{}, index {}, time {}, delta_t {}".format(
            ref_time, ref_file, it, times[it], delta_t
        )
    )
    return ref_file, it, delta_t.astype(int)
