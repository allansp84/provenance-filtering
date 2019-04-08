# -*- coding: utf-8 -*-

import sys
import numpy


def read_array_file(filepath):
    """ Reads an ASCII array file.

    An NxM array has the file formatted as:
    <M> <t>
    <a_0_1> <a_0_2> ... <a_0_M-1>
    <a_1_1> <a_1_2> ... <a_1_M-1>
    .
    .
    .
    <a_N-1_1> <a_N-1_2> ... <a_N-1_s-1>

    With <M> <t> being a header that describes number of columns and
    array type, respectively. The type codes are:
    0: uint8
    3: int32
    5: float32
    7: float16
    9: float64

    :param filepath: output file name and path;
    :return: read array;
    """

    try:
        f = open(filepath, 'r')
        line = f.readline().strip("\n")
        rows, cols, dtype = [int(x) for x in line.split()]
        f.close()

        if dtype == 0:
            inarray = numpy.loadtxt(filepath, dtype=numpy.uint8, skiprows=1)
        elif dtype == 5:
            inarray = numpy.loadtxt(filepath, dtype=numpy.float32, skiprows=1)
        elif dtype == 7:
            inarray = numpy.loadtxt(filepath, dtype=numpy.float16, skiprows=1)
        elif dtype == 3:
            inarray = numpy.loadtxt(filepath, dtype=numpy.int32, skiprows=1)
        elif dtype == 9:
            inarray = numpy.loadtxt(filepath, dtype=numpy.float64, skiprows=1)
        else:
            raise ValueError

        inarray = numpy.resize(inarray, (rows, cols))

        return inarray

    except OSError:
        sys.stderr.write("Failure to read array file: {0:s}!\n".format(filepath))
        return None

    except ValueError:
        sys.stderr.write("Unknown type formating in input array {0:s}!\n".format(filepath))
        return None


def write_array_file(filepath, array_list, outfmt='%.12e'):
    """ Writes an ASCII array file.

    An NxM array has the file formatted as:
    <M> <t>
    <a_0_1> <a_0_2> ... <a_0_M-1>
    <a_1_1> <a_1_2> ... <a_1_M-1>
    .
    .
    .
    <a_N-1_1> <a_N-1_2> ... <a_N-1_s-1>

    With <M> <t> being a header that describes number of columns and
    array type, respectively. The type codes are:
    0: uint8
    3: int32
    5: float32
    7: float16
    9: float64

    :param filepath: output file name and path;
    :param array_list: either a list of arrays or a single array. The list is vertically stacked before being saved;
    :param outfmt: a format string for the output number formatting;
    :return: True if writing as successful, False otherwise.
    """

    outarray = numpy.vstack(array_list)

    try:
        if outarray.dtype.name == 'uint8':
            hd = "{0:d} {1:d} {2:d}".format(outarray.shape[0], outarray.shape[1], 0)
            numpy.savetxt(filepath, outarray, header=hd, comments='', fmt='%d')
        elif outarray.dtype.name == 'float32':
            hd = "{0:d} {1:d} {2:d}".format(outarray.shape[0], outarray.shape[1], 5)
            numpy.savetxt(filepath, outarray, header=hd, comments='', fmt=outfmt)
        elif outarray.dtype.name == 'float16':
            hd = "{0:d} {1:d} {2:d}".format(outarray.shape[0], outarray.shape[1], 7)
            numpy.savetxt(filepath, outarray, header=hd, comments='', fmt=outfmt)
        elif outarray.dtype.name == 'float64':
            hd = "{0:d} {1:d} {2:d}".format(outarray.shape[0], outarray.shape[1], 9)
            numpy.savetxt(filepath, outarray, header=hd, comments='', fmt=outfmt)
        elif outarray.dtype.name == 'int32':
            hd = "{0:d} {1:d} {2:d}".format(outarray.shape[0], outarray.shape[1], 3)
            numpy.savetxt(filepath, outarray, header=hd, comments='', fmt='%d')
        else:
            raise ValueError

        return True

    except OSError:
        sys.stderr.write("Failure to write array file: {0:s}!\n".format(filepath))
        return False

    except ValueError:
        sys.stderr.write("Invalid data type for output array: {0:s}".format(outarray.dtype.name))
        return False


def read_array_bin_file(filepath):
    """ Reads a binary array file.

    The binary file has a short header of 96 bits, comprising three integers that describe, in order:
    - Number of rows;
    - Number of columns;
    - Data type,

    of the saved array. The header is used to reconstruct the array. Array bits are saved in sequence.

    With <M> <t> being a header that describes number of columns and
    array type, respectively. The type codes are:
    0: uint8
    3: int32
    5: float32
    7: float16
    9: float64

    :param filepath: output file name and path;
    :return: read array;
    """

    try:
        binf = open(filepath, 'r')

        header = numpy.fromfile(binf, count=3, dtype=numpy.int32)
        print(header)

        rows = header[0]
        cols = header[1]
        dt = header[2]

        if dt == 0:
            inarray = numpy.fromfile(binf, dtype=numpy.uint8)
        elif dt == 5:
            inarray = numpy.fromfile(binf, dtype=numpy.float32)
        elif dt == 7:
            inarray = numpy.fromfile(binf, dtype=numpy.float16)
        elif dt == 9:
            inarray = numpy.fromfile(binf, dtype=numpy.float64)
        elif dt == 3:
            inarray = numpy.fromfile(binf, dtype=numpy.int32)
        else:
            raise ValueError

        inarray.resize(rows, cols)
        binf.close()

        return inarray

    except OSError:
        sys.stderr.write("Failure to read binary array file: {0:s}!\n".format(filepath))
        return None

    except ValueError:
        sys.stderr.write("Unknown type formating in input array {0:s}!\n".format(filepath))
        return None


def write_array_bin_file(filepath, array_list):
    """ Writes a binary array file.

    The binary file has a short header of 96 bits, comprising three integers that describe, in order:
    - Number of rows;
    - Number of columns;
    - Data type,

    of the saved array. The header is used to reconstruct the array. Array bits are saved in sequence.

    With <M> <t> being a header that describes number of columns and
    array type, respectively. The type codes are:
    0 - uint8;
    3 - int32;
    5 - float32;
    7 - float16;
    9 - float64.

    :param filepath: output file name and path;
    :param array_list: either a list of arrays or a single array. The list is vertically stacked before being saved;
    :return: True if writing as successful, False otherwise.
    """

    outarray = numpy.vstack(array_list)
    header = [outarray.shape[0]]
    try:
        header.append(outarray.shape[1])
    except IndexError:
        header.append(1)

    if outarray.dtype.name == 'uint8':
        header.append(0)
    elif outarray.dtype.name == 'float32':
        header.append(5)
    elif outarray.dtype.name == 'float16':
        header.append(7)
    elif outarray.dtype.name == 'int32':
        header.append(3)

    elif outarray.dtype.name == 'int64':
        header.append(9)

    else:
        print("Invalid array type: ", outarray.dtype.name)
        raise ValueError

    header = numpy.array(header, dtype=numpy.int32)

    try:
        binf = open(filepath, 'w')
        header.tofile(binf)
        outarray.tofile(binf)
        binf.close()
        return True

    except OSError:
        sys.stderr.write("Failure to write binary array file: {0:s}!\n".format(filepath))
        return False

    except ValueError:
        sys.stderr.write("Invalid data type for output array: {0:s}".format(outarray.dtype.name))
        return False
