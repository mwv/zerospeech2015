VERSION="0.1.0"

from collections import defaultdict

def read_mapping(fname):
    mapping = defaultdict(list)
    for line in open(fname):
        orig, start, end, dest = line.strip().split(' ')
        mapping[orig].append((float(start), float(end), dest))
    return dict(mapping)

class FileNameError(Exception):
    pass

class IntervalError(Exception):
    pass

def find(mapping, fname, start, end):
    try:
        sublist = mapping[fname]
    except KeyError:
        raise FileNameError
    for ival_start, ival_end, ival_fname in sublist:
        if ival_start <= start and ival_end >= end:
            return ival_fname, ival_start, ival_end
        if ival_start > end:
            break
    raise IntervalError


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='original_to_split.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=('Relate temporal intervals in unsplit english files'
            ' to split english files'),
            epilog="""Example usage:

$ python original_to_split.py s0504a 157.997 160.552

prints:

s0504a_48 0.000 2.555

indicating that the interval [157.997-160.552] in the unsplit file s0504a is
the same as the interval [0.000-2.555] in the split file s0504a_48.

If you specify an interval in the unsplit corpus that doesn't exist in the
split, like:

$ python original_to_split.py s1102b 216.510 216.910

the program will print:

Interval not found in any split file: s1102b 216.510 216.910
""")
        parser.add_argument('file', metavar='INPUTFILE',
                            nargs=1,
                            help='file in the unsplit corpus')
        parser.add_argument('start', metavar='START',
                            nargs=1,
                            help='start')
        parser.add_argument('end', metavar='END',
                            nargs=1,
                            help='end')
        parser.add_argument('-V', '--version', action='version',
                            version="%(prog)s version {version}"
                            .format(version=VERSION))

        return vars(parser.parse_args())
    args = parse_args()

    fname = args['file'][0]
    start = float(args['start'][0])
    end = float(args['end'][0])

    mapping = read_mapping('ENGLISH_SPLIT_LOG')
    try:
        f_fname, f_start, f_end = find(mapping, fname, start, end)
        print '{0} {1:.3f} {2:.3f}'.format(f_fname, start - f_start, end - f_start)
    except IntervalError:
        print 'Interval not found in any split file: {0} {1:.3f} {2:.3f}'.format(fname, start, end)
    except FileNameError:
        print 'Filename not found: {0}'.format(fname)
