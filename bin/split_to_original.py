VERSION="0.1.0"

from collections import defaultdict

def read_mapping(fname):
    mapping = {}
    for line in open(fname):
        orig, start, end, dest = line.strip().split(' ')
        mapping[dest] = (float(start), float(end))
    return mapping

class FileNameError(Exception):
    pass

class IntervalError(Exception):
    pass


def find(mapping, fname, start, end):
    try:
        return mapping[fname]
    except KeyError:
        raise FileNameError


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='split_to_original.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Relate temporal intervals in split english files'
            ' to unsplit english files',
            epilog="""Example usage:

$ python split_to_original.py s0504a_48 0.000 2.555

prints:

s0504a 157.997 160.552

indicating that the interval [0.000-2.555] in the split file s0504a_48 is the
same as the interval [157.997-160.552] in the unsplit file s0504a.
""")
        parser.add_argument('file', metavar='INPUTFILE',
                            nargs=1,
                            help='file in the split corpus')
        parser.add_argument('start', metavar='START',
                            nargs=1,
                            help='start')
        parser.add_argument('end', metavar='END',
                            nargs=1,
                            help='end')
        parser.add_argument('-V', '--version', action='version',
                            version="%(prog)s version {version}".format(
                                version=VERSION))
        return vars(parser.parse_args())
    args = parse_args()

    fname = args['file'][0]
    start = float(args['start'][0])
    end = float(args['end'][0])

    mapping = read_mapping('ENGLISH_SPLIT_LOG')


    try:
        orig_start, orig_end = find(mapping, fname, start, end)
        print '{0} {1:.3f} {2:.3f}'.format(fname.split('_')[0],
                                           start + orig_start, end + orig_start)
    except IntervalError:
        print 'Interval not found in any file: {0} {1:.3f} {2:.3f}'.format(
            fname, start, end)
    except FileNameError:
        print 'Filename not found: {0}'.format(fname)
