from __future__ import division

import os
import os.path as path
import glob
import string
from collections import defaultdict, Counter, namedtuple
from itertools import izip, chain, repeat
import fnmatch
import sys
from contextlib import contextmanager
import shutil
import time

import pandas as pd
import numpy as np
import scipy.stats
from scipy.spatial.distance import pdist, squareform

import toml

Interval = namedtuple('Interval', ['start', 'end'])
Fragment = namedtuple('Fragment', ['name', 'interval', 'mark'])
FileSet = namedtuple('FileSet', ['phn', 'wrd', 'wav'])
Token = namedtuple('Token', ['word', 'filename', 'interval', 'phones'])
FileSet = namedtuple('FileSet', ['phn', 'wrd', 'wav'])

@contextmanager
def verb_print(label, verbose, when_done=False, timeit=False, with_dots=True):
    if verbose:
        msg = label + ('...' if with_dots else '') + ('' if when_done else '\n')
        if timeit:
            t0 = time.time()
        print msg,
        sys.stdout.flush()
    try:
        yield
    finally:
        if verbose and when_done:
            if timeit:
                print 'done. Time: {0:.3f}s'.format(time.time() - t0)
            else:
                print 'done.'
            sys.stdout.flush()


def gather_files_per_speaker(source):
    fs = defaultdict(list)
    for phnfile in glob.iglob(path.join(source, 'phn', '*.phn')):
        bname = path.splitext(path.basename(phnfile))[0]
        speaker = bname[:3]
        wavfile = path.join(source, 'wav', bname + '.wav')
        wrdfile = path.join(source, 'wrd', bname + '.wrd')
        if path.exists(wavfile) and \
           path.exists(wrdfile):
            fs[speaker].append(FileSet(phnfile, wrdfile, wavfile))
    return fs

def read_phn(fname):
    bname = path.basename(fname)
    fragments = []
    for line in open(fname):
        try:
            start, end, mark = line.strip().split(' ')
        except ValueError as e:
            print fname
            print line
            raise e
        fragments.append(Fragment(bname, Interval(start, end), mark))
    return fragments


def read_wrd(fname):
    bname = path.basename(fname)
    fragments = []
    for line in open(fname):
        splitline = line.strip().split(' ')
        start = splitline[0]
        end = splitline[1]
        wordchunk = ' '.join(splitline[2:])
        word = wordchunk.split(';')[0]
        fragments.append(Fragment(bname, Interval(start, end), word))
    return fragments


def rglob(rootdir, pattern):
    for root, _, files in os.walk(rootdir):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                yield path.join(root, basename)


def jsd(ps):
    """
    Parameters
    ----------
    ps : nbins x ndistributions
    """

    h = scipy.stats.entropy
    return h(ps.sum(1)/ps.shape[1]) - (h(ps)/ps.shape[1]).sum()


def pairwise_jsd(p, q):
    return jsd(np.vstack((p, q)).T)


def index_by(iterable, query, key):
    for ix, element in enumerate(iterable):
        if key(element) == query:
            return ix
    raise ValueError('{0} is not in list'.format(query))


def make_concordance(wrd_annot, phn_annot):
    concordance = defaultdict(list)
    for bname, fragment in chain(*(izip(repeat(k), v)
                                   for k, v in wrd_annot.iteritems())):
        interval = fragment.interval
        filename = bname
        word = fragment.mark
        phn_fragments = phn_annot[bname]
        try:
            phn_start_ix = index_by(phn_fragments, interval.start,
                                    lambda f: f.interval.start)
        except ValueError as exc:
            print 'error in start', bname, interval.start
            raise exc
        try:
            phn_end_ix = index_by(phn_fragments, interval.end,
                                  lambda f: f.interval.end) + 1
        except ValueError as exc:
            print 'error in end', bname, interval.end
            raise exc
        phones = phn_fragments[phn_start_ix: phn_end_ix]
        token = Token(word, filename, interval, phones)
        concordance[fragment.mark].append(token)
    return concordance


def train_test_split(seq, cut):
    ranked_ixs = np.arange(len(seq))
    np.random.shuffle(ranked_ixs)
    return seq[ranked_ixs[:cut]], seq[ranked_ixs[cut:]]


def get_speakers():
    SPEAKERS = {'s01': ('f', 'y', 'f'), 's02': ('f', 'o', 'm'),
                's03': ('m', 'o', 'm'), 's04': ('f', 'y', 'f'),
                's05': ('f', 'o', 'f'), 's06': ('m', 'y', 'f'),
                's07': ('f', 'o', 'f'), 's08': ('f', 'y', 'f'),
                's09': ('f', 'y', 'f'), 's10': ('m', 'o', 'f'),
                's11': ('m', 'y', 'm'), 's12': ('f', 'y', 'm'),
                's13': ('m', 'y', 'f'), 's14': ('f', 'o', 'f'),
                's15': ('m', 'y', 'm'), 's16': ('f', 'o', 'm'),
                's17': ('f', 'o', 'm'), 's18': ('f', 'o', 'f'),
                's19': ('m', 'o', 'f'), 's20': ('f', 'o', 'f'),
                's21': ('f', 'y', 'm'), 's22': ('m', 'o', 'f'),
                's23': ('m', 'o', 'm'), 's24': ('m', 'o', 'm'),
                's25': ('f', 'o', 'm'), 's26': ('f', 'y', 'f'),
                's27': ('f', 'o', 'm'), 's28': ('m', 'y', 'm'),
                's29': ('m', 'o', 'f'), 's30': ('m', 'y', 'm'),
                's31': ('f', 'y', 'm'), 's32': ('m', 'y', 'f'),
                's33': ('m', 'y', 'f'), 's34': ('m', 'y', 'm'),
                's35': ('m', 'o', 'm'), 's36': ('m', 'o', 'f'),
                's37': ('f', 'y', 'm'), 's38': ('m', 'o', 'm'),
                's39': ('f', 'y', 'm'), 's40': ('m', 'y', 'f')}
    return pd.DataFrame([[speaker, v[0], v[1]]
                         for speaker, v in sorted(SPEAKERS.items())],
                        columns=['speaker','gender', 'age'])


def split_speakers(corpus_path, nspeakers):
    wrd_annot = {path.splitext(path.basename(fname))[0]:
                 read_wrd(fname)
                 for fname in glob.iglob(path.join(corpus_path, '*.wrd'))}

    words = set(fragment.mark.translate(string.maketrans('',''), string.punctuation)
                for fragments in wrd_annot.itervalues() for fragment in fragments)
    ix2word = dict(enumerate(sorted(list(words))))

    words_per_speaker = defaultdict(list)
    for key, fragments in wrd_annot.iteritems():
        speaker = key[:3]
        words_per_speaker[speaker].extend([f.mark for f in fragments])

    speakers = get_speakers()
    ix2speaker = dict(enumerate(sorted(speakers.speaker.values)))
    speaker2ix = {v:k for k, v in ix2speaker.iteritems()}

    counts = np.zeros((len(speaker2ix), len(words)), dtype=np.uint)
    for speaker_ix, speaker in ix2speaker.iteritems():
        c = Counter(words_per_speaker[speaker])
        for word_ix, word in ix2word.iteritems():
            counts[speaker_ix][word_ix] = c[word]
    dissimilarities = squareform(pdist(counts, metric=pairwise_jsd))
    ranked_ixs = np.argsort(dissimilarities.sum(0))

    fy = set(speakers.speaker[(speakers.gender == 'f') &
                              (speakers.age == 'y')].values)
    fo = set(speakers.speaker[(speakers.gender == 'f') &
                              (speakers.age == 'o')].values)
    my = set(speakers.speaker[(speakers.gender == 'm') &
                              (speakers.age == 'y')].values)
    mo = set(speakers.speaker[(speakers.gender == 'm') &
                              (speakers.age == 'o')].values)

    # number of desired female young (nfy), female old (nfo), male young (nmy), male old (nmo)
    # split up speakers dict
    nfyd = nspeakers['dev']['young_female']
    nfyt = nspeakers['test']['young_female']
    nfod = nspeakers['dev']['old_female']
    nfot = nspeakers['test']['old_female']
    nmyd = nspeakers['dev']['young_male']
    nmyt = nspeakers['test']['young_male']
    nmod = nspeakers['dev']['old_male']
    nmot = nspeakers['test']['old_male']


    nfy = nfyd + nfyt
    fy_sel = speakers.loc[ranked_ixs][(speakers.speaker.isin(fy))][:nfy]
    fy_train, fy_test = train_test_split(fy_sel.speaker.values,
                                         nfyd)

    nfo = nfod + nfot
    fo_sel = speakers.loc[ranked_ixs][(speakers.speaker.isin(fo))][:nfo]
    fo_train, fo_test = train_test_split(fo_sel.speaker.values,
                                         nfod)

    nmy = nmyd + nmyt
    my_sel = speakers.loc[ranked_ixs][(speakers.speaker.isin(my))][:nmy]
    my_train, my_test = train_test_split(my_sel.speaker.values,
                                         nmyd)

    nmo = nmod + nmot
    mo_sel = speakers.loc[ranked_ixs][(speakers.speaker.isin(mo))][:nmo]
    mo_train, mo_test = train_test_split(mo_sel.speaker.values,
                                         nmod)

    train = list(fy_train) + list(fo_train) + list(my_train) + list(mo_train)
    test = list(fy_test) + list(fo_test) + list(my_test) + list(mo_test)
    return train, test


def check_source(source):
    if not path.exists(source):
        print 'no such directory: {0}'.format(source)
        exit()


def check_destination(destination):
    if not path.exists(destination):
        os.makedirs(destination)
        os.makedirs(path.join(destination, 'wav', 'dev'))
        os.makedirs(path.join(destination, 'wav', 'test'))
        os.makedirs(path.join(destination, 'phn', 'dev'))
        os.makedirs(path.join(destination, 'phn', 'test'))
        os.makedirs(path.join(destination, 'wrd', 'dev'))
        os.makedirs(path.join(destination, 'wrd', 'test'))

def move_files(files, train, test, destination):
    train = set(train)
    test = set(test)

    for speaker in train:
        for phnfile, wrdfile, wavfile in files[speaker]:
            bname = path.splitext(path.basename(phnfile))[0]
            shutil.copyfile(phnfile,
                            path.join(destination, 'phn', 'dev',
                                      bname + '.phn'))
            shutil.copyfile(wrdfile,
                            path.join(destination, 'wrd', 'dev',
                                      bname + '.wrd'))
            shutil.copyfile(wavfile,
                            path.join(destination, 'wav', 'dev',
                                      bname + '.wav'))
    for speaker in test:
        for phnfile, wrdfile, wavfile in files[speaker]:
            bname = path.splitext(path.basename(phnfile))[0]
            shutil.copyfile(phnfile,
                            path.join(destination, 'phn', 'test',
                                      bname + '.phn'))
            shutil.copyfile(wrdfile,
                            path.join(destination, 'wrd', 'test',
                                      bname + '.wrd'))
            shutil.copyfile(wavfile,
                            path.join(destination, 'wav', 'test',
                                      bname + '.wav'))


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='buckeye_split_devtest.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Split the buckeye corpus into dev and test sets',
            epilog="""Example usage:

$ python buckeye_split_devtest.py /path/to/buckeye/ /path/to/destination/

""")
        parser.add_argument('source', metavar='SOURCE',
                            nargs=1,
                            help='location of (noise split) buckeye corpus')
        parser.add_argument('destination', metavar='DESTINATION',
                            nargs=1,
                            help='location for output')
        parser.add_argument('-c', '--config',
                            dest='config',
                            default='config.toml',
                            help='location of configuration file')
        parser.add_argument('--log',
                            action='store',
                            dest='log',
                            default=None,
                            help='save log of transformations')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='verbose')
        return vars(parser.parse_args())

    args = parse_args()
    source = args['source'][0]
    check_source(source)
    destination = args['destination'][0]
    check_destination(destination)
    config_file = args['config']
    with open(config_file) as fid:
        config = toml.loads(fid.read())
    verbose = args['verbose']

    with verb_print('gathering files', verbose, True, True, True):
        files = gather_files_per_speaker(source)

    with verb_print('splitting speakers', verbose, True, True, True):
        train, test = split_speakers(source, config['speakers'])

    with verb_print('moving files', verbose, True, True, True):
        move_files(files, train, test, destination)

    print 'TRAIN'
    print '\n'.join(train)
    print
    print 'TEST'
    print '\n'.join(test)
    print
