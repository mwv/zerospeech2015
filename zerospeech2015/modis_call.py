from subprocess import call

def call_modis(file_list_file, out_path, motif_min_size=0.5, nfeatures=39):
    cpu_info_output = 'cpu_info'

    # match detection threshold, between 0.15 (high precision) and
    # 3.0 (high recall). bogdan has found highest f-score around 2.2
    threshold = 1.8
    params = ['-v',  # verbose
              '-m',  # merge overlapping occurrences
              '-U', cpu_info_output, # file to save computation time information to
              '--model=2', # median model with SSM checking
              '--ssm', # enable self-similarity checking
              '--motifMinSize={0}'.format(motif_min_size),
              '--file-list={0}'.format(file_list_file),
              '--ascii',
              '--vecdim={0}'.format(nfeatures),
              out_path,
              '0', # word discovery
              '1', # frame-distance type: 0=
              '0.25', # seed-size
              str(threshold)]
    call(['modis'] + params)
