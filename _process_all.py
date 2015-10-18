import os, sys
from collections import OrderedDict
#import multiprocessing
from ipyparallel import Client
from glob import glob

def myexec(fpath):
    okay = False
    exc = ''
    import os.path
    import traceback
    import matplotlib.pyplot as plt
    fname = os.path.basename(fpath)
    print('Executing {} ...'.format(fname))
    try:
        execfile(fpath)
        plt.close('all')
        okay = True
    except:
        exc = traceback.format_exc()
    return fname, (okay, exc)


cwd = os.path.dirname(sys.argv[0])
files = [os.path.abspath(fn) 
         for fn in glob(os.path.join(cwd, '[0-1][0-9]*.py'))]

## Problem is that both, data and figures, are given in same path
## so chdir is not the solution to store figures in Figures ...
#wdir=os.path.join(cwd, 'Figures')
#os.chdir(wdir)

## no proper closing of figures
#pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
#results = OrderedDict(sorted(pool.map(myexec, files)))

## proper closing of figures per process 
## at least in ipython session with %matplotlib enabled
c = Client()
lview = c.load_balanced_view()
results = OrderedDict(sorted(lview.map(myexec, files).result))

for key, vals in results.iteritems():
    print('\n\n## {} ##\n'.format(key))
    if vals[0]: 
        print('Status: Okay.')
    else:
        print('Status: Not okay.')
        print(vals[1])
