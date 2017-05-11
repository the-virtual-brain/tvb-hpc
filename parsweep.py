#!/usr/bin/env python3

from __future__ import print_function
import sys
import numpy as np
import os.path
import glob
import numpy as np
import itertools
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pytools
import time
import argparse
import logging

here = os.path.dirname(os.path.abspath(__file__))

def load_connectome(dataset):#{{{
    # load connectome & normalize
    if dataset == 'hcp':
        npz = np.load('hcp-100.npz')
        weights = npz['weights'][0].astype(np.float32)
        lengths = npz['lengths'][0].astype(np.float32)
    elif dataset == 'sep':
        npz = np.load('sep.npz')
        weights = npz['weights'].astype(np.float32)
        lengths = npz['lengths'].astype(np.float32)
    else:
        Nfname, = glob.glob(os.path.expanduser('~/data/data/sep/*/%s_N.txt' % dataset))
        Dfname, = glob.glob(os.path.expanduser('~/data/data/sep/*/%s_dist.txt' % dataset))
        weights = np.loadtxt(Nfname).astype(np.float32)[36:, :36]
        lengths = np.loadtxt(Dfname).astype(np.float32)[36:, :36]
    # weights /= {'N':2e3, 'Nfa': 1e3, 'FA': 1.0}[mattype]
    weights /= weights.max()
    assert (weights <= 1.0).all()
    return weights, lengths#}}}

def expand_params(couplings, speeds):#{{{
    ns = speeds.size
    nc = couplings.size
    params = itertools.product(speeds, couplings)
    params_matrix = np.array([vals for vals in params])
    return params_matrix#}}}

def setup_params(nc, ns):#{{{
    # the correctness checks at the end of the simulation
    # are matched to these parameter values, for the moment
    couplings = np.logspace(1.6, 3.0, nc)
    speeds = np.logspace(0.0, 2.0, ns)
    return couplings, speeds#}}}

def make_kernel(source_file, warp_size, block_dim_x, kernel_name='', ext_options='', #{{{
        caching='none', lineinfo=False, nh='nh', model='kuromoto'):
    with open(source_file, 'r') as fd:
        source = fd.read()
        source = source.replace('M_PI_F', '%ff' % (np.pi, ))
        opts = ['--ptxas-options=-v', ]# '-maxrregcount=32']# '-lineinfo']
        if lineinfo:
            opts.append('-lineinfo')
        opts.append('-DWARP_SIZE=%d' % (warp_size, ))
        opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x, ))
        opts.append('-DNH=%s' % (nh, ))
        if ext_options:
	        opts.append(ext_options)
        cache_opt = {
            'none': None,
            'shuffle': '-DCACHING_SHUFFLE',
            'shared': '-DCACHING_SHARED',
            'shared_sync': '-DCACHING_SHARED_SYNC',
        }[caching]
        if cache_opt:
            opts.append(cache_opt)
        idirs = [here]
        logger.info('nvcc options %r', opts)
        network_module = SourceModule(
                source, options=opts, include_dirs=idirs,
                no_extern_c=True,
                keep=False,
        )
        # no API to know the mangled function name ahead of time
        # if the func sig changes, just copy-paste the new name here..
        # TODO parse verbose output of nvcc to get function name
        if model == 'rww':
                # Reduced Wong Wang model
                step_fn = network_module.get_function('_Z18integrate_wongwangjjjjjffPfS_S_S_S_')
        else:
                # Kuramoto is the default  
                step_fn = network_module.get_function('_Z9integratejjjjjffPfS_S_S_S_')
    # nvcc the bold model kernel
    with open('balloon.c', 'r') as fd:
        source = fd.read()
        opts = []
        opts.append('-DWARP_SIZE=%d' % (warp_size, ))
        opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x, ))
        bold_module = SourceModule(source, options=opts)
        bold_fn = bold_module.get_function('bold_update')
    with open('covar.c', 'r') as fd:
        source = fd.read()
        opts = ['-ftz=true']  # for faster rsqrtf in corr
        opts.append('-DWARP_SIZE=%d' % (warp_size, ))
        opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x, ))
        covar_module = SourceModule(source, options=opts)
        covar_fn = covar_module.get_function('update_cov')
        cov_corr_fn = covar_module.get_function('cov_to_corr')
    return step_fn, bold_fn, covar_fn, cov_corr_fn #}}}

def cf(array):#{{{
    # coerce possibly mixed-stride, double precision array to C-order single precision
    return array.astype(dtype='f', order='C', copy=True)#}}}

def nbytes(data):#{{{
    # count total bytes used in all data arrays
    nbytes = 0
    for name, array in data.items():
        nbytes += array.nbytes
    return nbytes#}}}

def make_gpu_data(data):#{{{
    # put data onto gpu
    gpu_data = {}
    for name, array in data.items():
        gpu_data[name] = gpuarray.to_gpu(cf(array))
    return gpu_data#}}}

def parse_args():#{{{
    parser = argparse.ArgumentParser(description='Run parameter sweep.')
    parser.add_argument('-c', '--n_coupling', help='num grid points for coupling parameter', default=32, type=int)
    parser.add_argument('-s', '--n_speed', help='num grid points for speed parameter', default=32, type=int)
    parser.add_argument('-t', '--test', help='check results', action='store_true')
    parser.add_argument('-n', '--n_time', help='number of time steps to do (default 400)', type=int, default=400)
    parser.add_argument('-v', '--verbose', help='increase logging verbosity', action='store_true')
    parser.add_argument('-p', '--no_progress_bar', help='suppress progress bar', action='store_false')
    parser.add_argument('--caching',
            choices=['none', 'shared', 'shared_sync', 'shuffle'],
            help="caching strategy for j_node loop (default shuffle)",
            default='none'
            )
    parser.add_argument('--dataset',
            help="dataset to use, from ~/data/data/sep/*",
            default='hcp',
            )
    parser.add_argument('--node_threads', default=1, type=int)
    parser.add_argument('--model',
            choices=['rww', 'kuramoto'],
            help="neural mass model to be used during the simulation",
            default='kuramoto'
            )
    parser.add_argument('--lineinfo', default=False, action='store_true')
    args = parser.parse_args()
    return args
    #}}}


if __name__ == '__main__':

    # parse args {{{
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger('[parsweep.py]')
    logger.info('dataset %s', args.dataset)
    logger.info('device pci_bus_id %s', pycuda.autoinit.device.pci_bus_id())
    logger.info('caching strategy %r', args.caching)
    if args.test and args.n_time % 200:
        logger.warning('rerun w/ a multiple of 200 time steps (-n 200, -n 400, etc) for testing') #}}}

    # setup data#{{{
    weights, lengths = load_connectome(args.dataset)
    nc = args.n_coupling
    ns = args.n_speed
    logger.info('single connectome, %d x %d parameter space', ns, nc)
    logger.info('%d total num threads', ns * nc)
    couplings, speeds = setup_params(nc=nc, ns=ns)
    params_matrix = expand_params(couplings, speeds)#}}}

    # dimensions#{{{
    dt, tavg_period = 1.0, 10.0
    states = 1
    if args.model == 'rww':
            states = 2
    # TODO buf_len per speed/block
    min_speed = speeds.min()
    n_work_items, n_params = params_matrix.shape
    n_nodes = weights.shape[0]
    buf_len_ = ((lengths / min_speed / dt).astype('i').max() + 1)
    buf_len = 2**np.argwhere(2**np.r_[:30] > buf_len_)[0][0]  # use next power of 2
    logger.info('real buf_len %d, using power of 2 %d', buf_len_, buf_len)
    n_inner_steps = int(tavg_period / dt)#}}}

    # setup data#{{{
    data = { 'weights': weights, 'lengths': lengths, 'params': params_matrix.T }
    base_shape = n_work_items,
    for name, shape in dict(
            tavg0=(n_nodes,),
            tavg1=(n_nodes,),
            state=(buf_len, states * n_nodes),
            bold_state=(4, n_nodes),
            bold=(n_nodes, ),
            covar_means=(2 * n_nodes, ),
            covar_cov=(n_nodes, n_nodes, ),
            corr=(n_nodes, n_nodes, ),
            ).items():
        data[name] = np.zeros(shape + base_shape, 'f')
    data['bold_state'][1:] = 1.0#}}}

    gpu_data = make_gpu_data(data)#{{{
    logger.info('history shape %r', data['state'].shape)
    logger.info('on device mem: %.3f MiB' % (nbytes(data) / 1024 / 1024, ))#}}}

    # setup CUDA stuff#{{{
    step_fn, bold_fn, covar_fn, cov_corr_fn = make_kernel(
            source_file='network.c',
            warp_size=32,
            block_dim_x=args.n_coupling,
            ext_options='-DRAND123',
            caching=args.caching,
            lineinfo=args.lineinfo,
            nh=buf_len,
            model=args.model,
            )#}}}

    # setup simulation#{{{
    tic = time.time()
    nstep = args.n_time # 4s
    streams = [drv.Stream() for i in range(32)]
    events = [drv.Event() for i in range(32)]
    tavg_unpinned = []
    bold_unpinned = []
    tavg = drv.pagelocked_zeros((32, ) + data['tavg0'].shape, dtype=np.float32)
    bold = drv.pagelocked_zeros((32, ) + data['bold'].shape, dtype=np.float32)
    #}}}

    # adjust gridDim to keep block size <= 1024 {{{
    block_size_lim = 1024
    n_coupling_per_block = block_size_lim // args.node_threads
    n_coupling_blocks = args.n_coupling // n_coupling_per_block
    if n_coupling_blocks == 0:
        n_coupling_per_block = args.n_coupling
        n_coupling_blocks = 1
    final_block_dim = n_coupling_per_block, args.node_threads, 1
    final_grid_dim = speeds.size, n_coupling_blocks
    logger.info('final block dim %r', final_block_dim)
    logger.info('final grid dim %r', final_grid_dim)
    assert n_coupling_per_block * n_coupling_blocks == args.n_coupling #}}}

    # run simulation#{{{
    logger.info('submitting work')
    for i in range(nstep):

        event = events[i % 32]
        stream = streams[i % 32]

        stream.wait_for_event(events[(i - 1) % 32])

        step_fn(np.uintc(i * n_inner_steps), np.uintc(n_nodes), np.uintc(buf_len), np.uintc(n_inner_steps),
                np.uintc(n_params), np.float32(dt), np.float32(min_speed),
                gpu_data['weights'], gpu_data['lengths'], gpu_data['params'], gpu_data['state'],
                gpu_data['tavg%d' % (i%2,)],
                block=final_block_dim,
                grid=final_grid_dim,
                stream=stream)

        event.record(streams[i % 32])

        # TODO check next integrate not zeroing current tavg?
        tavgk = 'tavg%d' % ((i + 1) % 2, )
        bold_fn(np.uintc(n_nodes),
                # BOLD model dt is in s, requires 1e-3
                np.float32(dt * n_inner_steps * 1e-3),
                gpu_data['bold_state'], gpu_data[tavgk], gpu_data['bold'],
                block=(couplings.size, 1, 1), grid=(speeds.size, 1), stream=stream)

        if i >= (nstep // 2):
            i_time = i - nstep // 2
            covar_fn(np.uintc(i_time), np.uintc(n_nodes),
                gpu_data['covar_cov'], gpu_data['covar_means'], gpu_data[tavgk],
                block=final_block_dim, grid=final_grid_dim, stream=stream)

        # async wrt. other streams & host, but not this stream.
        if i >= 32:
            stream.synchronize()
            tavg_unpinned.append(tavg[i % 32].copy())
            if i%180==0:
                bold_unpinned.append(bold[i % 32].copy())

        drv.memcpy_dtoh_async(tavg[i % 32], gpu_data[tavgk].ptr, stream=stream)
        drv.memcpy_dtoh_async(bold[i % 32], gpu_data['bold'].ptr, stream=stream)
        
        if i == (nstep - 1):
            cov_corr_fn(np.uintc(nstep // 2), np.uintc(n_nodes),
                    gpu_data['covar_cov'], gpu_data['corr'],
                    block=(couplings.size, 1, 1), grid=(speeds.size, 1), stream=stream)

    logger.info('waiting for work to finish..')

    # recover uncopied data from pinned buffer
    if nstep > 32:
        for i in range(nstep % 32, 32):
            stream.synchronize()
            tavg_unpinned.append(tavg[i].copy())
            if i % 180 == 0:
                bold_unpinned.append(bold[i].copy())

    for i in range(nstep % 32):
        stream.synchronize()
        tavg_unpinned.append(tavg[i].copy())
        if i % 180 == 0:
            bold_unpinned.append(bold[i].copy())

    corr = gpu_data['corr'].get()

    elapsed = time.time() - tic
    # release pinned memory
    bold = np.array(bold_unpinned)
    tavg = np.array(tavg_unpinned)
    # inform about time
    logger.info('elapsed time %0.3f', elapsed)
    logger.info('%0.3f M step/s', 1e-6 * nstep * n_inner_steps * n_work_items / elapsed)#}}}

    np.savez('results-%s.npz' % args.dataset, bold=bold, corr=corr)

    # check results (for smaller sizes)#{{{
    if args.test:
        r, c = np.triu_indices(n_nodes, 1)
        win_size = 200 # 2s
        win_tavg = tavg.reshape((-1, win_size) + tavg.shape[1:])
        err = np.zeros((len(win_tavg), n_work_items))
        # TODO do cov/corr in kernel
        for i, tavg_ in enumerate(win_tavg):
            for j in range(n_work_items):
                fc = np.corrcoef(tavg_[:, :, j].T)
                err[i, j] = ((fc[r, c] - weights[r, c])**2).sum()
        # look at 2nd 2s window (converges quickly)
        err_ = err[-1].reshape((speeds.size, couplings.size))
        # change on fc-sc metric wrt. speed & coupling strength
        derr_speed = np.diff(err_.mean(axis=1)).sum()
        derr_coupl = np.diff(err_.mean(axis=0)).sum()
        logger.info('derr_speed=%f, derr_coupl=%f', derr_speed, derr_coupl)
        if args.dataset == 'hcp':
            assert derr_speed > 500.0
            assert derr_coupl < -1500.0
        if args.dataset == 'sep':
            assert derr_speed > 5e4
            assert derr_coupl > 1e4
        # test cov/corr
        for j in range(n_work_items):
            cv1 = np.cov(win_tavg[-1][:, :, j].T)
            cv2 = gpu_data['covar_cov'].get()[:, :, j]
            np.testing.assert_allclose(cv1, cv2, 1e-2, 1)
            cf1 = np.corrcoef(win_tavg[-1][:, :, j].T)
            cf2 = corr[:, :, j]
            np.testing.assert_allclose(cf1, cf2, 1e-2, 0.1)
        logger.info('result OK')
        #}}}

# vim: sw=4 sts=4 ts=4 et ai
