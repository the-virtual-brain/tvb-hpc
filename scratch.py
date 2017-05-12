couplings = np.logspace(1.6, 3.0, nc)
speeds = np.logspace(0.0, 2.0, ns)
ns = speeds.size
nc = couplings.size
params = itertools.product(speeds, couplings)
params_matrix = np.array([vals for vals in params])

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
assert derr_speed > 500.0
assert derr_coupl < -1500.0
