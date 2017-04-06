# tvb hacks for gpu hackathon

This is mainly a port of existing OpenCL kernel to CUDA, for this week.

- `parsweep.py` - driver script
- `network.c` - CUDA kernel source
- `activate` - env setup script to source

Running this looks like

```
~ bsub -R "select [ngpus>0] rusage[ngpus_shared=1]" -Is -tty /bin/bash
Job <45062> is submitted to default queue <normal.i>.
<<Waiting for dispatch ...>>
<<Starting on juronc05>>
~ cd tvb-hpc-hack
~/tvb-hpc-hack source activate 
~/tvb-hpc-hack ./parsweep.py 
INFO:[parsweep_cuda]:single connectome, 32 x 32 parameter space
INFO:[parsweep_cuda]:1024 total num threads
INFO:[parsweep_cuda]:history shape (249, 84, 1024)
INFO:[parsweep_cuda]:on device mem: 82.101 MiB
time-stepping        [########################################] ETA     0.0s 
INFO:[parsweep_cuda]:elapsed time 9.850
INFO:[parsweep_cuda]:0.416 M step/s
```

The driver script accepts a few options of use:

```
~/tvb-hpc-hack ./parsweep.py -h
usage: parsweep.py [-h] [-b BLOCK_DIM] [-g GRID_DIM] [-t] [-v]

Run parameter sweep.

optional arguments:
  -h, --help            show this help message and exit
  -b BLOCK_DIM, --block-dim BLOCK_DIM
                        block dimension (default 32)
  -g GRID_DIM, --grid-dim GRID_DIM
                        grid dimensions (default 32)
  -t, --test            check results
  -v, --verbose         increase logging verbosity
```

