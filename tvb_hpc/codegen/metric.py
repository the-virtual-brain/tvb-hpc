

class Covar:
    template = """
// stable one-pass co-moment algo, cf wikipedia
__kernel void update_cov(int i_sample,
			 int n_node,
			 __global float *cov,
			 __global float *means,
			 __global float *data)
{
    int it = get_global_id(0), nt = get_global_size(0);

    if (i_sample == 0)
    {
	for (int i_node = 0; i_node < n_node; i_node++)
	    means[i_node * nt + it] = data[i_node * nt + it];
	return;
    }

    float recip_n = 1.0f / i_sample;

    // double buffer to avoid copying memory
    __global float *next_mean = means, *prev_mean = means;
    if (i_sample % 2 == 0) {
	prev_mean += n_node * nt;
    } else {
	next_mean += n_node * nt;
    }

    for (int i_node = 0; i_node < n_node; i_node++)
    {
	int i_idx = i_node * nt + it;
	next_mean[i_idx] = prev_mean[i_idx] \
	    + (data[i_idx] - prev_mean[i_idx]) * recip_n;
    }

    for (int i_node = 0; i_node < n_node; i_node++)
    {
	int i_idx = i_node * nt + it;
	float data_mean_i = data[i_idx] - prev_mean[i_idx];

	for (int j_node = 0; j_node < n_node; j_node++)
	{
	    int j_idx = j_node * nt + it;
	    float data_mean_j = data[j_idx] - next_mean[j_idx];

	    int cij_idx = (j_node * n_node + i_node) * nt + it;
	    cov[cij_idx] += data_mean_j * data_mean_i;
	}
    }
}
"""

class CovToCorr:
    template = """
__kernel void cov_to_corr(int n_sample, int n_node,
			  __global float *cov,
			  __global float *corr)
{
    int it = get_global_id(0), nt = get_global_size(0);

    float recip_n_samp = 1.0f / n_sample;

    // normalize comoment to covariance
    for (int ij = 0; ij < (n_node * n_node); ij++)
	cov[ij*nt + it] *= recip_n_samp;

    // compute correlation coefficient
#define COV(i, j) cov[(i*n_node + j)*nt + it]
#define CORR(i, j) corr[(i*n_node + j)*nt + it]

    for (int i = 0; i < n_node; i++)
    {
	float var_i = COV(i, i);
	for (int j = 0; j < n_node; j++)
	{
	    float var_j = COV(j, j);
	    CORR(i, j) = COV(i, j) / sqrt(var_i * var_j);
	}
    }
}
"""