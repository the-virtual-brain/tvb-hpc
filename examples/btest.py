import numpy as np
import loopy as lp

target = lp.CudaTarget()

kernel = lp.make_kernel(
	"{ [i_node,j_node]: 0<=i_node,j_node<n_node}", 
	"""
	<float32> coupling_value = params(1)
	<float32> speed_value = params(0)

	<float32> dt=0.1
	<float32> M_PI_F = 2.0
	<float32> rec_n = 1.0f / n_node
	<float32> rec_speed_dt = 1.0f / speed_value / dt
	<float32> omega = 10.0 * 2.0 * M_PI_F / 1e3
	<float32> sig = sqrt(dt) * sqrt(2.0 * 1e-5)
	<float32> rand = 1.0

	for i_node
		tavg[i_node]=0.0f  							{id = clear}
	end 
	
	for i_node
		<float32> theta_i = state[i_node] 					{id = coupling1, dep=*}
		<float32> sum = 0.0							{id = coupling2}
		for j_node
			<float32> wij = weights[j_node]					{id = coupling3, dep=coupling1:coupling2}
			if wij != 0.0
                		<int> dij = lengths[j_node] * rec_speed_dt		{id = coupling4, dep=coupling3}
                		<float32> theta_j = state[j_node]
                		sum = sum + wij * sin(theta_j - theta_i)
			end
		end

		theta_i = theta_i + dt * (omega + coupling_value * rec_n * sum) 	{id = out1, dep=coupling4}	
		theta_i = theta_i + (sig * rand) 					{id = out2, dep=out1}	
		theta_i = wrap_2_pi(theta_i) 						{id = out3, dep=out2}							
		tavg[i_node] = tavg[i_node] + sin(theta_i) 				{id = out4, dep=out3}
		state[i_node] = theta_i							{dep=*coupling1}						
	end 

	""", assumptions="n_node>=0")

kernel = lp.add_dtypes(kernel, dict(tavg=np.float32, state=np.float32, weights=np.float32, lengths=np.float32))
kernel = kernel.copy(target=lp.CudaTarget())
code = lp.generate_code_v2(kernel)
print (kernel)
print (code.host_code())
print (code.device_code())
