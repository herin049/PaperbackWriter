#pragma once

#include "../../tensor/tensor.hpp"
#include "../../tensor/tensor_span.hpp"
#include "../../tensor/tensor_random.hpp"
#include "../../tensor/memory/copy.hpp"
#include "../../tensor/debug/print.hpp"
#include "../../cuda/tensor/cuda_funcs.cuh"
#include "../../cublas/cublas_funcs.cuh"

class lstm
{
	public:
		lstm();
		~lstm() { cublasDestroy(handle); }
		void forward(float * input, int timestep);
		void backward(float * goal, int timestep);
		void updateweights();
		void updatebiases();
		void reset_delta_next();
		void copy_states();
		void sample(const int timesteps, char key[]);
		void clip();

	private:

		cublasHandle_t handle;

		/* Constants */
		static const int hiddensize = 500;
		static const int inputsize = 89;
		static const int concat_size = hiddensize + inputsize;
		static const int sequencelength = 100;
		float weight_mean;
		float weight_sd;
		float learning_rate;

		/* Weights and Biases */

		/* In order Forget, Input, Cell, Output, Y */

		pbw::tensor<float, 4, hiddensize * concat_size> weights;
		pbw::tensor<float, inputsize * hiddensize> out_weights;

		/* In order Forget, Input, Cell, Output, Y */

		pbw::tensor<float, 4, hiddensize> biases;
		pbw::tensor<float, inputsize> out_biases;

		/* Gradients */

		/* In order Forget, Input, Cell, Output, Y */

		pbw::tensor<float, 4, hiddensize * concat_size> dweights;
		pbw::tensor<float, inputsize * hiddensize> dout_weights;

		/* In order Forget, Input, Cell, Output, Y */

		pbw::tensor<float, 4, hiddensize> dbiases;
		pbw::tensor<float, inputsize> dout_biases;

		/* Memory Variables */

		/* In order Forget, Input, Cell, Output, Y */

		pbw::tensor<float, 4, hiddensize * concat_size> mweights;
		pbw::tensor<float, inputsize * hiddensize> mout_weights;

		/* In order Forget, Input, Cell, Output, Y */

		pbw::tensor<float, 4, hiddensize> mbiases;
		pbw::tensor<float, inputsize> mout_biases;

		/* Forward Pass Variables */

		pbw::tensor<float, sequencelength + 1, inputsize> inputs;
		pbw::tensor<float, sequencelength + 1, concat_size> concatinated;
		pbw::tensor<float, sequencelength + 1, hiddensize> inputgates;
		pbw::tensor<float, sequencelength + 1, hiddensize> forgetgates;
		pbw::tensor<float, sequencelength + 1, hiddensize> cellgates;
		pbw::tensor<float, sequencelength + 1, hiddensize> outputgates;
		pbw::tensor<float, sequencelength + 1, hiddensize> cellstates;
		pbw::tensor<float, sequencelength + 1, hiddensize> hiddenstates;
		pbw::tensor<float, sequencelength + 1, inputsize> yvalues;
		pbw::tensor<float, sequencelength + 1, inputsize> outputvalues;

		/* Hidden Intermediate Variables */

		pbw::tensor<float, 5, hiddensize> hidden_intermediates;

		/* Concatinated Intermediate Variables */
		pbw::tensor<float, 5, concat_size> concat_intermediates;

		/* Backward Variables */
		pbw::tensor<float, inputsize> target_vec;
		pbw::tensor<float, inputsize> dY;
		pbw::tensor<float, hiddensize> dinputgate;
		pbw::tensor<float, hiddensize> dforgetgate;
		pbw::tensor<float, hiddensize> doutputgate;
		pbw::tensor<float, hiddensize> dcellgate;
		pbw::tensor<float, hiddensize> dcellstate;
		pbw::tensor<float, hiddensize> dhiddenstate;
		pbw::tensor<float, hiddensize> dhidden_next;
		pbw::tensor<float, hiddensize> dcell_next;
		pbw::tensor<float, concat_size> dconcatinated;

		/* Sample Variables */

		pbw::tensor<float, inputsize> sampleinput;
		pbw::tensor<float, inputsize> samplenewinput;
		pbw::tensor<float, concat_size> sampleconcatinated;
		pbw::tensor<float, hiddensize> sampleinputgate;
		pbw::tensor<float, hiddensize> sampleforgetgate;
		pbw::tensor<float, hiddensize> samplecellgate;
		pbw::tensor<float, hiddensize> sampleoutputgate;
		pbw::tensor<float, hiddensize> samplecellstate;
		pbw::tensor<float, hiddensize> samplehiddenstate;
		pbw::tensor<float, inputsize> sampleyvalue;
		pbw::tensor<float, inputsize> sampleoutputvalue;
		pbw::tensor<float, hiddensize> samplehiddenprev;
		pbw::tensor<float, hiddensize> samplecellprev;

};