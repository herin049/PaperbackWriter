#pragma once

#include "lstm.hpp"

lstm::lstm()
	: weights(hiddensize * concat_size * 4), dweights(hiddensize * concat_size * 4), mweights(hiddensize * concat_size * 4), biases(hiddensize * 4),
	dbiases(hiddensize * 4), mbiases(hiddensize * 4), out_weights(inputsize * hiddensize),  dout_weights(inputsize * hiddensize), mout_weights(inputsize * hiddensize),
	out_biases(inputsize), dout_biases(inputsize), mout_biases(inputsize), inputs(inputsize * (sequencelength + 1)), concatinated(concat_size * (sequencelength + 1)),
	inputgates(hiddensize * (sequencelength + 1)), forgetgates(hiddensize * (sequencelength + 1)), cellgates(hiddensize * (sequencelength + 1)), outputgates(hiddensize * (sequencelength + 1)),
	cellstates(hiddensize * (sequencelength + 1)), hiddenstates(hiddensize * (sequencelength + 1)), yvalues(inputsize * (sequencelength + 1)), outputvalues(inputsize * (sequencelength + 1)),
	hidden_intermediates(5 * hiddensize), concat_intermediates(5 * concat_size), target_vec(inputsize), dY(inputsize), dinputgate(hiddensize), doutputgate(hiddensize), dforgetgate(hiddensize),
	dcellgate(hiddensize), dcellstate(hiddensize), dhiddenstate(hiddensize), dhidden_next(hiddensize), dcell_next(hiddensize), dconcatinated(concat_size), sampleinput(inputsize), samplenewinput(inputsize),
	sampleconcatinated(concat_size), sampleinputgate(hiddensize), sampleforgetgate(hiddensize), samplecellgate(hiddensize), sampleoutputgate(hiddensize), samplecellstate(hiddensize), samplehiddenstate(hiddensize),
	sampleyvalue(inputsize), sampleoutputvalue(inputsize), samplehiddenprev(hiddensize), samplecellprev(hiddensize)
{
	std::cout << "----------------------------------------" << std::endl;
	std::cout << "|          LSTM INITIALIZED            |" << std::endl;
	std::cout << "----------------------------------------" << std::endl;

	learning_rate = 0.25f;
	weight_mean = 0.0f;
	weight_sd = 0.15f;

	/* Randomize Weights */
	pbw::tensor_funcs::random::normal(weights[0], weight_mean + 0.5f, weight_sd);
	pbw::tensor_funcs::random::normal(weights[1], weight_mean + 0.5f, weight_sd);
	pbw::tensor_funcs::random::normal(weights[2], weight_mean, weight_sd);
	pbw::tensor_funcs::random::normal(weights[3], weight_mean + 0.5f, weight_sd);
	pbw::tensor_funcs::random::normal(out_weights.span(), weight_mean, weight_sd);

	cublasCreate(&handle);
}

void lstm::forward(float * input, int timestep)
{
	/* Copying and concatination with previous timestep */

	pbw::tensor_funcs::copy(input, inputs[timestep]);

	pbw::tensor_funcs::concatinate(hiddenstates[timestep - 1], inputs[timestep], concatinated[timestep]);

	/* Gate Updates */

	pbw::cublas::math::matrixvectormult(handle, concatinated[timestep], biases[0], forgetgates[timestep], weights[0]);
	pbw::cublas::math::matrixvectormult(handle, concatinated[timestep], biases[1], inputgates[timestep], weights[1]);
	pbw::cublas::math::matrixvectormult(handle, concatinated[timestep], biases[2], cellgates[timestep], weights[2]);
	pbw::cublas::math::matrixvectormult(handle, concatinated[timestep], biases[3], outputgates[timestep], weights[3]);

	pbw::cuda::math::activations::sigmoid(forgetgates[timestep], forgetgates[timestep]);
	pbw::cuda::math::activations::sigmoid(inputgates[timestep], inputgates[timestep]);
	pbw::cuda::math::activations::tanh(cellgates[timestep], cellgates[timestep]);
	pbw::cuda::math::activations::sigmoid(outputgates[timestep], outputgates[timestep]);

	/* Cell State Update */

	pbw::cuda::math::pointwise::multiply(forgetgates[timestep], cellstates[timestep - 1], hidden_intermediates[0]);
	pbw::cuda::math::pointwise::multiply(inputgates[timestep], cellgates[timestep], hidden_intermediates[1]);
	pbw::cublas::math::add(handle, hidden_intermediates[0], hidden_intermediates[1], cellstates[timestep]);

	/* Hidden State Update */

	pbw::cuda::math::activations::tanh(cellstates[timestep], hidden_intermediates[0]);
	pbw::cuda::math::pointwise::multiply(hidden_intermediates[0], outputgates[timestep], hiddenstates[timestep]);

	/* Output Update */

	pbw::cublas::math::matrixvectormult(handle, hiddenstates[timestep], out_biases.span(), yvalues[timestep], out_weights.span());
	pbw::cuda::math::activations::softmax(handle, yvalues[timestep], outputvalues[timestep]);
}

void lstm::backward(float * goal, int timestep)
{
	/* Copying */
	pbw::tensor_funcs::copy(goal, target_vec.span());

	/* dY = calculated - goal */
	pbw::cublas::math::subtract(handle, outputvalues[timestep], target_vec.span(), dY.span());

	/* dWY += outerproduct(dy, h) */
	pbw::cublas::math::outerproduct(handle, dY.span(), hiddenstates[timestep], dout_weights.span());

	/* dbY += dy */
	pbw::cublas::math::add(handle, dY.span(), dout_biases.span());

	/* dHidden = WY(T)(dy) */
	pbw::cublas::math::matrixvectormultT(handle, dY.span(), dhiddenstate.span(), out_weights.span());

	/* dHidden += dHiddenNext */
	pbw::cublas::math::add(handle, dhidden_next.span(), dhiddenstate.span());

	/* dOutput = dHidden * tanh(Cell) */
	pbw::cuda::math::activations::tanh(cellstates[timestep], hidden_intermediates[0]);
	pbw::cuda::math::pointwise::multiply(dhiddenstate.span(), hidden_intermediates[0], doutputgate.span());

	/* dOutput = dOutput * dsigmoid(Output) */
	pbw::cuda::math::activations::dsigmoid(outputgates[timestep], hidden_intermediates[0]);
	pbw::cuda::math::pointwise::multiply(doutputgate.span(), hidden_intermediates[0], doutputgate.span());

	/* dWOutput += outerproduct(dOutput, ConcatinatedVec) */
	pbw::cublas::math::outerproduct(handle, doutputgate.span(), concatinated[timestep], dweights[3]);

	/* dbOutput += dOutput */
	pbw::cublas::math::add(handle, doutputgate.span(), dbiases[3]);

	/* dCell =  dCellNext + dHidden * Output * dtanh(tanh(Cell)) */
	pbw::cuda::math::activations::tanh(cellstates[timestep], hidden_intermediates[0]);
	pbw::cuda::math::activations::dtanh(hidden_intermediates[0], hidden_intermediates[0]);
	pbw::cuda::math::pointwise::multiply(outputgates[timestep], hidden_intermediates[0], hidden_intermediates[0]);
	pbw::cuda::math::pointwise::multiply(dhiddenstate.span(), hidden_intermediates[0], hidden_intermediates[0]);
	pbw::cublas::math::add(handle, dcell_next.span(), hidden_intermediates[0], dcellstate.span());

	/* dCellGate = dCell * Input */
	pbw::cuda::math::pointwise::multiply(dcellstate.span(), inputgates[timestep], dcellgate.span());
	/* dCellGate = dCellGate * dtanh(CellGate) */
	pbw::cuda::math::activations::dtanh(cellgates[timestep], hidden_intermediates[0]);
	pbw::cuda::math::pointwise::multiply(dcellgate.span(), hidden_intermediates[0], dcellgate.span());

	/* dWCell += outerproduct(dCellGate, ConcatinatedVec) */
	pbw::cublas::math::outerproduct(handle, dcellgate.span(), concatinated[timestep], dweights[2]);

	/* dbCell += dCellBar */
	pbw::cublas::math::add(handle, dcellgate.span(), dbiases[2]);

	/* dInput = dCell * CellPrev */
	pbw::cuda::math::pointwise::multiply(dcellstate.span(), cellstates[timestep - 1], dinputgate.span());

	/* dInput = dsigmoid(input) * dInput */
	pbw::cuda::math::activations::dsigmoid(inputgates[timestep], hidden_intermediates[0]);
	pbw::cuda::math::pointwise::multiply(hidden_intermediates[0], dinputgate.span(), dinputgate.span());

	/* dWInput += outerproduct(dInput, ConcatinatedVec) */
	pbw::cublas::math::outerproduct(handle, dinputgate.span(), concatinated[timestep], dweights[1]);

	/* dbInput += dInput */
	pbw::cublas::math::add(handle, dinputgate.span(), dbiases[1]);

	/* dForget = dCell * CellPrev */
	pbw::cuda::math::pointwise::multiply(dcellstate.span(), cellstates[timestep - 1], dforgetgate.span());

	/* dForget = dsigmoid(Forget) * dForget */
	pbw::cuda::math::activations::dsigmoid(forgetgates[timestep], hidden_intermediates[0]);
	pbw::cuda::math::pointwise::multiply(hidden_intermediates[0], dforgetgate.span(), dforgetgate.span());

	/* dWForget += outerproduct(dForget, ConcatinatedVec) */
	pbw::cublas::math::outerproduct(handle, dforgetgate.span(), concatinated[timestep], dweights[0]);

	/* dbForget += dForget */
	pbw::cublas::math::add(handle, dforgetgate.span(), dbiases[0]);

	/* dCellNext = Forget * dCell */
	pbw::cuda::math::pointwise::multiply(forgetgates[timestep], dcellstate.span(), dcell_next.span());

	/* dConcatinated = WForget(T)(dForget) + WInput(T)(dInput) + WCell(T)(dCellBar) + WOutput(T)(dOutput) */
	pbw::cublas::math::matrixvectormultT(handle, dforgetgate.span(), concat_intermediates[0], weights[0]);
	pbw::cublas::math::matrixvectormultT(handle, dinputgate.span(), concat_intermediates[1], weights[1]);
	pbw::cublas::math::matrixvectormultT(handle, dcellgate.span(), concat_intermediates[2], weights[2]);
	pbw::cublas::math::matrixvectormultT(handle, doutputgate.span(), concat_intermediates[3], weights[3]);

	pbw::cublas::math::add(handle, concat_intermediates[0], concat_intermediates[1]);
	pbw::cublas::math::add(handle, concat_intermediates[2], concat_intermediates[3]);

	pbw::cublas::math::add(handle, concat_intermediates[1], concat_intermediates[3], dconcatinated.span());

	/* dHiddenNext = dConcatinated[hiddensize:]*/
	pbw::tensor_funcs::subset(dconcatinated.span(), dhidden_next.span(), inputsize);
}

void lstm::updateweights()
{

	pbw::cuda::math::optimizers::adagrad(weights[0], dweights[0], mweights[0], learning_rate, 0.0000001f);
	pbw::cuda::math::optimizers::adagrad(weights[1], dweights[1], mweights[1], learning_rate, 0.0000001f);
	pbw::cuda::math::optimizers::adagrad(weights[2], dweights[2], mweights[2], learning_rate, 0.0000001f);
	pbw::cuda::math::optimizers::adagrad(weights[3], dweights[3], mweights[3], learning_rate, 0.0000001f);
	pbw::cuda::math::optimizers::adagrad(out_weights.span(), dout_weights.span(), mout_weights.span(), learning_rate, 0.0000001f);
}

void lstm::updatebiases()
{
	pbw::cuda::math::optimizers::adagrad(biases[0], dbiases[0], mbiases[0], learning_rate, 0.0000001f);
	pbw::cuda::math::optimizers::adagrad(biases[1], dbiases[1], mbiases[1], learning_rate, 0.0000001f);
	pbw::cuda::math::optimizers::adagrad(biases[2], dbiases[2], mbiases[2], learning_rate, 0.0000001f);
	pbw::cuda::math::optimizers::adagrad(biases[3], dbiases[3], mbiases[3], learning_rate, 0.0000001f);
	pbw::cuda::math::optimizers::adagrad(out_biases.span(), dout_biases.span(), mout_biases.span(), learning_rate, 0.0000001f);
}

void lstm::reset_delta_next()
{
	pbw::cuda::math::zero(dhidden_next.span());
	pbw::cuda::math::zero(dcell_next.span());
}

void lstm::copy_states()
{
	pbw::tensor_funcs::copy(hiddenstates[sequencelength], hiddenstates[0]);
	pbw::tensor_funcs::copy(cellstates[sequencelength], cellstates[0]);
}

void lstm::sample(const int timesteps, char key[])
{
	auto temp_data = new float[inputsize];
	temp_data[0] = 1;
	
	pbw::tensor_funcs::copy(temp_data, sampleinput.span());
	pbw::tensor_funcs::copy(hiddenstates[0], samplehiddenprev.span());
	pbw::tensor_funcs::copy(cellstates[0], samplecellprev.span());

	for (int t = 0; t < timesteps; t++)
	{
		pbw::tensor_funcs::concatinate(samplehiddenprev.span(), sampleinput.span(), sampleconcatinated.span());

		pbw::cublas::math::matrixvectormult(handle, sampleconcatinated.span(), biases[0], sampleforgetgate.span(), weights[0]);
		pbw::cublas::math::matrixvectormult(handle, sampleconcatinated.span(), biases[1], sampleinputgate.span(), weights[1]);
		pbw::cublas::math::matrixvectormult(handle, sampleconcatinated.span(), biases[2], samplecellgate.span(), weights[2]);
		pbw::cublas::math::matrixvectormult(handle, sampleconcatinated.span(), biases[3], sampleoutputgate.span(), weights[3]);

		pbw::cuda::math::activations::sigmoid(sampleforgetgate.span(), sampleforgetgate.span());
		pbw::cuda::math::activations::sigmoid(sampleinputgate.span(), sampleinputgate.span());
		pbw::cuda::math::activations::tanh(samplecellgate.span(), samplecellgate.span());
		pbw::cuda::math::activations::sigmoid(sampleoutputgate.span(), sampleoutputgate.span());

		pbw::cuda::math::pointwise::multiply(sampleforgetgate.span(), samplecellprev.span(), hidden_intermediates[0]);
		pbw::cuda::math::pointwise::multiply(sampleinputgate.span(), samplecellgate.span(), hidden_intermediates[1]);
		pbw::cublas::math::add(handle, hidden_intermediates[0], hidden_intermediates[1], samplecellstate.span());

		pbw::cuda::math::activations::tanh(samplecellstate.span(), hidden_intermediates[0]);
		pbw::cuda::math::pointwise::multiply(hidden_intermediates[0], sampleoutputgate.span(), samplehiddenstate.span());



		pbw::cublas::math::matrixvectormult(handle, samplehiddenstate.span(), out_biases.span(), sampleyvalue.span(), out_weights.span());
		pbw::cuda::math::activations::softmax(handle, sampleyvalue.span(), sampleoutputvalue.span());

		int position;

		pbw::cublas::math::sample(handle, sampleoutputvalue.span(), position);

		//pbw::cublas::math::getmax(handle, sampleoutputvalue.span(), position);

		std::fill(&temp_data[0], temp_data + inputsize, 0);
		temp_data[position - 1] = 1;

		std::cout << key[position - 1];

		pbw::tensor_funcs::copy(temp_data, sampleinput.span());
		pbw::tensor_funcs::copy(samplehiddenstate.span(), samplehiddenprev.span());
		pbw::tensor_funcs::copy(samplecellstate.span(), samplecellprev.span());

	}

	delete[] temp_data;

	std::cout << "" << std::endl;
	std::cout << "" << std::endl;
	std::cout << "" << std::endl;
}

void lstm::clip()
{
	pbw::cuda::math::clip(dweights.span(), -1.0f, 1.0f);
	pbw::cuda::math::clip(dbiases.span(), -1.0f, 1.0f);
	pbw::cuda::math::clip(dout_weights.span(), -1.0f, 1.0f);
	pbw::cuda::math::clip(dout_biases.span(), -1.0f, 1.0f);
}