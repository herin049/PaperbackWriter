#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "include/tensor/tensor.hpp"
#include "include/tensor/tensor_span.hpp"
#include "include/cuda/kernels/pointwise/multiply.cuh"
#include "include/tensor/tensor_random.hpp"
#include "include/cuda/tensor/cuda_funcs.cuh"
#include "include/cublas/cublas_funcs.cuh"
#include "include/tensor/debug/print.hpp"
#include "cublas_v2.h"
#include "cublas_api.h"
#include "include/ai/lstm/lstm.hpp"
#include <fstream>
#include <unordered_map>
#include <set>
#include <iostream>

int main()
{
	std::fstream input_data;
	input_data.open("C:/Users/1019l/Desktop/pbw/PaperbackWriter/tests/data.txt");
	std::set<char> unique_buffer = { std::istreambuf_iterator<char>(input_data), std::istreambuf_iterator<char>() };
	std::unordered_map<char, std::size_t> data_map;
	for (std::set<char>::iterator index = unique_buffer.begin(); index != unique_buffer.end(); ++index)
	{
		data_map.insert({ *index,std::distance(unique_buffer.begin(),index) });
	}
	std::cout << "UNIQUE CHARACTERS: " << data_map.size() << std::endl;
	int unique_chars = data_map.size();
	char vals[89];
	for (auto elem : data_map)
	{
		vals[elem.second] = elem.first;
	}
	std::vector<int> positions;
	char ch;
	std::fstream fin("C:/Users/1019l/Desktop/pbw/PaperbackWriter/tests/data.txt", std::fstream::in);
	while (fin >> std::noskipws >> ch) {
		positions.push_back(data_map.at(ch));
	}

	lstm network;
	size_t counter = 0;
	size_t epoch = 0;

	while (true)
	{
		if ((counter + 101 > positions.size()))
		{
			counter = 0;
			epoch++;
		}
		for (int i = 1; i < 101; i++)
		{
			float * input = new float[89];
			std::fill(&input[0], input + 89, 0);
			input[positions.at(counter + i - 1)] = 1;
			network.forward(input, i);
		}
		for (int k = 100; k > 0; k--)
		{
			float * target = new float[89];
			std::fill(&target[0], target + 89, 0);
			target[positions.at(counter + k)] = 1;
			network.backward(target, k);
		}

		network.clip();
		network.updateweights();
		network.updatebiases();
		network.copy_states();
		network.reset_delta_next();
		counter += 100;
		if ((counter % 10000) == 0)
		{
			std::cout << "----------------------------" << std::endl;
			std::cout << "EPOCH: " << epoch << std::endl;
			std::cout << "ITERATION: " << counter << std::endl;
			std::cout << "----------------------------" << std::endl;
			network.sample(1000, vals);
		}
	}

	std::cin.get();

	return 0;
}