#pragma once

enum activefunc {
	SIGMOID,
	LINEAR
};

double sigmoid_derivative(double value) {
	return value * (1.0 - value);
}
double linear_derivative(double vlaue) {
	return 1;
}
double learning_rate = 0.0005;
class layer {


public:
	layer(int pre_num, int num) : node_num(num), node_pre_num(pre_num) {
		weight.resize(pre_num);
		for (int i = 0; i < weight.size(); i++)
			weight[i].resize(num);

		values.resize(num);
		active_values.resize(num);
		diff.resize(pre_num);
	};
	vector<vector<double>> weight;
	vector<double> values;
	vector<double> active_values;
	vector <double> diff;
	int node_num;
	int node_pre_num;
	double bias;
	double sigmoid(double x) {
		return 1.0 / (1.0 + exp(-x));
	}
	virtual void backpropagation() {};
	
	void init() {
		for (int i = 0; i < node_pre_num; i++) {
			for (int j = 0; j < node_num; j++) {
				weight[i][j] = (rand() % 20 - 10) / 10.0;
				//weight[i][j] = 0;
			}
		}
		for (int i = 0; i < node_num; i++) {
			values[i] = 0;
			diff[i] = 0;
			active_values[i] = 0;
		}

		bias = 3;

	}
	void feedforward(vector<double>left_active_values,activefunc active_func_name) {
		for (int j = 0; j < node_num; j++) {
			values[j] = bias;
			for (int i = 0; i < node_pre_num; i++) 
				values[j] += left_active_values[i] * weight[i][j];

			if (active_func_name == SIGMOID)
				active_values[j] = sigmoid(values[j]);
			else if (active_func_name == LINEAR)
				active_values[j] = values[j];
		}
}

};



// 행이 이전 노드 열이 다음 노드 
class output_layer : public layer {
public:
	output_layer(int pre_num, int num) : layer(pre_num, num) {
	};

	void backpropagation(double answer, double predict, vector<double>left_active_values,activefunc active_func_name) {
		double delta;
		if (active_func_name == SIGMOID)
			delta = (predict - answer) * sigmoid_derivative(predict);
		else if (active_func_name == LINEAR)
			delta = (predict - answer) * linear_derivative(predict);

		for (int j = 0; j < node_pre_num; j++) {
			weight[j][0] -= learning_rate * delta * left_active_values[j];
			diff[j] = delta * weight[j][0];
		}
		bias -= learning_rate * delta;

		
	};

};
class hidden_layer : public layer {

public:
	hidden_layer(int pre_num, int num) : layer(pre_num, num) {
	};

	void backpropagation(vector<double> right_diff,vector<double>left_active_values,activefunc active_func_name) {
		double delta;
		for (int k = 0; k < node_num; k++) {
			if (active_func_name == SIGMOID)
				delta = right_diff[k] * sigmoid_derivative(active_values[k]);
			else if (active_func_name == LINEAR)
				delta = right_diff[k] * linear_derivative(active_values[k]);
			else
				cout << "WWWWWW";
			for (int j = 0; j < node_pre_num; j++) {
				weight[j][k] -= learning_rate * delta * left_active_values[j];
				diff[j] += delta * weight[j][k];
			}
			bias -= learning_rate * delta;
		}
	};

};


// 1 . wegiht 지정
// 2 . #pragma once

// back 기준 이전레이어의 Diff값 