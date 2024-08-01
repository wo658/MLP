#include "MLP_Functions.h"
#include "Layer.h"

#include <windows.h>
double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}
int main() {
	//Regression 
	//Data Load
	string dataPath = "C:\\Users\\ecmdev\\Desktop\\123\\MLP_Ex\\MLP_Ex\\ProcessDifference_train.csv";
	string dataPath_test = "C:\\Users\\ecmdev\\Desktop\\123\\MLP_Ex\\MLP_Ex\\ProcessDifference_test.csv";
	const char* NameofData = dataPath.c_str();
	const char* testData = dataPath.c_str();
	vector<vector<double>> train;
	vector<vector<double>> test;
	train = readFile(NameofData);
	test = readFile(testData);


	//x_train/Y_train Split
	vector<vector<double>> x_train, y_train;
	vector<vector<double>> x_test, y_test;
	splitData(train, x_train, y_train);
	splitData(test, x_test, y_test);


	// 입력 차원 54  출력 차원 1  , data 수 6999개
	// x_train[0]~x_train[6998]
	//정규화  ,   backpropagation

	// cout << x_train[0].size() << endl;
	hidden_layer h_layer1(54, 32), h_layer2(32, 16);
	output_layer o_layer(16, 1);


	double error;
	
	double y_answer = 0, y_predict = 0;





	// 정규화 최대 최소 범위 구하기
	double max = x_train[0][0], min = x_train[0][0];
	for (int i = 0; i < x_train.size(); i++)
		for (int j = 0; j < x_train[0].size(); j++) {
			if (max < x_train[i][j])
				max = x_train[i][j];
			if (min > x_train[i][j])
				min = x_train[i][j];
		}

	//cout << max << " " << min << endl;
	// 정규화 0 ~ 1 
	for (int i = 0; i < x_train.size(); i++) {
		for (int j = 0; j < x_train[0].size(); j++) {
			x_train[i][j] = (x_train[i][j] - min) / (max - min);
		}

		// 선택사항 1 y 정규화
		// 1 . 회귀 문제의 경우 타겟값이 넓은 범위에 걸쳐 있을 때 정규화
		y_train[i][0] = (y_train[i][0] - min) / (max - min);
	}


	h_layer1.init();
	h_layer2.init();
	o_layer.init();

	////////////////////////////////	 train      ////////////////////////////////////////////

	for (int epho = 0; epho < 10; epho++) {
		for (int i = 0; i < x_train.size(); i++) {
			y_answer = y_train[i][0];

			// feedforward
			h_layer1.feedforward(x_train[i], SIGMOID);
			h_layer2.feedforward(h_layer1.active_values, SIGMOID);
			o_layer.feedforward(h_layer2.active_values, LINEAR);

			//y_predict = o_layer.active_values[0];
			y_predict = o_layer.values[0];
			error = (y_predict - y_answer) * (y_predict - y_answer);

			// output 노드에 활성화 함수를 적용시킬것인지


			// backpropagation
			//double delta = (y_predict - y_answer) * y_predict * (1 - y_predict);

			o_layer.backpropagation(y_answer, y_predict, h_layer2.active_values, LINEAR);
			h_layer2.backpropagation(o_layer.diff, h_layer1.active_values, SIGMOID);
			h_layer1.backpropagation(h_layer2.diff, x_train[i], SIGMOID);



			/*

			
			double delta = (y_predict - y_answer) * 1;
			for (int j = 0; j < h_layer2.node_num; j++) {
				o_layer.weight[j][0] -= learning_rate * delta * h_layer2.active_values[j];
				h_layer2.diff[j] = delta * o_layer.weight[j][0];
			}
			o_layer.bias -= learning_rate * delta;






			for (int k = 0; k < h_layer2.node_num; k++) {
				double delta2 = h_layer2.diff[k] * h_layer2.active_values[k] * (1 - h_layer2.active_values[k]);
				for (int j = 0; j < h_layer1.node_num; j++) {
					h_layer2.weight[j][k] -= learning_rate * delta2 * h_layer1.active_values[j];
					h_layer1.diff[j] += delta2 * h_layer2.weight[j][k];
				}
				h_layer2.bias -= learning_rate * delta2;
			}

			for (int k = 0; k < h_layer1.node_num; k++) {
				double delta3 = h_layer1.diff[k] * h_layer1.active_values[k] * (1 - h_layer1.active_values[k]);
				for (int j = 0; j < x_train[0].size(); j++) {
					h_layer1.weight[j][k] -= learning_rate * delta3 * x_train[i][j];
				}
				h_layer1.bias -= learning_rate * delta3;
			}
			*/

			//if (i % 100 == 0) {
			//	cout << "error :" << error << endl;
			//	Sleep(100);
			//}
		}
		cout << "error :" << error << endl;
		Sleep(1000);
	}

	////////////////////////////////////////////////	Test	 ///////////////////////////////////////


	// 정규화 0 ~ 1 
	for (int i = 0; i < x_test.size(); i++) {
		for (int j = 0; j < x_test[0].size(); j++) {
			x_test[i][j] = (x_test[i][j] - min) / (max - min);
		}

		// 1 . 회귀 문제의 경우 타겟값이 넓은 범위에 걸쳐 있을 때 정규화
		y_test[i][0] = (y_test[i][0] - min) / (max - min);
	}


	for (int i = 0; i < x_test.size(); i++) {
		// value 초기화
		fill(h_layer1.values.begin(), h_layer1.values.end(), 0.0);
		fill(h_layer2.values.begin(), h_layer2.values.end(), 0.0);
		o_layer.values[0] = 0.0;

		y_answer = y_test[i][0];
		h_layer1.feedforward(x_test[i], SIGMOID);
		h_layer2.feedforward(h_layer1.active_values, SIGMOID);
		o_layer.feedforward(h_layer2.active_values, LINEAR);

		//y_predict = o_layer.active_values[0];
		y_predict = o_layer.values[0];
		error = (y_predict - y_answer) * (y_predict - y_answer);
		cout << "error : " << error << endl;
	}


	system("pause");


	return 0;
}
