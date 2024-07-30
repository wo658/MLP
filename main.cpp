#include "MLP_Functions.h"
#include "Layer.h"
int main() {
	//Regression 
	//Data Load
	string dataPath = "C:\\Users\\ecmdev\\Desktop\\123\\MLP_Ex\\MLP_Ex\\ProcessDifference_train.csv";
	const char *NameofData = dataPath.c_str();
	vector<vector<double>> train;
	train = readFile(NameofData);
	
	//x_train/Y_train Split
	vector<vector<double>> x_train, y_train;
	splitData(train, x_train, y_train);
	

	// 입력 차원 54  출력 차원 1  , data 수 6999개
	// x_train[0]~x_train[6998]
	// 미완 -> 정규화  ,   backpropagation

	cout << x_train[0].size() << endl;
	hidden_layer h_layer1(54,32), h_layer2(32,32);
	output_layer o_layer(32,1);
	
	h_layer1.init();
	h_layer2.init();
	o_layer.init();
	double error;
	double learning_rate = 0.1;
	double y_answer=0,y_pridict=0;




	for (int i = 0; i < h_layer2.node_num; i++)
		cout << h_layer2.weight[0][i] << " ";

	for (int epho = 0; epho < 100; epho++) {

		// 1 . data 순서에 따라 삽입
		// 2 . feedfowading
		// 3 . back
		// 4 . w , b  update
		// error 특정 값 이하 종료 
		for (int i = 0; i < x_train.size(); i++) {

			y_answer = y_train[i][0];



			// 모든 데이터 x에 대하여 값 넣어주기
			for (int j = 0; j < x_train[0].size(); j++)
			{

				for (int k = 0; k < h_layer1.node_num ; k++) {

					h_layer1.values[k] = h_layer1.values[k] +  x_train[i][j] * h_layer1.weight[j][k];
					h_layer1.active_values[k] = h_layer1.values[k];

					// w 값에 따라 hidden layer1 update 이후 활성함수 적용
				}
				// 
			}

			for (int j = 0; j < h_layer1.node_num; j++)
			{
				for (int k = 0; k < h_layer2.node_num; k++) {
					h_layer2.values[k] = h_layer2.values[k] + h_layer1.active_values[k] * h_layer2.weight[j][k];
					h_layer2.active_values[k] = h_layer2.values[k];
				}
			}
			for (int j = 0; j < h_layer2.node_num; j++) 
			{
				o_layer.values[0] = o_layer.values[0] + h_layer2.values[j] * o_layer.weight[j][0];
			}


			y_pridict = o_layer.values[0];

			error = 0.5*(y_answer - y_pridict)*(y_answer - y_pridict);
			cout << error << endl;

			//  feed forwading 종료

			//  back 시작 
			//  back의 목적 => 각각의 w값들을 미분 변화율에 따라 업데이트 해주기 위함

			//  output node 의 차원값은 1 , 뒷 차원은 0으로 고정


			for (int j = 0; j < h_layer2.node_num; j++) {
			//	o_layer.weight[j][0] 업데이트

				o_layer.weight[j][0] = o_layer.weight[j][0] - learning_rate * 1 * (y_answer - y_pridict) * h_layer2.active_values[j];            /* E/w 미분 */
			}







			// value값 초기화




		}
	}







	system("pause");


	return 0;
}




