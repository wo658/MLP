#include "MLP_Functions.h"
#include "Layer.h"

double sigmoid(double x) {
	return -log(1 + exp(-x));
}
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
	

	// �Է� ���� 54  ��� ���� 1  , data �� 6999��
	// x_train[0]~x_train[6998]
	// �̿� -> ����ȭ  ,   backpropagation

	cout << x_train[0].size() << endl;
	hidden_layer h_layer1(54,32), h_layer2(32,16);
	output_layer o_layer(16,1);
	

	double error;
	double learning_rate = 0.1;
	double y_answer=0,y_predict=0;






	// ����ȭ �ִ� �ּ� ���� ���ϱ�
	double max=0, min=0;
	for (int i=0;i < x_train.size() ; i++)
		for (int j = 0; j < x_train[0].size(); j++) {
			if (max < x_train[i][j])
				max = x_train[i][j];
			if (min > x_train[i][j])
				min = x_train[i][j];
		}

	cout << max << " " << min << endl;
	// 3087   ,  - 0.99
	// 0 ~ 1 
	for (int i = 0; i < x_train.size(); i++) {
		for (int j = 0; j < x_train[0].size(); j++) {
			x_train[i][j] = (x_train[i][j] - min) / (max - min);

		}
		y_train[i][0] = (y_train[i][0] - min) / (max - min);
	}

	for (int i = 0; i < h_layer2.node_num; i++)
		cout << h_layer2.weight[0][i] << " ";


	h_layer1.init();
	h_layer2.init();
	o_layer.init();

	/////////////////////////////////////////////////////////////////////////

	for (int epho = 0; epho < 100; epho++) {

		// 1 . data ������ ���� ����
		// 2 . feedfowading
		// 3 . back
		// 4 . w , b  update
		// error Ư�� �� ���� ���� 
		for (int i = 0; i < x_train.size(); i++) {

			y_answer = y_train[i][0];



			// ��� ������ x�� ���Ͽ� �� �־��ֱ�
			for (int j = 0; j < x_train[0].size(); j++)
			{

				for (int k = 0; k < h_layer1.node_num ; k++) {

					x_train[i][j] = sigmoid(x_train[i][j]);
					h_layer1.values[k] = h_layer1.values[k] +  x_train[i][j] * h_layer1.weight[j][k];
					h_layer1.active_values[k] = sigmoid(h_layer1.values[k]);

					// w ���� ���� hidden layer1 update ���� Ȱ���Լ� ����
				}
				// 
			}

			for (int j = 0; j < h_layer1.node_num; j++)
			{
				for (int k = 0; k < h_layer2.node_num; k++) {
					h_layer2.values[k] = h_layer2.values[k] + h_layer1.active_values[k] * h_layer2.weight[j][k];
					h_layer2.active_values[k] = sigmoid(h_layer2.values[k]);
				}
			}
			for (int j = 0; j < h_layer2.node_num; j++) 
			{
				o_layer.values[0] = o_layer.values[0] + h_layer2.values[j] * o_layer.weight[j][0];
				
			}
			/**/
			o_layer.active_values[0] = sigmoid(o_layer.values[0]);

			y_predict = o_layer.active_values[0];
			/**/

			error = (y_predict - y_answer) * (y_predict - y_answer);
			cout << "error :" <<error << endl;

			//  feed forwading ����

			//  back ���� 
			//  back�� ���� => ������ w������ �̺� ��ȭ���� ���� ������Ʈ ���ֱ� ����

			//  output node �� �������� 1 , �� �迭���� 0���� ����


			for (int j = 0; j < h_layer2.node_num; j++) {
			//	o_layer.weight[j][0] ������Ʈ
				//cout << "old :: " << o_layer.weight[j][0] << endl;
				o_layer.weight[j][0] = o_layer.weight[j][0] - (learning_rate * (1 - sigmoid(o_layer.active_values[0])) * sigmoid(o_layer.active_values[0]) * (-y_answer +y_predict) * h_layer2.active_values[j]);            /* E/w �̺� */
				//cout << "new ::" << o_layer.weight[j][0] << endl;
				// ���� backpropagation �� ���� ��
				h_layer2.diff[j] = (1 * (-y_answer + y_predict) * o_layer.weight[j][0]) ;

			}

			//������ 1
			for (int j = 0; j < h_layer1.node_num; j++) {
				for (int k = 0; k < h_layer2.node_num; k++) {
					//������Ʈ
					cout << "old :: " << h_layer2.weight[j][k] << endl;
					h_layer2.weight[j][k] = h_layer2.weight[j][k] - (learning_rate * sigmoid(h_layer2.active_values[j]) * ( 1 - sigmoid(h_layer2.active_values[j])) * (h_layer2.diff[k])  * h_layer1.active_values[j]);
					cout << "new ::" << h_layer2.weight[j][k] << endl;;
					// ���� backpropagation �� ���� ��
					h_layer1.diff[j] = h_layer1.diff[j] + (1 * (h_layer2.diff[k])* (1 - sigmoid(h_layer2.values[j]) ) * sigmoid(h_layer2.values[j]));
			
				}
			}

			// ������ 2
			for (int j = 0; j < x_train[0].size() ; j++) {
				for (int k = 0; k < h_layer1.node_num; k++) {
					//������Ʈ
					h_layer1.weight[j][k] = h_layer1.weight[j][k] - (learning_rate * sigmoid(h_layer1.active_values[j]) * (1 - sigmoid(h_layer1.active_values[j])) * (h_layer1.diff[k]) );
					// ���� backpropagation �� ���� ��
					//h_layer1.diff[j] = h_layer1.diff[j] + (1 * (h_layer2.diff[k])  * h_layer2.weight[j][k]);

				}
			}



			// value�� �ʱ�ȭ

			for (int j = 0; j < h_layer1.node_num; j++)
				h_layer1.values[j] = 0.0;

			for (int j = 0; j < h_layer2.node_num; j++)
				h_layer2.values[j] = 0.0;

			o_layer.values[0] = 0.0;

			}
	}







	system("pause");


	return 0;
}



