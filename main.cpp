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


	hidden_layer h_layer1(x_train.size(),256), h_layer2(256,256);
	output_layer o_layer(256,1);
	
	h_layer1.init_weight();
	h_layer2.init_weight();
	
	o_layer.init_weight();

	cout << h_layer1.node_num << endl;
	cout << o_layer.node_pre_num << endl;

	system("pause");


	return 0;
}




