#pragma once

class layer {
	
	
public:
	layer(int pre_num, int num) : node_num(num), node_pre_num(pre_num) {
		weight.resize(pre_num);
		for (int i = 0; i < weight.size(); i++)
			weight[i].resize(num);

		values.resize(num+1);
		active_values.resize(num+1);

	};
	vector<vector<double>> weight;
	vector<double> values;
	vector<double> active_values;
	vector<vector <double>> diff;
	int node_num;
	int node_pre_num;
	void init() {
		for (int i = 0; i < node_pre_num; i++) {
			for (int j = 0; j < node_num; j++) {
				weight[i][j] = (rand()%20 - 10)/10.0 ;
			}
		}
		for (int i = 0; i < node_num; i++) {
				values[i] = 0;
			
		}



	}

};
// ���� ���� ��� ���� ���� ��� 
class output_layer : public layer{
public:
	output_layer(int pre_num,int num) : layer(pre_num,num) {
	};
};
class hidden_layer : public layer {

public:
	hidden_layer(int pre_num ,int num ) : layer(pre_num, num) {
	};


};


// 1 . wegiht ����
// 2 . 