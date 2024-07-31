#include "MLP_Functions.h"
#include "Layer.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
int main() {
    //Regression 
    //Data Load
    string dataPath = "C:\\Users\\whdgn\\source\\repos\\Project1\\ProcessDifference_train.csv";
    const char* NameofData = dataPath.c_str();
    vector<vector<double>> train;
    train = readFile(NameofData);

    //x_train/Y_train Split
    vector<vector<double>> x_train, y_train;
    splitData(train, x_train, y_train);


    // 입력 차원 54  출력 차원 1  , data 수 6999개
    // x_train[0]~x_train[6998]
    // 미완 -> 정규화  ,   backpropagation

    // cout << x_train[0].size() << endl;
    hidden_layer h_layer1(54, 32), h_layer2(32, 16);
    output_layer o_layer(16, 1);


    double error;
    double learning_rate = 0.1;
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

    cout << max << " " << min << endl;
    // 정규화 0 ~ 1 
    for (int i = 0; i < x_train.size(); i++) {
        for (int j = 0; j < x_train[0].size(); j++) {
            x_train[i][j] = (x_train[i][j] - min) / (max - min);
        }
        y_train[i][0] = (y_train[i][0] - min) / (max - min);
    }


    h_layer1.init();
    h_layer2.init();
    o_layer.init();

    /////////////////////////////////////////////////////////////////////////////

    for (int epho = 0; epho < 100; epho++) {
        for (int i = 0; i < x_train.size(); i++) {
            y_answer = y_train[i][0];

            // feedforward
            for (int k = 0; k < h_layer1.node_num; k++) {
                h_layer1.values[k] = h_layer1.bias;
                for (int j = 0; j < x_train[0].size(); j++) {
                    h_layer1.values[k] += x_train[i][j] * h_layer1.weight[j][k];
                }
                h_layer1.active_values[k] = sigmoid(h_layer1.values[k]);
            }

            for (int k = 0; k < h_layer2.node_num; k++) {
                h_layer2.values[k] = h_layer2.bias;
                for (int j = 0; j < h_layer1.node_num; j++) {
                    h_layer2.values[k] += h_layer1.active_values[j] * h_layer2.weight[j][k];
                }
                h_layer2.active_values[k] = sigmoid(h_layer2.values[k]);
            }

            o_layer.values[0] = o_layer.bias;
            for (int j = 0; j < h_layer2.node_num; j++) {
                o_layer.values[0] += h_layer2.active_values[j] * o_layer.weight[j][0];
            }
            o_layer.active_values[0] = sigmoid(o_layer.values[0]);

            y_predict = o_layer.active_values[0];
            error = (y_predict - y_answer);


            // backpropagation
            double delta = (y_predict - y_answer) * y_predict * (1 - y_predict);
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

            // value 초기화
            fill(h_layer1.values.begin(), h_layer1.values.end(), 0.0);
            fill(h_layer2.values.begin(), h_layer2.values.end(), 0.0);
            o_layer.values[0] = 0.0;

            cout << "error :" << error << endl;
        }
        cout << "error :" << error << endl;
    }






    system("pause");


    return 0;
}



