#include <nnet.h>

using namespace std;

int layer_nodes = 6;

vector< vector<float> > input_layer (2, vector<float>(layer_nodes, 0)); 
vector< vector<float> > weight0 (layer_nodes, vector<float>(layer_nodes, 0));

vector< vector<float> > hidden_layer1 (2, vector<float>(layer_nodes, 0)); 
vector< vector<float> > weight1 (layer_nodes, vector<float>(layer_nodes, 0));

vector< vector<float> > hidden_layer2 (2, vector<float>(layer_nodes, 0)); 
vector< vector<float> > weight2 (layer_nodes, vector<float>(layer_nodes, 0));

vector< vector<float> > output_layer (2, vector<float>(layer_nodes, 0)); 

vector< vector<float> > bias_weight (3, vector<float>(layer_nodes, 0));
float bias = 1;

float learning_rate = 0.01;

int target_ans;
vector<float> target_ans_binary (layer_nodes,0);

void get_ans_in_binary ()
{
	
	int tmp_target = target_ans;
	for (int i = layer_nodes-1; i >= 0; i--)
	{
		if (tmp_target - pow(2, i) >= 0)
		{
			target_ans_binary[layer_nodes-1-i] = 1;
			tmp_target -= pow(2, i);
		}
		else
		{
			target_ans_binary[layer_nodes-1-i] = 0;
		}
	}
}

void generate_ans ()
{
	target_ans = 0;
	for (int i = 0; i < layer_nodes; i++)
	{
		input_layer[0][i] = 0;
	}
	
	for (int i = 0; i < layer_nodes; i++)
	{
		int in = rand() % 6 + 1;
		input_layer[0][in]++;
		if (in % 2 != 0)
		{
			target_ans += in-1;	
		}
	}
}

void forward_propagation ()
{
	//Layers go from i to j
	
	//sigmoid input layer
	for (int i = 0; i < layer_nodes; i++)
	{
		input_layer[1][i] = 1/(1+exp(-input_layer[0][i]));
	}
	
	//through the first set of weights
	for (int j = 0; j < layer_nodes; j++)
	{
		//reset the inputs to hidden_layer1
		hidden_layer1[0][j] = 0;
		
		for (int i = 0; i < layer_nodes; i++)
		{
			// Applying the weights
			hidden_layer1[0][j] += input_layer[1][i]*weight0[i][j];
		}
		// Adding the Bias
		hidden_layer1[0][j] += bias*bias_weight[0][j];
		
		// sigmoid hidden_layer1
		hidden_layer1[1][j] = 1/(1+exp(-hidden_layer1[0][j]));
	}
	
	for (int j = 0; j < layer_nodes; j++)
	{
		//reset the inputs to hidden_layer2
		hidden_layer2[0][j] = 0;
		
		for (int i = 0; i < layer_nodes; i++)
		{
			// Applying the weights
			hidden_layer2[0][j] += hidden_layer1[1][i]*weight1[i][j];
		}
		// Adding the Bias
		hidden_layer2[0][j] += bias*bias_weight[1][j];
		
		// sigmoid hidden_layer1
		hidden_layer2[1][j] = 1/(1+exp(-hidden_layer2[0][j]));
	}
	
	for (int j = 0; j < layer_nodes; j++)
	{
		//reset the inputs to output_layer
		output_layer[0][j] = 0;
		
		for (int i = 0; i < layer_nodes; i++)
		{
			// Applying the weights
			output_layer[0][j] += hidden_layer2[1][i]*weight2[i][j];
		}
		// Adding the Bias
		output_layer[0][j] += bias*bias_weight[2][j];
		
		// sigmoid hidden_layer1
		output_layer[1][j] = 1/(1+exp(-output_layer[0][j]));
	}
	
}

void back_propagation()
{
	// error with respect to the output_layer[1]
	vector<float> error_respect_output1 (layer_nodes, 0);
	// output_layer[1] with respect to output_layer[0]
	vector<float> output1_respect_output0 (layer_nodes, 0);
	for (int i = 0; i < layer_nodes; i++)
	{
		error_respect_output1[i] = output_layer[1][i] - target_ans_binary[i];
		
		float output = output_layer[1][i];
		output1_respect_output0[i] = output - output*output;
	}
	
	// output_layer[0] with respect to weight2 = hidden_layer2[1]
	
	// error with respect to weight2
	vector< vector<float> > error_respect_weight2 (layer_nodes, vector<float>(layer_nodes, 0));
	// error with respect to bias_weight[2]
	vector<float> error_respect_bias_weight2 (layer_nodes, 0);
	for (int j = 0; j < layer_nodes; j++)
	{
		for (int i = 0; i < layer_nodes; i++)
		{
			error_respect_weight2[i][j] = error_respect_output1[j]*output1_respect_output0[j]*hidden_layer2[1][i];
		}
		error_respect_bias_weight2[j] = error_respect_output1[j]*output1_respect_output0[j]*bias;
	}
	
	//**************************************************
	// error with respect to hidden_layer2[1]
	vector<float> error_respect_hidden_layer21 (layer_nodes, 0);
	// hidden_layer2[1] with respect to hidden_layer2[0]
	vector<float> hidden_layer21_respect_hidden_layer20 (layer_nodes, 0);
	for (int i = 0; i < layer_nodes; i++)
	{
		for (int j = 0; j < layer_nodes; j++)
		{
			error_respect_hidden_layer21[i] += weight2[i][j]*error_respect_output1[j]*output1_respect_output0[j];
		}
		float output = hidden_layer2[1][i];
		hidden_layer21_respect_hidden_layer20[i] = output - output*output;
	}
	
	// hidden_layer2[0] with respect to weight1 = hidden_layer1[1]
	
	// error with respect to weight1
	vector< vector<float> > error_respect_weight1 (layer_nodes, vector<float>(layer_nodes, 0));
	// error with respect to bias_weight[1]
	vector<float> error_respect_bias_weight1 (layer_nodes, 0);
	for (int j = 0; j < layer_nodes; j++)
	{
		for (int i = 0; i < layer_nodes; i++)
		{
			error_respect_weight1[i][j] = error_respect_hidden_layer21[j]*hidden_layer21_respect_hidden_layer20[j]*hidden_layer1[1][i];
		}
		error_respect_bias_weight1[j] = error_respect_hidden_layer21[j]*hidden_layer21_respect_hidden_layer20[j]*bias;
	}
	
	//**************************************************
	// error with respect to hidden_layer1[1]
	vector<float> error_respect_hidden_layer11 (layer_nodes, 0);
	// hidden_layer1[1] with respect to hidden_layer1[0]
	vector<float> hidden_layer11_respect_hidden_layer10 (layer_nodes, 0);
	for (int i = 0; i < layer_nodes; i++)
	{
		for (int j = 0; j < layer_nodes; j++)
		{
			error_respect_hidden_layer11[i] += weight1[i][j]*error_respect_hidden_layer21[j]*hidden_layer21_respect_hidden_layer20[j];
		}
		float output = hidden_layer1[1][i];
		hidden_layer11_respect_hidden_layer10[i] = output - output*output;
	}
	
	// hidden_layer1[0] with respect to weight0 = input_layer[1]
	
	// error with respect to weight0
	vector< vector<float> > error_respect_weight0 (layer_nodes, vector<float>(layer_nodes, 0));
	// error with respect to bias_weight[0]
	vector<float> error_respect_bias_weight0 (layer_nodes, 0);
	for (int j = 0; j < layer_nodes; j++)
	{
		for (int i = 0; i < layer_nodes; i++)
		{
			error_respect_weight0[i][j] = error_respect_hidden_layer11[j]*hidden_layer11_respect_hidden_layer10[j]*input_layer[1][i];
		}
		error_respect_bias_weight0[j] = error_respect_hidden_layer11[j]*hidden_layer11_respect_hidden_layer10[j]*bias;
	}
	
	//*********************** Calcuate new weights
	
	for (int i = 0; i < layer_nodes; i++)
	{
		for (int j = 0; j < layer_nodes; j++)
		{
			weight0[i][j] -= error_respect_weight0[i][j]*learning_rate;
			weight1[i][j] -= error_respect_weight1[i][j]*learning_rate;
			weight2[i][j] -= error_respect_weight2[i][j]*learning_rate;
		}
		bias_weight[0][i] -= error_respect_bias_weight0[i]*learning_rate;
		bias_weight[1][i] -= error_respect_bias_weight1[i]*learning_rate;
		bias_weight[2][i] -= error_respect_bias_weight2[i]*learning_rate;
	}
	
}

int main(int argc, char** argv)
{	
	ros::init(argc, argv, "nnetS");
	
	ros::NodeHandle n;
	
	rosbag::Bag bag;
	bag.open("nnet_data.bag", rosbag::bagmode::Write);
	
	long count = 0;
	//               ,  ,
	float total = 100000000;
	while(count <= total)
	{
		generate_ans();
		get_ans_in_binary();
		forward_propagation();
		back_propagation();
		
		if (count % (long)(total/10000) == 0)
		{
			test_result(count/total, bag);
		}
		
		count++;
	}
    bag.close();

	return 0;
}

void test_result (float percentage, rosbag::Bag bag)
{
	cout << "done " << percentage*100 << "%"<< endl;
	cout << "target        = " << target_ans << endl;
	
	cout << "target binary = ";
	for (int i = 0; i < layer_nodes; i++)
	{
		cout << target_ans_binary[i] << "        ";
	}
	cout << endl;
	
	cout << "result binary = ";
	for (int i = 0; i < layer_nodes; i++)
	{
		cout << output_layer[1][i] << " ";
	}
	cout << endl;
	
	cout << "error         = ";
	float max_error = 0;
	float average_error = 1;
	vector<float> square_error_vector (layer_nodes, 0);
	for (int i = 0; i < layer_nodes; i++)
	{
		float error = abs(output_layer[1][i] - target_ans_binary[i]);
		square_error_vector[i] = pow(error, 2);
		cout << pow(error, 2) << " ";
		average_error *= 1-error;
		if (error > max_error)
		{
			max_error = error;
		}
		
	}
	cout << endl;
	
	cout << "max error = " << max_error << endl;
	cout << "average accuracy = " << 1.0-average_error << endl;
	
	std_msgs::Float32 target_ans_msg;
	target_ans_msg.data = target_ans;
	bag.write("target", ros::Time::now(), target_ans_msg);
	
	std_msgs::Float32MultiArray target_ans_binary_msg;
	target_ans_binary_msg.data = target_ans_binary;
	bag.write("target binary", ros::Time::now(), target_ans_binary_msg);
	
	std_msgs::Float32MultiArray output_layer_msg;
	output_layer_msg.data = output_layer[1];
	bag.write("result binary", ros::Time::now(), output_layer_msg);
	
	std_msgs::Float32MultiArray error_msg;
	error_msg.data = square_error_vector;
	bag.write("error", ros::Time::now(), error_msg);
	
	std_msgs::Float32 max_error_msg;
	max_error_msg.data = max_error;
	bag.write("max error", ros::Time::now(), max_error_msg);
	
	std_msgs::Float32 average_error_msg;
	average_error_msg.data = 1.0-average_error;
	bag.write("average accuracy", ros::Time::now(), average_error_msg);
	//*/
}

