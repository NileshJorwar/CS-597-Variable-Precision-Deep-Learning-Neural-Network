
#include<stdio.h>
#include<stdlib.h>
#include<random>
#include<assert.h>
#include<iostream>
#include<vector>
#include<math.h>
#include<string.h>
#include<fnmatch.h>
#include "half.hpp"
#include <sys/time.h>
#include<immintrin.h>
#include <pthread.h>
#include<thread>

#define NUM_THREADS 4


using std::vector;
using std::cout;
using std::endl;
using half_float::half;

using namespace std;



	half get_random() {

		
		return static_cast<half> (rand() % 100 + (-50)) / static_cast<half> (50);
		//return 0.00127;
	}


	std::vector<half> init_weights(int rows, int cols) {
	/*
	*Initialialising weights using normal distribution between the range of 0,1 
	*/	
	unsigned seed = 10;
	
	
		std::vector<half> temp(rows*cols);

		for (int i = 0; i < temp.size(); ++i) {

				temp[i] = get_random();
		}

		return temp;
	}

	std::vector<half> init_weights_bias(int rows, int cols) {
	/*
	* Initialise Biases using normal distribution
	*/
		unsigned seed = 100;
		
		std::vector<half> temp(rows*cols);

		for (int i = 0; i < temp.size(); ++i) {

				temp[i] = get_random();
		}

		return temp;
	}

		const half n1(784);// Layer 1 (Input layer) neuron count
		const half n2(30);// Layer 2 (Hidden layer) neuron count
		const half n3(10);// Layer 3 (Output layer) neuron count
		const half l2_rows(n2);// count of layer 2 rows
		const half l2_cols(n1);// count of layer 2 columns
		const half l3_rows(n3);// count of layer 3 rows 
		const half l3_cols(n2);// count of layer 3 columns
		const int epoch = 30;// Epoch count

		std::vector<half>  L2W;// Layer 2 Weights vector
		std::vector<half> L3W;// Layer 3 Weights vector
		std::vector<half> L2B;// Layer 2 Bias vector
		std::vector<half> L3B;// Layer 3 Bias vector

		std::vector<half> A2,A3;// Activation Vectors for hidden and output layers
		std::vector<half> A1(n1);// Activation vector for input layer

		std::vector<half> z2,z3;// z is product of weights and input(previous activations) and sum of bias 

		std::vector<half> d2,d3;// delta error for 2 and 3 layers

		std::vector<half> L2nw(n2*n1);// Layer 2 Weights vector
		std::vector<half> L3nw(n3*n2);// Layer 3 Weights vector
		std::vector<half> L2nb(n2*1);// Layer 2 Bias vector
		std::vector<half> L3nb(n3*1);// Layer 3 Bias vector
		std::vector<vector <half> > evaluation_mat (n3);// 10 * 10 matrix for storage of confusion matrix
		int tot_test_set = 10000;// Total number of test dataset
		int tot_train_set = 60000;// Total number of train dataset
		std::vector<half> error_calc (n3);// Calculate total error after every epoch
		 
	vector<half> error_mat (epoch);// end d3 error for each epoch 


std::vector<half> mod_output(vector<half> x) {
	/*
	*	Returns vector of 10 elements with 1 on the class label 
	*/	
	
	half temp_max;
	int location = 0;
	temp_max = x[0];

	for (int i = 1; i < x.size(); ++i) {
		if (x[i] > temp_max) {
			temp_max = x[i];
			location = i ;
		}
	}
	vector<half> ret (n3);
		for (int g = 0; g < ret.size(); ++g){
			ret[g] = 0;
		}
	
	ret[location] = 1.0;
	return ret;
}

int ind_identifier(vector<half> y) {
/*
*returns integer as a identifier for class label during testing
*/
	half max;
	int loc = 0;
	max = y[0];

	for (int i = 1; i < y.size(); ++i) {
		if (y[i] > max) {
			max = y[i];
			loc = i ;
			
		}
	}
	return loc ;

}


 std::vector<half> operator-(const std::vector<half>& v1, const std::vector<half>& v2){

// Operator overload for vector subtraction

	 long int vec_size = v1.size();
	 std::vector<half> res(vec_size);

	 for(int i=0;i<vec_size;i++){
		 res[i] = v1[i] - v2[i];

	 }


	 	 return res;

 }

 std::vector<half> operator+(const std::vector<half>& v1, const std::vector<half>& v2){


 // Operator overload for vector addition	


	 long int vec_size = v1.size();
	 std::vector<half> res(vec_size);

	 for(int i=0;i<vec_size;i++){
		 res[i] = v1[i]+v2[i];
	 }

	 	 return res;
 }

 std::vector<half> operator*(const std::vector<half>& v1, const std::vector<half>& v2){

 	// Operator overload for vector multiplication 

	 long int vec_size = v1.size();
	 std::vector<half> res(vec_size);

	 for(int i=0;i<vec_size;i++){
		 res[i] = v1[i]*v2[i];
	 }

	 return res;
}


vector<half> vec_division(vector<half> a, vector<half> b) {
	
	// Vector division function

	vector<half> c(a.size());

	for (int i = 0; i < a.size(); ++i) {
		c[i] = a[i] / b[i];
	}
	return c;
}


 std::vector<half> sigmoid(const std::vector<half>& v1){

 	// Gives activation for each vector 
 	// Using Sigmoid Activation

	 long int vec_size = v1.size();
	 std::vector<half> res(vec_size);

	 for(unsigned i=0;i<vec_size;i++){

		 res[i]= 1.0 / (1.0 + expf(-v1[i]));   // defines the sigmoid function
	 }

	 return res;
}


 std::vector <half> sigmoid_d (const std::vector <half>& m1) {

 	/*
 	* Function for derivative of sigmoid function 
 	*/

     const unsigned long VECTOR_SIZE = m1.size();
     std::vector <half> output (VECTOR_SIZE);
	vector <half> temp (VECTOR_SIZE);
	for (int h = 0 ; h < VECTOR_SIZE;++h){
		temp[h] = 1;
	}
	

	output = sigmoid(m1)*(temp - sigmoid(m1));
     return output;
 }
 
 /////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////


std::vector<half> dot_product; //Vector Used to store value in dot product

void dot_thread (const std::vector <half>& m1, const std::vector <half>& x,
                     const int m1_rows, const int m1_columns, const int m2_columns, const int m1_start) {

 	/*
 	* Dot product between 2 matrices Multi-threaded
 	*/
	//printf("Start Row is: %d\n", m1_start);
	//std::vector <float> output (m1_rows*m2_columns);
	
	//pthread_t threads[NUM_THREADS];
	int end = m1_start+m1_rows;
	//int i=0;
	
	 
	 for (int i=m1_start; i<end;i++){
		for (int j=0; j<m2_columns;j++){
			dot_product[i*m2_columns+j]=0;
			for (int k=0;k<m1_columns;k++){
				dot_product[i*m2_columns+j]+=(m1[i*m1_columns+k]*x[j*m1_columns+k]);
			}
		}
	  }

	 /*std::vector <float> output (m1_rows*m2_columns);

     for( int row = 0; row != m1_rows; ++row ) {

         for( int col = 0; col != m2_columns; ++col ) {

             output[ row * m2_columns + col ] = 0.f;

             for(int k = 0; k != m1_columns; ++k ) {

                 output[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
             }
         }
     }*/

     //return output;
 }
 
 std::vector <half> dot (const std::vector <half>& m1, const std::vector <half>& m2,
                     const int m1_rows, const int m1_columns, const int m2_columns) {

 	//
 	// Dot product between 2 matrices
 	//
	//std::vector <float> output_thread;
	std::vector <half> x (m2_columns*m1_columns);
	 for(int i=0;i<m2_columns;i++){
		 for(int j=0;j<m1_columns;j++){
			 x[i*m1_columns+j]=m2[j*m2_columns+i];
		 }
	 }
	std::thread threads[NUM_THREADS];

	 //std::vector <float> output (m1_rows*m2_columns);
	 //printf("before\n");
	 dot_product.resize(m1_rows*m2_columns);
	 //printf("after\n");
	 int row=0;
	 int i=0;
	 while(row<m1_rows){
		 //printf("row is: %d ", row);
		 int tid=0;
		 for (tid=0;tid<NUM_THREADS;tid++){
			 int rows =std::min(12,m1_rows-row);
			threads[tid]=std::thread(dot_thread,m1, x, rows, m1_columns, m2_columns, row);
			
			if (rows==m1_rows-row){
				row+=rows;
				//printf("Final Rows is: %d\n", rows);
				tid+=1;
				break;
			}
			row+=12;
		 }
			//printf("tid is: %d\n",tid);
		 for (int t=0;t<tid;t++){
			threads[t].join();
		 }
		 //i+=1;
		 //printf("I is: %d",i);
		 
	 }
	 return dot_product;


/*
	std::vector <half> output (m1_rows*m2_columns);
	 const int num_trails = 1;
	 const int col_reduced = m1_columns - m1_columns%64;
	 const int col_reduced_32 = m1_columns - m1_columns%32;
	 float scratchpad[8];
	 std::vector <half> x (m2_columns*m1_columns);
	 for(int i=0;i<m2_columns;i++){
		 for(int j=0;j<m1_columns;j++){
			 x[i*m1_columns+j]=m2[j*m2_columns+i];
		 }
	 }
	 
	 for (int i=0; i<m1_rows;i++){
		for (int j=0; j<m2_columns;j++){
			output[i*m2_columns+j]=0;
			for (int k=0;k<m1_columns;k++){
				output[i*m2_columns+j]+=(m1[i*m1_columns+k]*x[j*m1_columns+k]);
			}
		}
	  }
	 
	
     return output;*/
 }

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/*
 std::vector <half> dot (const std::vector <half>& m1, const std::vector <half>& m2,
                     const int m1_rows, const int m1_columns, const int m2_columns) {

 	//
 	// Dot product between 2 matrices
 	///
	std::vector <half> output (m1_rows*m2_columns);
	 const int num_trails = 1;
	 const int col_reduced = m1_columns - m1_columns%64;
	 const int col_reduced_32 = m1_columns - m1_columns%32;
	 float scratchpad[8];
	 std::vector <half> x (m2_columns*m1_columns);
	 for(int i=0;i<m2_columns;i++){
		 for(int j=0;j<m1_columns;j++){
			 x[i*m1_columns+j]=m2[j*m2_columns+i];
		 }
	 }
	 
	 for (int i=0; i<m1_rows;i++){
		for (int j=0; j<m2_columns;j++){
			output[i*m2_columns+j]=0;
			for (int k=0;k<m1_columns;k++){
				output[i*m2_columns+j]+=(m1[i*m1_columns+k]*x[j*m1_columns+k]);
			}
		}
	  }
	 
	
     return output;
 }
 */
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

 std::vector<half> transpose (half *m, const int C, const int R) {

 	/* 
 	*	Transpose of Matrix	
 	*/

     std::vector<half> mT (C*R);

     for(int n = 0; n!=C*R; n++) {
         int i = n/C;
         int j = n%C;
         mT[n] = m[R*j + i];
     }

     return mT;
 }


 void print ( const vector <half>& v1, int v1_rows, int v1_columns ) {

 	/*
 	* Print function for 1-D representation of a matrix
 	*/
	 for( int i = 0; i != v1_rows; ++i ) {
         for( int j = 0; j != v1_columns; ++j ) {
             cout << v1[ i * v1_columns + j ] << " ";
         }
         cout << '\n';
     }
     cout << endl;
 }

void print_vectors(vector<vector<half> > pvc) {

	/*
	* Print function for 2-D representation of matrix
	*/

	for (int g = 0; g < pvc.size(); ++g) {
		for (auto y = pvc[g].begin(); y != pvc[g].end(); ++y) {
			//printf("%.2f\t", *y);
			cout << *y << '\t';
		}
		cout << endl;
	}

}

 std::vector<half> getlabelVector(int n){

 	/* 
 	* Get label vector of image with an input
 	*/
	 std::vector<half> res (n3);
	 for (int y = 0 ; y < res.size();++y){
	 		res[y] = 0;
	 }

	 res[n]=1;

	 return res;
 }

 vector<half> update_wandb (vector <half> &a , vector <half> &b, int num){

 	/*
 	*	Update the weights and biases matrices of all network layers after every mini batch
 	*
 	*/
	 half eta(0.03);
	 int mini_batch_size = 10;
	vector <half> res (num);

 	//a = a - ((eta/mini_batch_size) * b);
	 for (int i = 0 ; i < res.size();++i){
 		b[i] = (eta/mini_batch_size) * b[i];
		res[i] = a[i] - b[i];	
	 }
	//cout << endl << endl << " REACHED HERE###################" << endl << endl;	
	return res;
	
 	}


void reinit (vector<half> &v){
	
	/*
	*	Re initialise the delta vectors after every mini batches
	*/	
	for (int i = 0;i < v.size();++i){
		v[i] = 0.0;
	}

}

int main(){
	
	FILE *fp;
	//char filename1[] = "/home/cc/mnist_train.csv";// training input file link
	//char filename2[] = "/home/cc/mnist_test.csv";// testing input file link
	char filename1[] = "../Data/mnist_train_1.csv";// training input file link
	char filename2[] = "../Data/mnist_test_1.csv";// testing input file link
	
	FILE *write_out ;// file pointer for output file
	//write_out = fopen("/home/cc/training-report-s-F16.txt","w");// file link to write the output of model and testing results
	write_out = fopen("../Data/F16_new.txt","w");// file link to write the output of model and testing results
	
	char buff[12000];// buffer for reading from the csv file
	half activation[785];// input buffer holding vector
	unsigned i=0;
	char * token;	
	struct timeval start_time, middle_time, end_time;
	int lab=0, batch_size=0;

		/* 
		* INITIALISING ALL WEIGHTS AND BIASES OF THE NETWORK 
		*/

		L2W = init_weights(l2_rows, l2_cols);// initializing wieghts for layer 2
		
		cout << "L2W-" << endl;
		print(L2W,1,n2*n1);

		L2B = init_weights_bias( l2_rows,1); // initializing weights for layer 3
		/*
		 * Layer 3- 15 * 10 weights initialization vector
		 * 			1 * 10 Bias vector
		 */
		cout << "L2B-" << endl;
		print(L2B,1,n2);		
		L3W = init_weights(l3_rows, l3_cols);
		
		cout << "L3W-" << endl;
		print(L3W,1,n2*n3);
		L3B = init_weights_bias(l3_rows,1);
		cout << "L3B-" << endl;
		print(L3B,1,n3);

		/*
		*	TRAINING MODEL
		*/
		
		if ( (fp = fopen(filename1, "r") ) == NULL){
				printf("Cannot open %s.\n", filename1);
			}
			fpos_t position;
			fgetpos (fp, &position);
			
				gettimeofday(&start_time, NULL);
		for (int z = 1;z <= epoch;++z){// FOR EACH EPOCH
			printf ("epoch:%d\n",z);
			
			if(feof(fp)){
				fsetpos (fp, &position);
			}
				while(!feof(fp)){
					i=0;
				
					if(fgets(buff, 12000 ,fp )!=NULL){

						token = strtok(buff,",");
						++batch_size;
							//if(batch_size<=10){
								
									while(token!=NULL){
										activation[i] = atof(token);
										 // can use atof to convert to half
										token = strtok(NULL,",");
										i++;
									}
									std::vector<half> label = getlabelVector(activation[0]);
							// feed forward
									for (int j=0; j<=A1.size();j++ ){
										
										A1[j]=activation[j+1];
										}//A1[j]=activation[j+1];
									z2=dot(L2W,A1,n2,n1,1)+L2B;
				//printf("Printing z2:");					
				//print(z2,1,30);
									A2 = sigmoid(z2);
									z3=dot(L3W,A2,n3,n2,1)+L3B;
				//printf("Printing z3");					
				//print(z3,1,10);
									A3 = sigmoid(z3);
													// back propagation
									//d3 = (A3-label)*sigmoid_d(z3);
									error_calc = error_calc +  (A3 - label) * (A3 - label);
									d3 = (A3-label) * sigmoid_d(z3);	
									L3nb = d3 + L3nb;
									L3nw = dot(d3,transpose(&A2[0],1,n2),n3,1,n2) + L3nw; // gradient discent
									d2 = dot(transpose(&L3W[0],n2,n3),d3,n2,n3,1)*sigmoid_d(z2);
									L2nb = d2 + L2nb;
									L2nw = dot(d2,transpose(&A1[0],1,n1),n2,1,n1) + L2nw ; //gradient discent									
									//printf("D2:\n");
//print(d2,30,1);
//			printf("D3:\n");
//			print(d3,10,1);
					
							//}
							/*else{
								//printf("weights updated")
								L3W = update_wandb(L3W,L3nw,n2*n3);
								L3B = update_wandb(L3B,L3nb,n3);
								L2W = update_wandb(L2W,L2nw,n1*n2);
								L2B = update_wandb(L2B,L2nb,n2);
								batch_size = 0;
								reinit(L3nw);
								reinit(L3nb);
								reinit(L2nw);
								reinit(L2nb);
								continue;
							}*/
					}
					if (batch_size==100){
								//printf("weights updated")
								L3W = update_wandb(L3W,L3nw,n2*n3);
								L3B = update_wandb(L3B,L3nb,n3);
								L2W = update_wandb(L2W,L2nw,n1*n2);
								L2B = update_wandb(L2B,L2nb,n2);
								batch_size = 0;
								reinit(L3nw);
								reinit(L3nb);
								reinit(L2nw);
								reinit(L2nb);
								continue;
							}
					//printf("reading line and learning\n");
				}
				//printf("activations\n");
				//print(A3,n3,1);
				
			
		//printf ("epoch:%d\n",z);
				//error_calc = (half) accumulate(error_calc.begin(), error_calc.end(), 0.0) / (error_calc.size() * 2 * tot_train_set);
				//printf("Printing D3:\n");
				//print(d3,n3,1);
				error_mat[z-1] =  (half) accumulate(error_calc.begin(), error_calc.end(), 0.0) / (error_calc.size() * 2 * tot_train_set);
				//printf("Error after epoch: %f\n",error_mat[z-1]);
				cout << "Error after epoch:" << error_mat[z-1] << endl;
				for (int h = 0 ; h < n3;++h){
					error_calc[h] = 0;
				}
				if (error_mat[z-1] < 0.001){
					break;
				}
		}
				gettimeofday(&middle_time, NULL);
		cout << "Layer 2 W-" << endl;		
		print(L2W,1,n2*n1);
		cout << "Layer 2 B-" << endl;
		print(L2B,1,n2);

		cout << "Layer 3 W-" << endl;
		print(L3W,1,n3*n2);
		cout << "Layer 3 B-" << endl;
		print(L3B,1,n3);
		cout << "Printing Error over Epochs:" << endl;
		for (int u = 0; u < error_mat.size();++u){
			cout << "Epoch " << u+1 << ":" << error_mat[u] << endl;	
			fprintf(write_out,"Epoch %d: %lf\n",u+1,error_mat[u]);
			}


double training_time = 1000 * (middle_time.tv_sec - start_time.tv_sec)
			+ ((middle_time.tv_usec - start_time.tv_usec) / 1000.0);
			cout << "Training time is:" << training_time << endl;
fprintf(write_out, "TRAINING TIME: %.3f\n\n",training_time/1000 );
			/*
			*	TESTING MODEL 
			*	every image from test file is fed forward through the network. The actual label and predicted label is recorded 
			*	and added to the evaluation matrix.
			*/

		if ( (fp = fopen(filename2, "r") ) == NULL)
				{
					printf("Cannot open %s.\n", filename2);
						  //  result = FAIL;
				}
				else{

					printf("File opened; ready to read.\n");
					for (int r = 0; r < evaluation_mat.size(); ++r) {
						evaluation_mat[r].resize(n3);
					}
					vector<half> pred_out;
					std::vector<half> label;
					int act_ind; // index of class label for actual output
					int pred_ind;// index of class label for predicted output 
					while(!feof(fp)){
						i=0;
						if(fgets(buff, 2000 ,fp )!=NULL){

							token = strtok(buff,",");
						//	printf("%s\n",token);

							//lab = atoi(token);

							while(token!=NULL){
								//printf("%s\n",token);
								activation[i]= atof(token); // can use atof to convert to half
								token = strtok(NULL, ",");
								i++;
							}
							//printf("value of i:%d\n",i);
						 }
					//cout << "THe value of label is :" << activation[0] << endl;
						label = getlabelVector(activation[0]);// 
							// feed forward
				
						for (int j=1; j<=A1.size();j++ ){
							A1[j]=activation[j+1];						
						}
						//A1[j]=activation[j+1];
						//printf("Printing A1:\n");						
						//print(A1,784,1);
						z2=dot(L2W,A1,n2,n1,1)+L2B;
						A2 = sigmoid(z2);
						//print(A2,15,1);

						z3=dot(L3W,A2,n3,n2,1)+L3B;
						A3 = sigmoid(z3);

	
						pred_out = mod_output(A3);
						
						act_ind = ind_identifier(label);
						//cout << "act_ind : " << act_ind << endl;
												
						pred_ind = ind_identifier(pred_out);
						//cout << "pred_ind: " << pred_ind << endl;						
						//print(A3,n3,1);	
						evaluation_mat[act_ind][pred_ind] += 1.0;

						
						//printf("\nprint results\n");
						
					}
						
				}

				gettimeofday(&end_time, NULL);

		double testing_time = 1000 * (end_time.tv_sec - middle_time.tv_sec)
			+ ((end_time.tv_usec - middle_time.tv_usec) / 1000.0);

			fprintf(write_out, "\nTESTING TIME: %.3lf\n\n",testing_time/1000 );
			cout << "Testing time is :" << testing_time << endl;

		cout << "Printing Evaluation Matrix:" << endl;
		print_vectors(evaluation_mat);
		fprintf(write_out,"\n\n");
		fprintf(write_out, "			EVALUATION MATRIX\n\n");
	for (int g = 0; g < evaluation_mat.size(); ++g) {
		for (auto y = evaluation_mat[g].begin(); y != evaluation_mat[g].end(); ++y) {
			fprintf(write_out,"%.2f\t", *y);
		}
		fprintf(write_out,"\n");
	}
	fprintf(write_out, "\n");


	vector<half> TP(10);
	vector<half> FP(10);
	vector<half> FN(10);
	vector<half> TN(10);

	/*
	 * True Positive (TP) = diagonal elements of the evaluation matrix
	 */
	for (int e = 0; e < TP.size(); ++e) {
		TP[e] = evaluation_mat[e][e];
	}

	/*
	 * False Negative (FN) = sum of the column of the evaluation matrix for each class minus the diagonal element
	 */
	for (int f = 0; f < FN.size(); ++f) {
		FN[f] = 0.0;
		for (int i = 0; i < n3; ++i) {
			FN[f] += evaluation_mat[i][f];
		}
		FN[f] -= evaluation_mat[f][f];
	}

	/*
	 * False Positive (FP) = sum of the row elements of evaluation matrix for each class minus the diagonal element
	 */
	for (int t = 0; t < FP.size(); ++t) {
		FP[t] = 0.0;
		for (int i = 0; i < n3; ++i) {
			FP[t] += evaluation_mat[t][i];
		}
		FP[t] -= evaluation_mat[t][t];
	}

	for (int r = 0; r < TN.size(); ++r) {
		TN[r] = 0.0;

		TN[r] = tot_test_set - (TP[r] + FN[r] + FP[r]);
	}

	cout << "True Positive :" << endl;
	print(TP,1,n3);
	cout << endl;
	fprintf(write_out,"			TRUE POSITIVE:\n");
	for (int t = 0 ; t < TP.size();++t){
		fprintf(write_out,"%.0f\t",TP[t]);
		
	}
	fprintf(write_out,"\n\n");
	cout << "False Positive :" << endl;
	print(FP,1,n3);
	cout << endl;
	
	fprintf(write_out,"			FALSE POSITIVE:\n");
	for (int t = 0 ; t < FP.size();++t){
		fprintf(write_out,"%.0f\t",FP[t]);
		
	}
	fprintf(write_out,"\n\n");

	cout << "False Negative:" << endl;
	print(FN,1,n3);
	cout << endl;

	fprintf(write_out,"			FALSE NEGATIVE:\n");
	for (int t = 0 ; t < FN.size();++t){
		fprintf(write_out,"%.0f\t",FN[t]);
		
	}
	fprintf(write_out,"\n\n");

	cout << "True Negative :" << endl;
	print(TN,1,n3);
	cout << endl;
	
	fprintf(write_out,"			TRUE NEGATIVE:\n");
	for (int t = 0 ; t < TN.size();++t){
		fprintf(write_out,"%.0f\t",TN[t]);
		
	}
	fprintf(write_out,"\n\n");

	vector<half> precision(n3);
	vector<half> recall(n3);

	vector<half> accuracy(n3);

	precision = vec_division(TP, (TP + FP));
	recall = vec_division(TP,(TP + FN));
	accuracy = vec_division((TP + TN) , ((TP + FN) + (FP + TN)));

	half avg_precision((half) accumulate(precision.begin(), precision.end(), 0.0) / precision.size());
	cout << "Printing Precision: " << avg_precision << endl;
	print(precision,1,n3);
	cout << endl;

	
	fprintf(write_out,"			PRECISION: %.3f\n",avg_precision);
	for (int t = 0 ; t < precision.size();++t){
		fprintf(write_out,"%.3f\t",precision[t]);
		
	}
	fprintf(write_out,"\n\n");

	half avg_recall((half) accumulate(recall.begin(), recall.end(), 0.0)
			/ recall.size());
	cout << "Printing Recall: " << avg_recall << endl;
	print(recall,1,n3);
	cout << endl;
	fprintf(write_out,"			RECALL: %.3f\n",avg_recall);
	for (int t = 0 ; t < recall.size();++t){
		fprintf(write_out,"%.3f\t",recall[t]);
		
	}
	fprintf(write_out,"\n\n");


	half avg_accuracy((half) accumulate(accuracy.begin(), accuracy.end(), 0.0)
			/ accuracy.size());
	cout << "Printing Accuracy: " << avg_accuracy << endl;
	print(accuracy,1,n3);
	cout << endl;
	fprintf(write_out,"			ACCURACY: %.3f\n",avg_accuracy);
	for (int t = 0 ; t < accuracy.size();++t){
		fprintf(write_out,"%.3f\t",accuracy[t]);
		
	}
	fprintf(write_out,"\n\n");
	


	fclose(write_out);
return 0;








}
