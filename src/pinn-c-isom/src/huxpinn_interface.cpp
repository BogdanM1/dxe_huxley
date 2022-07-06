#include "huxpinn_interface.hpp"
#include <fstream>

using namespace std;
using std::vector;

#define MAX_LINE_LENGTH 255
 
int huxpinn_nqpoints; 
TF_Graph* huxpinn_graph;
TF_Session* huxpinn_session;
vector<TF_Output> huxpinn_input_ops;
vector<TF_Output> huxpinn_out_ops;
vector<int64_t> huxpinn_input_dims;
vector<float> huxpinn_input_values;
vector<float> huxpinn_output_values;

double huxpinn_Kxb, huxpinn_xstart, huxpinn_xend, huxpinn_L0, huxpinn_A, huxpinn_xstep;
int huxpinn_xdiv;

int huxpinn_nfeatures = 2;

void huxpinn_init(int* n_qpoints, double*Kxb, double*xstart, double*xend, int *xdiv, 
				  double*L0, double *A, char* model_path)
{
	huxpinn_nqpoints = *n_qpoints;
	huxpinn_Kxb = *Kxb;
	huxpinn_xstart = *xstart;
	huxpinn_xend = *xend;
	huxpinn_xdiv = *xdiv;
	huxpinn_xstep = (*xend-*xstart)/(*xdiv);
	huxpinn_A = *A;
	huxpinn_L0 = *L0;
	huxpinn_xdiv++;
 
	huxpinn_input_dims = {huxpinn_nqpoints*huxpinn_xdiv, huxpinn_nfeatures};
	huxpinn_input_values.resize(huxpinn_nqpoints*huxpinn_xdiv*huxpinn_nfeatures);
	
	for(int iqp = 0; iqp < huxpinn_nqpoints; iqp++)
	{
		int qindex_start = iqp*(huxpinn_nfeatures*huxpinn_xdiv);
		for(int ix = 0; ix < huxpinn_xdiv; ix++)
		{
			huxpinn_input_values[qindex_start + ix*huxpinn_nfeatures] = huxpinn_xstart + ix*huxpinn_xstep;
			huxpinn_input_values[qindex_start + ix*huxpinn_nfeatures + 1] = 0.0;
		}
			
	}
  
	// load huxpinn_graph
	huxpinn_graph = tf_utils::LoadGraph(model_path);
	if(huxpinn_graph == nullptr) 
	{
		std::cout << "Can't load huxpinn_graph" << std::endl;
		return;
	}
	
	// input/output operations
	huxpinn_input_ops = {{TF_GraphOperationByName(huxpinn_graph, "Placeholder_2"), 0}};
	huxpinn_out_ops = {{TF_GraphOperationByName(huxpinn_graph, "dense_3/BiasAdd"), 0}}; 
	
	// create huxpinn_session 
	huxpinn_session = tf_utils::CreateSession(huxpinn_graph);
	if(huxpinn_session == nullptr) 
	{
		std::cout << "Can't create huxpinn_session" << std::endl;
		return;
	}	

}

void huxpinn_set_values(int * qindex, double *time, double* activation, double* stretch, double *stretch_prev)
{
	int qindex_start = (* qindex)*(huxpinn_nfeatures*huxpinn_xdiv);
	for(int ix = 0; ix < huxpinn_xdiv; ix++)
	{
		huxpinn_input_values[qindex_start +  ix*huxpinn_nfeatures + 1] = *time;
	}	
}

void huxpinn_converged()
{

	
}

void huxpinn_predict()
{
	vector<TF_Tensor*> input_tensors = {tf_utils::CreateTensor(TF_FLOAT, huxpinn_input_dims, huxpinn_input_values)};
 	vector<TF_Tensor*> output_tensors = {nullptr};	
	/**************** debug *******************
	cout << "input values\n";
	for(int i=0; i< (int)huxpinn_input_values.size(); i+=2)
	{  
		printf("%.8lf %.8lf\n",huxpinn_input_values[i], huxpinn_input_values[i+1]);
	}
	cout << "\n";
	/*******************************************/

	tf_utils::RunSession(huxpinn_session, huxpinn_input_ops, input_tensors, huxpinn_out_ops, output_tensors);
	huxpinn_output_values = tf_utils::GetTensorData<float>(output_tensors[0]);

	/*********** debug *******************
	cout << "prediction\n";
	for(int i=0; i< (int)huxpinn_output_values.size(); i+=1)
		printf("%.19lf\n",huxpinn_output_values[i]);
	cout << "\n";
	/***************************************/	
	tf_utils::DeleteTensors(input_tensors);
	tf_utils::DeleteTensors(output_tensors);	
 
}

void huxpinn_get_values(int *qindex, double * stress, double * dstress)
{ 
	int qindex_start = (* qindex)*(huxpinn_nfeatures*huxpinn_xdiv);
	*stress = .0;
	*dstress = .0;
	double stretch = 1.0;
	double n,x, n_prev, x_prev;
	for(int ix=1; ix<huxpinn_xdiv; ix++)
	{
		n_prev = huxpinn_output_values[qindex_start + ix - 1];
		x_prev = huxpinn_input_values[qindex_start +  (ix-1)*huxpinn_nfeatures];
		
		n = huxpinn_output_values[qindex_start + ix];
		x = huxpinn_input_values[qindex_start +  ix*huxpinn_nfeatures];
		
		//printf("%lf %lf\n",x,n); // ukloni 
		
		(*stress) += (0.5*(n_prev*x_prev + n*x)*huxpinn_xstep);
		(*dstress) += (0.5*(n_prev + n)*huxpinn_xstep);
	}
	
	*stress = ((*stress) * huxpinn_Kxb)/ huxpinn_A;
	*dstress = ((*dstress)*huxpinn_Kxb * stretch* huxpinn_L0)/ huxpinn_A;
}

void huxpinn_destroy()
{
	tf_utils::DeleteSession(huxpinn_session);	
	tf_utils::DeleteGraph(huxpinn_graph); 
	huxpinn_input_dims.clear();
	huxpinn_input_values.clear();
	huxpinn_output_values.clear(); 
}


/*
// test */
int main()
{
	int nqp = 1, iqp = 0;
	double stretch = 1.0;
  double stretch_prev = 1.0;
  double activation = 1.0;
	double time;
	double stress, dstress;
 
  double sim_duration = .4;
	double dt = 0.001; 
  double stretch_start = 1.0, stretch_end=0.75;
  double stretch_delta = (stretch_end - stretch_start)/(sim_duration/dt);

	/** params */
	double Kxb = 0.58, A = 130.0, L0 = 1100.0, xstart = -2.08, xend = 20.28;
	int xdiv = 43; //55; 
	/***********/
	
	huxpinn_init(&nqp, &Kxb, &xstart, &xend, &xdiv, &L0, &A);              
	while (abs(time - sim_duration) > 1e-6)
	{
		time = time + dt;
		huxpinn_set_values(&iqp,&time,&activation,&stretch,&stretch_prev);
		huxpinn_predict();
		huxpinn_get_values(&iqp,&stress,&dstress);	
		printf("%lf,%lf,%lf,%lf\n",time,stretch,stress,dstress);
      
    stretch_prev = stretch;
    stretch +=stretch_delta;
	}
	huxpinn_destroy();
}
/**/
