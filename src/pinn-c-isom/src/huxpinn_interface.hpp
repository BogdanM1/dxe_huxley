#include "tf_utils.hpp"
#include "scope_guard.hpp"
#include <iostream>
#include <vector>

void huxpinn_init(int* n_qpoints, double*Kxb, double*xstart, double*xend, int *xdiv, double*L0, double *A,
                  char* model_path="/home/bogdan/dxe_huxley/models/model.pb");
void huxpinn_set_values(int * qindex, double *time, double* activation, double* stretch, double *stretch_prev);
void huxpinn_converged();
void huxpinn_predict();
void huxpinn_get_values(int *qindex, double * stress, double * dstress);
void huxpinn_destroy();
