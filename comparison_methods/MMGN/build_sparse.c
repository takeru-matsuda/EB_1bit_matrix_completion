#include "mex.h"
//#include "matrix.h"
#include "math.h"

void build_sparse(const double *U, const double *V, const int32_T *omega, int32_T *rowind, int32_T *colind, double *vals, mwSize m, mwSize n, mwSize r, mwSize nv) {
    
    mwIndex id1 = 0;
    mwIndex id2 = 0;
    for (mwSize i = 0; i < nv; i++)
    {
        id1 = (mwSize)ceil(omega[i]/m)+1;
        id2 = omega[i] - (id1-1)*m;
        if(id2 == 0){
          id2 = m;
          id1 = id1-1;
        }
        
        
       for (mwSize j = 0; j < 2*r; j++)
      {
        // get the row indices
          rowind[2*r*i + j] = i+1;
        // get the column indices
        // get the values 
          if (j<r){
            colind[2*r*i + j] = id2 + m*j;
            vals[2*r*i + j] = V[id1 + j*n - 1]; // (id1, j+1)-th entry of V, since j starts from 0
            
          }else{
            colind[2*r*i + j] = r*m + (id1-1)*r + (j-r) + 1;
            vals[2*r*i + j] = U[id2 + (j-r)*m - 1]; // (id2, j-r+1)-th entry of U
          }             
       } 

    }
 
     
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check input and output arguments
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("build_sparse:invalidInput", "Three inputs required.");
    }
    if (nlhs != 3) {
        mexErrMsgIdAndTxt("build_sparse:invalidOutput", "Three output required.");
    }

    // Get input matrices
    const double *U = mxGetPr(prhs[0]);
    const double *V = mxGetPr(prhs[1]);
    //const double *omega = mxGetPr(prhs[2]);
    const int32_T *omega = (int32_T*)mxGetData(prhs[2]);
    mwSize m = mxGetM(prhs[0]); // Number of rows
    mwSize r = mxGetN(prhs[0]); // Number of columns
    mwSize n = mxGetM(prhs[1]);
    mwSize nv = mxGetNumberOfElements(prhs[2]);


    // Create output matrix
    plhs[0] = mxCreateNumericMatrix(2*r*nv, 1, mxINT32_CLASS, mxREAL);
    int32_T *rowind = (int32_T*)mxGetData(plhs[0]);
    plhs[1] = mxCreateNumericMatrix(2*r*nv, 1, mxINT32_CLASS, mxREAL);
    int32_T *colind = (int32_T*)mxGetData(plhs[1]);
    plhs[2] = mxCreateDoubleMatrix(2*r*nv, 1, mxREAL);
    double *vals = mxGetPr(plhs[2]);
    
//     
//     mexPrintf("m is %d .\n", m);
//     mexPrintf("n is %d .\n", n);
//     mexPrintf("r is %d .\n", r);   
//     mexPrintf("nv is %d .\n", nv);
    
    // Call the matrix multiplication function
    // build_sparse(U, V, omega, m, n, r, nv, rowind, colind, vals, ID1, ID2);
    build_sparse(U, V, omega, rowind, colind, vals, m, n, r, nv);
}








