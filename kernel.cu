// --- Assumption: graph with positive edges

#include <stdio.h>
#include <string>
#include <map>
#include <iostream>
#include <fstream>

#include "Utilities.cuh"

#define BLOCKSIZE	256

using namespace std;

map<string, int> nameToNum;					// --- names of vertices
map<string, map<string, int>> weightMap;	// --- weights of edges 

/************************/
/* READ GRAPH FROM FILE */
/************************/
int *readGraphFromFile(int &N, char *fileName) {

	string vertex1, vertex2;                
	ifstream graphFile;
	int currentWeight;			
	N = 0;												// --- Init the number of found vertices
	graphFile.open(fileName);							// --- Open the graph file
   
	graphFile >> vertex1;								// --- Read first vertex
	while(vertex1 != "--END--") {						// --- Loop until end of file has not been found
		graphFile >> vertex2;							// --- Read second vertex
		graphFile >> currentWeight;						// --- Read weight between first and second vertex
		if (nameToNum.count(vertex1) == 0) {			// --- If vertex has not yet been added ...
			nameToNum[vertex1] = N;						//	   assign a progressive number to the vertex
			weightMap[vertex1][vertex1] = 0;			//	   assign a zero weight to the "self-edge"
			N++;										// --- Update the found number of vertices
		}
		if (nameToNum.count(vertex2) == 0) {
			nameToNum[vertex2] = N;
			weightMap[vertex2][vertex2] = 0;
			N++;
		}
		weightMap[vertex1][vertex2] = currentWeight;	// --- Update weight between vertices 1 and 2
		graphFile >> vertex1;
	}    
	graphFile.close();									// --- Close the graph file

	// --- Construct the array
	int *weightMatrix = (int*) malloc(N * N * sizeof(int));
	// --- Loop over all the vertex couples in the wights matrix
	for (int ii = 0; ii < N; ii++)						
		for (int jj = 0; jj < N; jj++)
			weightMatrix[ii * N + jj] = INT_MAX / 2;	// --- Init the weights matrix elements to infinity
	map<string, int>::iterator i, j;
	// --- Loop over all the vertex couples in the map
	//     (*i).first and (*j).first are the weight entries of the map, while (*i).second and (*j).second are their corresponding indices
	for (i = nameToNum.begin(); i != nameToNum.end(); ++i)
		for (j = nameToNum.begin(); j != nameToNum.end(); ++j) {
			// --- If there is connection between vertices (*i).first and (*j).first, the update the weight matrix
			if (weightMap[(*i).first].count((*j).first) != 0) 
				weightMatrix[N * (*i).second + (*j).second] = weightMap[(*i).first][(*j).first];
		}
	return weightMatrix;
}

/************************************/
/* PRINT MINIMUM DISTANCES FUNCTION */
/************************************/
void printMinimumDistances(int N, int *a) {
  
	map<string, int>::iterator i, j;
  
	// --- Prints all the node labels at the first row
	for (i = nameToNum.begin(); i != nameToNum.end(); ++i) printf("\t%s", i->first.c_str());
  
	printf("\n");
	
	j = nameToNum.begin();
	// --- Loop over the rows
	for (i = nameToNum.begin(); i != nameToNum.end(); i++) {
      
		printf("%s\t", i->first.c_str());
      
		// --- Loop over the columns
		for (j = nameToNum.begin(); j != nameToNum.end(); j++) {

			int dd =  a[i->second * N + j->second];
			if (dd != INT_MAX / 2) printf("%d\t",dd);
			else printf("--\t");
		}
      
		printf("\n");
	}
}

void printPathRecursive(int row, int col, int *minimumDistances, int *path, int N) {
    map<string, int>::iterator i = nameToNum.begin();
    map<string, int>::iterator j = nameToNum.begin();
	if (row == col) {advance(i, row); printf("%s\t", i -> first.c_str()); }
    else {
		if (path[row * N + col] == INT_MAX / 2) printf("No path exists\t\n");
		else {
			printPathRecursive(row, path[row * N + col], minimumDistances, path, N);
			advance(j, col);
			printf("%s\t", j -> first.c_str());
		}
	}
}

void printPath(int N, int *minimumDistances, int *path) {

	map<string, int>::iterator i;
	map<string, int>::iterator j;
  
	// --- Loop over the rows
	i = nameToNum.begin();
	for (int p = 0; p < N; p++) {
		
		// --- Loop over the columns
		j = nameToNum.begin();
		for (int q = 0; q < N; q++) {
			printf("From %s to %s\t", i -> first.c_str(), j -> first.c_str());
			printPathRecursive(p, q, minimumDistances, path, N);
			printf("\n");
			j++;
		}
		
		i++;
	}
}

/**********************/
/* FLOYD-WARSHALL CPU */
/**********************/
void h_FloydWarshall(int *h_graphMinimumDistances, int *h_graphPath, const int N) {
	for (int k = 0; k < N; k++)
		for (int row = 0; row < N; row++)
			for (int col = 0; col < N; col++) {
				if (h_graphMinimumDistances[row * N + col] > (h_graphMinimumDistances[row * N + k] + h_graphMinimumDistances[k * N + col])) {
					h_graphMinimumDistances[row * N + col] = (h_graphMinimumDistances[row * N + k] + h_graphMinimumDistances[k * N + col]);
					h_graphPath[row * N + col] = h_graphPath[k * N + col];
				}
			}
}

/*************************/
/* FLOYD-WARSHALL KERNEL */
/*************************/
__global__ void d_FloydWarshall(int k, int *d_graphMinimumDistances, int *d_graphPath, int N) {
  
	int col = blockIdx.x * blockDim.x + threadIdx.x;	// --- Each thread along x is assigned to a matrix column
	int row = blockIdx.y;								// --- Each block along y is assigned to a matrix row
	
	if (col >= N) return;

	int arrayIndex = N * row + col;				

	// --- All the blocks load the entire k-th column into shared memory
	__shared__ int d_graphMinimumDistances_row_k;	       
	if(threadIdx.x == 0) d_graphMinimumDistances_row_k = d_graphMinimumDistances[N * row + k];
	__syncthreads();

	if (d_graphMinimumDistances_row_k == INT_MAX / 2)	// --- If element (row, k) = infinity, no update is needed
    return;

	int d_graphMinimumDistances_k_col = d_graphMinimumDistances[k * N + col]; 
	if(d_graphMinimumDistances_k_col == INT_MAX / 2)	// --- If element (k, col) = infinity, no update is needed
    return;

	int candidateBetterDistance = d_graphMinimumDistances_row_k + d_graphMinimumDistances_k_col;
	if (candidateBetterDistance < d_graphMinimumDistances[arrayIndex]) {
		d_graphMinimumDistances[arrayIndex] = candidateBetterDistance;
		d_graphPath[arrayIndex] = d_graphPath[k * N + col];
	}
}

/********/
/* MAIN */
/********/
int main() {
  
	int N = 0;					// --- Number of vertices

	// --- Read graph array from file
	int *h_graphArray = readGraphFromFile(N, "graph.txt");		
	printf("\n******************\n");
	printf("* Original graph *\n");
	printf("******************\n");
	printMinimumDistances(N, h_graphArray);

	// --- Floyd-Warshall on CPU
	int *h_graphMinimumDistances = (int *) malloc(N * N * sizeof(int));
	int *h_graphPath			 = (int *) malloc(N * N * sizeof(int));
	memcpy(h_graphMinimumDistances, h_graphArray, N * N * sizeof(int));
	for (int k = 0; k < N; k++) 
		for (int l = 0; l < N; l++) 
			if (h_graphArray[k * N + l] == INT_MAX / 2) h_graphPath[k * N + l] = INT_MAX / 2;
			else h_graphPath[k * N + l] = k;

	h_FloydWarshall(h_graphMinimumDistances, h_graphPath, N);
	printf("\n*************************\n");
	printf("* CPU result: distances *\n");
	printf("*************************\n");
	printMinimumDistances(N, h_graphMinimumDistances);
	printf("\n********************\n");
	printf("* CPU result: path *\n");
	printf("********************\n");
	printMinimumDistances(N, h_graphPath);
	printf("\n********************\n");
	printf("* CPU result: path *\n");
	printf("********************\n");
	printPath(N, h_graphMinimumDistances, h_graphPath);


	// --- Graph array device allocation and host-device memory transfer
	int *d_graphMinimumDistances;	gpuErrchk(cudaMalloc(&d_graphMinimumDistances, N * N * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_graphMinimumDistances, h_graphArray, N * N * sizeof(int), cudaMemcpyHostToDevice));
	int *d_graphPath;				gpuErrchk(cudaMalloc(&d_graphPath, N * N * sizeof(int)));
	for (int k = 0; k < N; k++) 
		for (int l = 0; l < N; l++) 
			if (h_graphArray[k * N + l] == INT_MAX / 2) h_graphPath[k * N + l] = INT_MAX / 2;
			else h_graphPath[k * N + l] = k;
	gpuErrchk(cudaMemcpy(d_graphPath, h_graphPath, N * N * sizeof(int), cudaMemcpyHostToDevice));

	// --- Iterations
	for (int k = 0; k < N; k++) {
		d_FloydWarshall <<<dim3(iDivUp(N, BLOCKSIZE), N), BLOCKSIZE>>>(k, d_graphMinimumDistances, d_graphPath, N);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}

	// --- Copy results back to the host
	gpuErrchk(cudaMemcpy(h_graphMinimumDistances, d_graphMinimumDistances, N * N * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_graphPath, d_graphPath, N * N * sizeof(int), cudaMemcpyDeviceToHost));
	printf("\n**************\n");
	printf("* GPU result *\n");
	printf("**************\n");
	printMinimumDistances(N, h_graphMinimumDistances);
	printf("\n********************\n");
	printf("* GPU result: path *\n");
	printf("********************\n");
	printPath(N, h_graphMinimumDistances, h_graphPath);
	
}



