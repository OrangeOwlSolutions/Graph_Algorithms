//
//      Implementation of Dijkstra's Single-Source Shortest Path (SSSP) algorithm on the GPU.
//      The basis of this implementation is the paper:
//
//          "Accelerating large graph algorithms on the GPU using CUDA" by
//          Parwan Harish and P.J. Narayanan
//

#include <sstream>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <float.h>

#define NUM_ASYNCHRONOUS_ITERATIONS 20  // Number of async loop iterations before attempting to read results back

#define BLOCK_SIZE 16

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/***********************/
/* GRAPHDATA STRUCTURE */
/***********************/
typedef struct
{
	// --- Contains a pointer to the edge list for each vertex
	int *vertexArray;

	// --- Overall number of vertices
	int numVertices;

	// (E) This contains pointers to the vertices that each edge is attached to
	int *edgeArray;

	// Edge count
	int edgeCount;

	// (W) Weight array
	float *weightArray;

} GraphData;

/**********************************/
/* GENERATE RANDOM GRAPH FUNCTION */
/**********************************/
void generateRandomGraph(GraphData *graph, int numVertices, int neighborsPerVertex)
{
	graph -> numVertices	= numVertices;
	graph -> vertexArray	= (int *)	malloc(graph -> numVertices * sizeof(int));
	graph -> edgeCount		= numVertices * neighborsPerVertex;
	graph -> edgeArray		= (int *)	malloc(graph -> edgeCount   * sizeof(int));
	graph -> weightArray	= (float *)	malloc(graph -> edgeCount   * sizeof(float));

	for (int i = 0; i < graph->numVertices; i++) graph -> vertexArray[i] = i * neighborsPerVertex;

	for (int i = 0; i < graph->edgeCount; i++)
	{
		graph->edgeArray[i] = (rand() % graph->numVertices);
		graph->weightArray[i] = (float)(rand() % 1000) / 1000.0f;
		printf("edgeArray[%i] = %i\n", i, graph->edgeArray[i]);
		printf("weightArray[%i] = %f\n", i, graph->weightArray[i]);
	}
}

/***************************/
/* MASKARRAYEMPTY FUNCTION */
/***************************/
//
// --- Check whether the mask array is empty. This tells the algorithm whether it needs to continue running or not.
//
bool maskArrayEmpty(int *maskArray, int count)
{
	for (int i = 0; i < count; i++)
	{
		if (maskArray[i] == 1)
		{
			return false;
		}
	}

	return true;
}

/*************************/
/* ARRAY INITIALIZATIONS */
/*************************/
__global__ void initializeArrays(int* __restrict__ maskArray, float* __restrict__ costArray, float* __restrict__ updatingCostArray,
	const int sourceVertex, const int numVertices)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < numVertices) {

		//printf("sourceVertex = %i; tid = %i \n",sourceVertex,tid);
		if (sourceVertex == tid)
		{
			maskArray[tid] = 1;
			//printf("Initializing maskArray[%i] = %i \n",tid,maskArray[tid]);
			costArray[tid] = 0.0;
			updatingCostArray[tid] = 0.0;
		}
		else
		{
			maskArray[tid] = 0;
			costArray[tid] = FLT_MAX;
			updatingCostArray[tid] = FLT_MAX;
		}
	}
}

/*************/
/* KERNEL #1 */
/*************/
//__global__  void Kernel1(const int* __restrict__ vertexArray, const int* __restrict__ edgeArray, const float* __restrict__ weightArray,
//                        int* __restrict__ maskArray, float* __restrict__ costArray, float* __restrict__ updatingCostArray,
//                        const int numVertices, const int edgeCount)
//{
//    int tid = blockIdx.x*blockDim.x+threadIdx.x;
//
//	if (tid < numVertices) {
//	    if ( maskArray[tid] != 0 )
//		{
//			maskArray[tid] = 0;
//
//			int edgeStart = vertexArray[tid];
//			int edgeEnd;
//			if (tid + 1 < (numVertices))
//			{
//				edgeEnd = vertexArray[tid + 1];
//			}
//			else
//			{
//				edgeEnd = edgeCount;
//			}
//
//			for(int edge = edgeStart; edge < edgeEnd; edge++)
//			{
//				int nid = edgeArray[edge];
//
//				// One note here: whereas the paper specified weightArray[nid], I
//				//  found that the correct thing to do was weightArray[edge].  I think
//				//  this was a typo in the paper.  Either that, or I misunderstood
//				//  the data structure.
//				if (updatingCostArray[nid] > (costArray[tid] + weightArray[edge]))
//				{
//					updatingCostArray[nid] = (costArray[tid] + weightArray[edge]);
//				}
//			}
//		}
//	}
//}

__global__  void Kernel1(const int* __restrict__ vertexArray, const int* __restrict__ edgeArray, const float* __restrict__ weightArray,
	int* __restrict__ maskArray, float* __restrict__ costArray, float* __restrict__ updatingCostArray,
	const int numVertices, const int edgeCount)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < numVertices) {
		if (maskArray[tid] != 0)
		{
			maskArray[tid] = 0;

			int edgeStart = vertexArray[tid];
			int edgeEnd;
			if (tid + 1 < (numVertices))
			{
				edgeEnd = vertexArray[tid + 1];
			}
			else
			{
				edgeEnd = edgeCount;
			}

			for (int edge = edgeStart; edge < edgeEnd; edge++)
			{
				int nid = edgeArray[edge];

				// One note here: whereas the paper specified weightArray[nid], I
				//  found that the correct thing to do was weightArray[edge].  I think
				//  this was a typo in the paper.  Either that, or I misunderstood
				//  the data structure.
				//if (updatingCostArray[nid] > (costArray[tid] + weightArray[nid]))
				//{
				//	updatingCostArray[nid] = (costArray[tid] + weightArray[nid]);
				//}
				if (updatingCostArray[nid] > (costArray[tid] + weightArray[edge]))
				{
					updatingCostArray[nid] = (costArray[tid] + weightArray[edge]);
				}
			}
		}
	}
}

/*************/
/* KERNEL #2 */
/*************/
__global__  void Kernel2(const int* __restrict__ vertexArray, const int* __restrict__ edgeArray, const float* __restrict__ weightArray,
	int* __restrict__ maskArray, float* __restrict__ costArray, float* __restrict__ updatingCostArray,
	const int numVertices)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < numVertices) {
		//printf("costArray[%i] = %f; updatingCostArray[%i] = %f \n",tid,costArray[tid],tid,updatingCostArray[tid]);
		if (costArray[tid] > updatingCostArray[tid])
		{
			costArray[tid] = updatingCostArray[tid];
			maskArray[tid] = 1;
		}

		updatingCostArray[tid] = costArray[tid];
	}
}

/************************/
/* RUNDIJKSTRA FUNCTION */
/************************/

// Run Dijkstra's shortest path on the GraphData provided to this function.  This function will compute the shortest path distance from
// sourceVertices[n] -> endVertices[n] and store the cost in outResultCosts[n].  The number of results it will compute is given by numResults.
//
// This function will run the algorithm on a single GPU.
//
// \param graph Structure containing the vertex, edge, and weight arra
//              for the input graph
// \param startVertices Indices into the vertex array from which to
//                      start the search
// \param outResultsCosts A pre-allocated array where the results for
//                        each shortest path search will be written
// \param numResults Should be the size of all three passed inarrays
//
void runDijkstra(GraphData* graph, int *sourceVertices, float *outResultCosts, int numResults)
{
	// --- Create vertex array Va, edge array Ea and weight array Wa from G(V,E,W)
	int		*vertexArrayDevice;			cudaMalloc((void**)&vertexArrayDevice, sizeof(int) * graph->numVertices);
	int		*edgeArrayDevice;			cudaMalloc((void**)&edgeArrayDevice, sizeof(int) * graph->edgeCount);
	float	*weightArrayDevice;			cudaMalloc((void**)&weightArrayDevice, sizeof(float) * graph->edgeCount);

	cudaMemcpy(vertexArrayDevice, graph->vertexArray, sizeof(int) * graph->numVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(edgeArrayDevice, graph->edgeArray, sizeof(int) * graph->edgeCount, cudaMemcpyHostToDevice);
	cudaMemcpy(weightArrayDevice, graph->weightArray, sizeof(float) * graph->edgeCount, cudaMemcpyHostToDevice);

	// --- Create mask array Ma, cost array Ca and updating cost array Ua of size V
	int		*maskArrayDevice;			cudaMalloc((void**)&maskArrayDevice, sizeof(int) * graph->numVertices);
	float	*costArrayDevice;			cudaMalloc((void**)&costArrayDevice, sizeof(float) * graph->numVertices);
	float	*updatingCostArrayDevice;	cudaMalloc((void**)&updatingCostArrayDevice, sizeof(float) * graph->numVertices);

	int *maskArrayHost = (int*)malloc(sizeof(int) * graph->numVertices);

	for (int i = 0; i < numResults; i++)
	{
		// --- Initialize mask Ma to false, cost array Ca and Updating cost array Ua to \u221e
		initializeArrays << <iDivUp(graph->numVertices, BLOCK_SIZE), BLOCK_SIZE >> >(maskArrayDevice, costArrayDevice, updatingCostArrayDevice, sourceVertices[i], graph->numVertices);

		// --- Read mask array from device -> host
		cudaMemcpy(maskArrayHost, maskArrayDevice, sizeof(int) * graph->numVertices, cudaMemcpyDeviceToHost);

		// for (int i=0; i<graph->numVertices; i++) { printf("maskarray[%i] = %i \n", i, maskArrayHost[i]); }

		while (!maskArrayEmpty(maskArrayHost, graph->numVertices))
		{
			// In order to improve performance, we run some number of iterations
			// without reading the results.  This might result in running more iterations
			// than necessary at times, but it will in most cases be faster because
			// we are doing less stalling of the GPU waiting for results.
			for (int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++)
			{
				// execute the kernel
				Kernel1 << <iDivUp(graph->numVertices, BLOCK_SIZE), BLOCK_SIZE >> >(vertexArrayDevice, edgeArrayDevice, weightArrayDevice, maskArrayDevice, costArrayDevice,
					updatingCostArrayDevice, graph->numVertices, graph->edgeCount);

				Kernel2 << <iDivUp(graph->numVertices, BLOCK_SIZE), BLOCK_SIZE >> >(vertexArrayDevice, edgeArrayDevice, weightArrayDevice, maskArrayDevice, costArrayDevice, updatingCostArrayDevice,
					graph->numVertices);

			}

			cudaMemcpy(maskArrayHost, maskArrayDevice, sizeof(int) * graph->numVertices, cudaMemcpyDeviceToHost);
			for (int i = 0; i<graph->numVertices; i++) printf("%f\n", maskArrayHost[i]);
			printf("\n\n");

		}

		// Copy the result back
		cudaMemcpy(&outResultCosts[i * graph->numVertices], costArrayDevice, sizeof(float) * graph->numVertices, cudaMemcpyDeviceToHost);
		for (int j = 0; j<graph->numVertices; j++) printf("%f\n", outResultCosts[i*graph->numVertices + j]);
	}

	free(maskArrayHost);

	cudaFree(vertexArrayDevice);
	cudaFree(edgeArrayDevice);
	cudaFree(weightArrayDevice);
	cudaFree(maskArrayDevice);
	cudaFree(costArrayDevice);
	cudaFree(updatingCostArrayDevice);

	std::cout << "Computed '" << numResults << "' results" << std::endl;
}

/****************/
/* MAIN PROGRAM */
/****************/
int main()
{
	// --- Number of source locations. A source location is a starting point for the path. The algorithm calculates optimal paths from multiple source
	//     locations
	int numSources = 1;
	// --- Number of graph nodes
	int generateVerts = 5;
	// --- Number of edges per graph node
	int generateEdgesPerVert = 2;

	// --- Allocate memory for arrays
	GraphData graph;
	generateRandomGraph(&graph, generateVerts, generateEdgesPerVert);

	printf("Vertex Count: %d\n", graph.numVertices);
	printf("Edge Count: %d\n", graph.edgeCount);

	std::vector<int> sourceVertices;

	for (int source = 0; source < numSources; source++)
	{
		sourceVertices.push_back(source % graph.numVertices);
	}

	int *sourceVertArray = (int*)malloc(sizeof(int) * sourceVertices.size());
	std::copy(sourceVertices.begin(), sourceVertices.end(), sourceVertArray);

	printf("Num source %d = ; Source %d = \n", sourceVertices.size(), sourceVertArray[0]);

	// --- Allocate space for the results
	float *results = (float*)malloc(sizeof(float) * sourceVertices.size() * graph.numVertices);

	runDijkstra(&graph, sourceVertArray, results, sourceVertices.size());

	for (int i = 0; i<graph.numVertices; i++) printf("%f\n", results[i]);

	free(sourceVertArray);
	free(results);

	getchar();

	return 0;

}

//// A C / C++ program for dijkstraCPU's single source shortest path algorithm.
//// The program is for adjacency matrix representation of the graph
//
//#include <stdio.h>
//#include <limits.h>
//
//// Number of vertices in the graph
//#define V 9
//
//// A utility function to find the vertex with minimum distance value, from
//// the set of vertices not yet included in shortest path tree
//int minDistance(int dist[], bool sptSet[])
//{
//// Initialize min value
//int min = INT_MAX, min_index;
//
//for (int v = 0; v < V; v++)
//	if (sptSet[v] == false && dist[v] <= min)
//		min = dist[v], min_index = v;
//
//return min_index;
//}
//
///************************************/
///* PRINT MINIMUM DISTANCES FUNCTION */
///************************************/
//int printMinDistances(int *shortestDistances, int N)
//{
//	printf("Shortest distance from source vertex\n");
//	for (int i = 0; i < N; i++) printf("%d \t\t %d\n", i, shortestDistances[i]);
//}
//
///************************/
///* dijkstraCPU FUNCTION */
///************************/
//void dijkstraCPU(int *graph, int *h_shortestDistances, int src, const int N)
//{
//	// --- h_finalizedVertices[i] is true if vertex i is included in the shortest path tree
//	//     or the shortest distance from the source node to i is finalized
//	bool *h_finalizedVertices = (bool *)malloc(N * sizeof(bool));
//
//	// --- Initialize h_shortestDistancesances as infinite and h_shortestDistances as false
//	for (int i = 0; i < N; i++)
//		h_shortestDistances[i] = INT_MAX, h_finalizedVertices[i] = false;
//
//	// --- h_shortestDistancesance of the source vertex from itself is always 0
//	h_shortestDistances[src] = 0;
//
//	// --- Dijkstra iterations
//	for (int iterCount = 0; iterCount < N - 1; iterCount++)
//	{
//		// --- Selecting the minimum distance vertex from the set of vertices not yet
//		//     processed. currentVertex is always eqcurrentVertexal to src in the first iteration.
//		int currentVertex = minDistance(h_shortestDistances, h_finalizedVertices);
//
//		// --- Mark the current vertex as processed
//		h_finalizedVertices[currentVertex] = true;
//
//		// --- Relaxation loop
//		for (int v = 0; v < N; v++)
//
//			// --- Update dist[v] only if it is not in h_finalizedVertices, there is an edge
//			//     from u to v, and the cost of the path from the source vertex to v through
//			//     currentVertex is smaller than the current value of h_shortestDistances[v]
//			if (!h_finalizedVertices[v] &&
//				 graph[currentVertex * N + v] &&
//				 h_shortestDistances[currentVertex] != INT_MAX &&
//				 h_shortestDistances[currentVertex] + graph[currentVertex * N + v] < h_shortestDistances[v])
//
//				h_shortestDistances[v] = h_shortestDistances[currentVertex] + graph[currentVertex * N + v];
//	}
//
//}
//
///********/
///* MAIN */
///********/
//int main()
//{
//	/* Let us create the example graph discussed above */
//	int graph[V][V] = {{0, 4, 0, 0, 0, 0, 0, 8, 0},
//					{4, 0, 8, 0, 0, 0, 0, 11, 0},
//					{0, 8, 0, 7, 0, 4, 0, 0, 2},
//					{0, 0, 7, 0, 9, 14, 0, 0, 0},
//					{0, 0, 0, 9, 0, 10, 0, 0, 0},
//					{0, 0, 4, 0, 10, 0, 2, 0, 0},
//					{0, 0, 0, 14, 0, 2, 0, 1, 6},
//					{8, 11, 0, 0, 0, 0, 1, 0, 7},
//					{0, 0, 2, 0, 0, 0, 6, 7, 0}
//					};
//
//	// --- At the end of the procedure, h_shortestDistances[i] will hold the shortest
//	//     distance from the source vertex to i
//	int *h_shortestDistances = (int *)malloc(V * sizeof(int));
//
//	dijkstraCPU((int *)graph, h_shortestDistances, 0, V);
//
//	printMinDistances(h_shortestDistances, V);
//
//	return 0;
//}
