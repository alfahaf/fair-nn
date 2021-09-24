#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <set>
#include <map>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <chrono>
#include <cassert>
using namespace std;

#include "fastrng.h"


const int maxn = 10001;//maximum number of points in the data set
const int maxQ = 101;//maximum number of queries asked
const int maxL = 501;//maximum value of L considered
const int maxk = 21;//maximum value of k considered
const int RUNS = 10; //repeat each test 10 times
const int SEED = 1234; // fixed seed
auto timer = std::chrono::high_resolution_clock();
int n , k , L, Q;
int dim;  // dimension
double w;
string dataset_fn, queryset_fn; 
double R;//radius of near neighbor
double* pnt[maxn];//data set points
double* query[maxQ];//query points
double* hfv[maxL][maxk];//random direction vectors used for each of the (k*L) hash functions
double hfb[maxL][maxk];//random bias for shifting the 1-dimensional grid used for each of the (k*L) hash functions

vector<int> ranks(maxn); 
vector<int> point_rank(maxn);
map<int, vector<double>> times;

void Initialize(int argc, char** argv){
	cout << "initializing ..." << endl;
	memset(pnt,0,sizeof pnt);

	if (argc == 1) {
		dataset_fn = string("data/mnist_data.txt");
		queryset_fn = string("data/mnist_queries.txt");
		dim = 784;
		k = 15;
		L = 100;//sqrt(n);
		R = 255*5;
		w= 3750;
	} else if (argc == 8) {
		dataset_fn = string(argv[1]);
		queryset_fn = string(argv[2]);
		dim = std::atoi(argv[3]);
		k = std::atoi(argv[4]);
		L = std::atoi(argv[5]);
		R = std::atof(argv[6]);
		w = std::atof(argv[7]);
	} else {
		std::cerr << "Wrong number of arguments." << endl;
		std::cerr << "Usage: " << argv[0] << " <dataset> <queryset> <dim> <k> <L> <R> <w>" << endl;
		exit(1);
	}

	// fixed constants
	n = 10000;
	Q = 50;
	cout << "Running experiment on " << dataset_fn << endl;
	cout <<"Params: L is " << L << " k is " << k << " w is " << w << " R is " << R << " n is " << n << endl;
	srand(SEED);
	for (int i = 0; i < maxn; i++) 
		pnt[i] = new double[dim];
	for (int i = 0; i < maxQ; i++) 
		query[i] = new double[dim];
	for (int i = 0; i < maxL; i++) 
		for (int j = 0; j < maxk; j++) 
			hfv[i][j] = new double[dim];

	for (int i = 0; i < n; i++) {
		ranks[i] = i;
	}
	std::shuffle(ranks.begin(), ranks.end(), std::default_random_engine(SEED));

	for (int i = 0; i < n; i++) {
		point_rank[ranks[i]] = i;
	}
}

void ReadInput(){
	cout << "Reading input ..." << endl;
	ifstream fin(dataset_fn);//reading the data set points
	for (int i=0 ; i<n ; ++i)
		for (int j=0 ; j<dim ; ++j){
			fin >> pnt[i][j];
		}
	fin.close();
	fin.open(queryset_fn);//reading the query points
	for (int i=0 ; i<Q ; ++i)
		for (int j=0 ; j<dim ; ++j){
			fin >> query[i][j];
		}
	fin.close();
	return;
}

void ReadHashFunctions(){
	cout << "reading hash functions...." << endl;
	ifstream fin("randv.csv");//reading random directions
	for (int i=0 ; i<L ; ++i)
		for (int j=0 ; j<k ; ++j)
			for (int ii=0 ; ii<dim ; ++ii){
				fin >> hfv[i][j][ii];
				char c;
				if (ii< dim-1)
					fin >> c;
			}
	fin.close();
	fin.open("randb.csv");//reading random biases
	for (int i=0 ; i<L ; ++i)
		for (int j=0 ; j<k ; ++j){
			fin >> hfb[i][j];
			hfb[i][j] = hfb[i][j]*w;
		}
	fin.close();
	return;
}

void GenerateHashFunctions(){
	cout << "Generating hash functions...." << endl;
	std::default_random_engine generator;
	std::normal_distribution<double> norm_dist(0.0, 1.0);

	for (int i=0 ; i<L ; ++i)
		for (int j=0 ; j<k ; ++j) 
			for (int ii=0 ; ii<dim ; ++ii)
				hfv[i][j][ii] = norm_dist(generator);

	std::uniform_int_distribution<int> int_dist(0, w);
	for (int i=0 ; i<L ; ++i)
		for (int j=0 ; j<k ; ++j)
			hfb[i][j] = int_dist(generator);
	return;
}

struct HP{//this is a structure for a hashed point
	int val[maxk];
	HP(){
		memset(val,0,sizeof val);
	}
	bool operator < (const HP& hp)const{
		for (int i=0 ; i<k ; ++i){
			if (val[i]<hp.val[i])
				return true;
			if (val[i]>hp.val[i])
				return false;
		}
		return false;
	}
	bool operator == (const HP& hp)const{
		for (int i=0 ; i<k ; ++i)
			if (val[i]!=hp.val[i])
				return false;
		return true;
	}
};

HP hPoint[maxL][maxn], hQuery[maxL][maxQ];//keeping the hashed data set points and hashed query points

void ComputeHashes(){
	cout << "Computing hashes....." << endl;
	for (int l=0 ; l<L ; ++l)//computing hash values of the data set points
		for (int i=0 ; i<n ; ++i)
			for (int kk=0 ; kk<k ; ++kk){
				double dotp = hfb[l][kk];
				for (int j=0 ; j<dim ; ++j)
					dotp += pnt[i][j]*hfv[l][kk][j];
				hPoint[l][i].val[kk] = ((int)(floor(dotp/w+1e-8)));
			}
	for (int l=0 ; l<L ; ++l)//computing hash values of the query points
		for (int i=0 ; i<Q ; ++i)
			for (int kk=0 ; kk<k ; ++kk){
				double dotp = hfb[l][kk];
				for (int j=0 ; j<dim ; ++j)
					dotp += query[i][j]*hfv[l][kk][j];
				hQuery[l][i].val[kk] = ((int)(floor(dotp/w+1e-8)));
			}
	return ;
}

double ComputeDist(int iq , int ip){//computes the distance between a query point (with id iq) with a data set point (with id ip)
	double dist = 0.0;
	for (int j=0 ; j<dim ; ++j)
		dist += (query[iq][j]-pnt[ip][j])*(query[iq][j]-pnt[ip][j]);
	dist=sqrt(dist);
	return dist;
}

set <pair<HP,int> > buckets[maxL];

void BuildLSH(){
	cout << "building LSH" << endl;
	for (int l=0 ; l<L ; ++l)//building the locality sensitive hashing data structure.
		for (int i=0 ; i<n ; ++i)
			buckets[l].insert(pair<HP,int>(hPoint[l][i],i));

	//the rest of this function computes some statistics about the current LSH which is used for tuning the parameters.
/*	int avgcount=0;
	for (int l=0 ; l<L ; ++l){
		for (int i=0 ; i<n ; ++i){
			int cnt= 0;
			set<pair<HP,int>>::iterator ed = buckets[l].lower_bound(pair<HP,int>(hPoint[l][i],maxn));
			for (set<pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hPoint[l][i],-1)) ; it!=ed ; ++it)
				cnt++;
			avgcount+=cnt;
		}
	}
	avgcount /= n*L;
	cout << "average collision per point " << avgcount << endl;
	avgcount = 0;
	int bnum=0;
	for (int l=0 ; l<L ; ++l){
		set<pair<HP,int>>::iterator it = buckets[l].begin();
		while (it!=buckets[l].end()){
			set<pair<HP,int>>::iterator it2 = buckets[l].lower_bound(pair<HP,int>((*it).first , maxn));
			int cnt=0;
			while (it!=it2){
				it++;
				cnt++;
			}
			avgcount+=cnt;
			bnum++;
		}
	}
	cout << "total number of buckets is " << bnum <<" and average num of buckets is " << bnum/L << endl;
	avgcount /= bnum;
	cout << "average bucket size is " << avgcount << endl;
*/	return;
}


//this function is also used only for tuning the parameters of LSH
void ComputeRetrievalRate(){
	int avgN =0;
	int avgNotNear = 0;
	int avgO = 0;
	int avgFN=0;
	int close=0;
	for (int i=0 ; i<Q ; ++i){
		int neighborhood=0;
		for (int j=0 ; j<n ; ++j){
			bool fall=false;
			for (int l=0 ; l<L ; ++l)
				if (hQuery[l][i]==hPoint[l][j]){
					fall= true;
					break;
				}
			if (fall){
				neighborhood++;
				if (ComputeDist(i,j)>1.3*R)
					avgO++;
				if (ComputeDist(i,j)>R)
					avgNotNear++;
			}
			if (ComputeDist(i,j)<R){
				close++;
				if (!fall)
					avgFN++;
			}
		}
		avgN += neighborhood;
	}
	cout << "average neighborhood size is " << avgN/Q << endl;
	cout << "average Outliers is " << avgO/Q << endl;
	cout << "average not near is " << avgNotNear/Q << endl;
	cout << "average False Negative is " << (avgFN*100)/close << "\%" << endl;
	return;
}


int sizes[maxL];// each time we get a query, we compute the sizes of the buckets corresponding to the query.
int sumsize;//and we compute the sum of sizes[0] ...sizes[(-1] of the sizes in the sumsize variable to be able to generate a random value in that range

deann::FastRng* rng; 


//This function returns a random bucket according to its size and then a random point inside the bucket.
//This is implemented by generating a random number from 1 to sumsize and then finding the point corresponding to this number
pair<int, int> single_sample_weighted(int iq, vector<set<int>>& query_buckets){
	int r = rand()%sumsize;
	r++;
	int l=0;
	while (r>sizes[l]){
		r-=sizes[l];
		l++;
	}
	// set<pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],-1));
	auto it = query_buckets[l].lower_bound(-1);
	while (true){
		r--;
		if (r==0)
			return make_pair(*it, l);
		it++;
	}
	return make_pair(-1, -1);
}

//This function returns a random bucket uniformly at random and then a random point inside the bucket.
//This is implemented by generating a random number from 1 to L and then a random point inside the bucket
pair<int, int> single_sample_uniform(int iq, vector<set<int>>& query_buckets){
	int l=-1;//index of the bucket
	for (l=(rand()%L) ; sizes[l]==0 ; l=(rand()%L));//avoid sampling an empty bucket
	int r = rand()%sizes[l];//index of the point in the bucket
	// set<pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],-1));
	auto it = query_buckets[l].lower_bound(-1);
	while (r){
		r--;
		it++;
	}
	return make_pair(*it, l);
}

//This function approximates the degree of a data set point (with index ip) among the buckets containing the query (with index iq)
//This is implemented by iteratively probing a random bucket and checking if it contains the point. 
//If the we see the point for the first time at iteration i, then we report L/i as an approximation of the degree
int approximate_degree(int iq , int ip){
	int num=0;
	while (num<L){
		num++;
		//int l = rand()%L;
		int l = (*rng)(); 
		if (hQuery[l][iq]==hPoint[l][ip])
			break;
	}
	return L/num;
}

int exact_degree(int iq , int ip){
	int cnt=0;
	for (int l =0 ; l < L; l++) {
		if (hQuery[l][iq]==hPoint[l][ip])
			cnt++;
	}
	return cnt; 
}


map <int,int> Degree;//this variable is used to compute the exact degrees of the points in the bucket correnponding to the current query. This is used for comparison with the optimal algorithm. Degree[ip] shows the exact degree of the point with index ip among the buckets of the current query.

const int cases = 7;//This corresponds to the four algorithms we are implementing 
//0 is the Uniform/Uniform algorithm, 1 is the Weighted/Uniform algorithm, 2 is the optimal algorithm, 3 is our proposed algorithm

double avgRes[cases];//This variable keeps the average results for each algorithm.
int TotalTestNum=0;// This variable shows the number of times we run the algorithm for different queries.

int TotalNonNear=0; // This variable records the number of times a non-near point was found 
int TotalRejections=0; // This variable records the number of times a near point was rejected 
long long TotalCloseColliding=0;
int TotalMaxDegree=0;

long long nonNear[cases];
long long rejections[cases];


/* this algorithm receives a query and draws snum number of samples according to the 
 * algorithm with the code cd (ranging from 0 to 3 as denoted above) 
 */
void RandomSample(int cd , int iq, int snum){
	if (snum==0)
		return;

	map<int,int> mp;//This shows the distribution over the points in the neighborhood of the query.

// This is in implementation of collecting all points, around 3x times slower as opt in my tests.

 	if (cd == 4) {
		vector<vector<int>> query_buckets; 
		for (int l=0 ; l<L ; ++l){
			set<pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],-1));
			vector<int> points;
			while (it!= buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],maxn))){
				points.push_back(it->second);
				it++;
			}
			query_buckets.push_back(points);
		}
        // Implementation of the dependent-between-query approach based on random ranks
		// sort all buckets by rank
		for (auto& points: query_buckets) {
			std::sort(points.begin(), points.end(), 
				[](const int& a, const int& b) -> bool
				{
					return point_rank[a] > point_rank[b]; // sort in descending order for quick removal of elements
			});
		}

		while (snum) {
			int s = n + 1;
			int min_rank = n + 1;
			vector<int> min_locations;
			for (int l=0 ; l<L ; ++l){
				if (query_buckets[l].size() == 0) {
					continue;
				}
				int q = query_buckets[l].at(query_buckets[l].size() - 1);
				if (point_rank.at(q) < min_rank) {
					min_rank = point_rank.at(q);
					s = q; 
					min_locations.clear();
					min_locations.push_back(l);
				}
				else if (point_rank.at(q) == min_rank) {
					min_locations.push_back(l);
				}
			}

			if (ComputeDist(iq, s) > R) {
				TotalNonNear++;
				for (auto &l: min_locations) {
					query_buckets[l].pop_back();
				}
				continue;
			}

			snum--;

			if (mp.find(s)!=mp.end())//update the distribution
				mp[s] = mp[s]+1;
			else
				mp[s] = 1;

			assert(min_rank != n + 1);


			auto new_rank = min_rank +  rand() % (maxn - min_rank - 1);
			auto q = ranks.at(new_rank);
			ranks.at(min_rank) = q;
			point_rank.at(q) = min_rank;
			ranks.at(new_rank) = s;
			point_rank.at(s) = new_rank;


			for (int l = 0; l < L; l++) {
				std::sort(query_buckets[l].begin(), query_buckets[l].end(), 
					[](const int& a, const int& b) -> bool
					{
						return point_rank.at(a) > point_rank.at(b);
					});
			}
		}
	 } else if (cd == 5) {
		vector<set<int>> query_buckets;
		for (int l=0 ; l<L ; ++l){
			set<pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],-1));
			int count=0;
			set<int> points;
			while (it!= buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],maxn))){
				points.insert(it->second);
				it++;
				count++;
			}
			query_buckets.push_back(points);
			sizes[l] = count;
			if (l==0)
				sumsize = sizes[l];
			else
				sumsize += sizes[l];
		}
	 } else if (cd == 6) {
		vector<vector<int>> query_buckets; 
		for (int l=0 ; l<L ; ++l){
			set<pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],-1));
			vector<int> points;
			while (it!= buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],maxn))){
				points.push_back(it->second);
				it++;
			}
			query_buckets.push_back(points);
		}
		// Implementation of the dependent-between-query approach based on random ranks
		// sort all buckets by rank
		for (auto& points: query_buckets) {
			std::sort(points.begin(), points.end(), 
				[](const int& a, const int& b) -> bool
				{
					return point_rank[a] > point_rank[b]; // sort in descending order for quick removal of elements
			});
		}
	 } else if (cd == 7) {
 		while (snum) {
			set<int> elements;
			for (int l=0 ; l<L ; ++l){
				set<pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],-1));
				while (it!= buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],maxn))){
					auto elem = (*it).second;
					// cout << elem << " " << ComputeDist(elem, iq) << endl;
					//if (ComputeDist(iq, elem) <= R) {
						elements.insert(elem);
					//}
					it++;
				}
			}
			vector<int> elem_vec;
			for (auto& e: elements) {
				elem_vec.push_back(e);
			}
			
			std::random_shuffle(elem_vec.begin(), elem_vec.end());
			for (auto &s: elem_vec) {
				if (ComputeDist(iq, s) <= R) {
					snum--;
					if (mp.find(s)!=mp.end())//update the distribution
						mp[s] = mp[s]+1;
					else
						mp[s] = 1;
					break;
				}
			}
		 }
 	} else {
		//computing the parameters sizes and sumsize of the buckets so that we dont have to compute it each time we draw a sample
		vector<set<int>> query_buckets;
		for (int l=0 ; l<L ; ++l){
			set<pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],-1));
			int count=0;
			set<int> points;
			while (it!= buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],maxn))){
				points.insert(it->second);
				it++;
				count++;
			}
			query_buckets.push_back(points);
			sizes[l] = count;
			if (l==0)
				sumsize = sizes[l];
			else
				sumsize += sizes[l];
		}

		while (snum){//generating the samples
			int s=-1;
			int l=-1;
			if (cd==0) {
				auto res = single_sample_uniform(iq, query_buckets);
				s = res.first;
				l = res.second;
			}
			else  {
				auto res = single_sample_weighted(iq, query_buckets);
				s = res.first;
				l = res.second;
			}
			if (ComputeDist(iq,s)>R) {//reject the points outside neighborhood
				query_buckets[l].erase(s); // remove the point from the data structure
				sizes[l]--;
				sumsize--;
				TotalNonNear++;
				continue;
			}
			bool valid = false;//shows if we should keep the sample or reject it.
			if (cd==0 || cd==1)//the uniform/uniform and weighted/uniform algorithms will always report the sampled point
				valid = true;
			else if (cd==2){//the optimal algorithm rejects the point w.p. 1/degree
				int d = exact_degree(iq, s);//Degree[s];
				valid = (rand() % d == 0);
			}
			else if (cd==3){//our algorithm rejects the point w.p. 1/degree'
				int dd = approximate_degree(iq,s);
				valid = (rand() % dd == 0);
			}
			if (valid){//if the point is not rejected then a sample is successfully retrieved
				snum--;
				if (mp.find(s)!=mp.end())//update the distribution
					mp[s] = mp[s]+1;
				else
					mp[s] = 1;
			} else {
				TotalRejections++;
			}
		}
	}

	double L1=0;
	for (map<int,int>::iterator it = mp.begin();it!=mp.end() ; ++it)//compute the L_1 distance of the distribution from the uniform niform
		L1 += abs((*it).second - 100);
	avgRes[cd]+=L1/mp.size();//add the solution to our averaging count for the specific algorithm denoted by cd
	TotalTestNum++;
	return ;
}


/*Given a query, computes its neighborhood size and the exact degree of the points in the buckets.
 */
void Query(int qq){
	cout << "in query " << qq << endl;
	Degree.clear();
	for (int l=0 ; l<L ; ++l){//computing the neighborhood and degrees
		set <pair<HP,int>>::iterator ed = buckets[l].lower_bound(pair<HP,int>(hQuery[l][qq],maxn));
		for (set <pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hQuery[l][qq],-1)); it!=ed ; ++it){
			if (ComputeDist(qq,(*it).second)<R){
				if (Degree.find((*it).second)==Degree.end())
					Degree[(*it).second] = 1;
				else
					Degree[(*it).second] = Degree[(*it).second]+1;
			}
		}
	}
	int SAMPLES = 100 * Degree.size(); //generate 100*m samples where m is the size of the neighborhood

	int avg_degree = 0;
	int max_degree = 0;

	for (auto &p: Degree) {
		avg_degree += p.second;
		max_degree = std::max(p.second, max_degree);
	}

	TotalCloseColliding += avg_degree / (double) Degree.size();
	TotalMaxDegree += max_degree;

	for (int i=0 ; i<RUNS ; ++i) {
		for (int j=0 ; j<cases ; ++j) {//for each of the four algorithms
				auto start = timer.now();

				TotalNonNear = 0;
				TotalRejections = 0;

				RandomSample(j, qq, SAMPLES);
				if (SAMPLES > 0) { 
					nonNear[j] += TotalNonNear / SAMPLES;
					rejections[j] += TotalRejections / SAMPLES;
				}

				auto query_time_in_s =  (timer.now() - start).count() / 1e9;
				if (times.find(j) == times.end()) {
					times.insert(make_pair(j, vector<double>()));
				}
				times.find(j)->second.push_back(query_time_in_s);
			}
		}
	return;
}

int main(int argc, char** argv){
	Initialize(argc, argv);//initializing the parameters
	ReadInput();//reading the data points and queries
	GenerateHashFunctions();//reading randomly generated hash functions
	ComputeHashes();// computing hash value of the points
	BuildLSH();//constructing the LSH buckets
	ComputeRetrievalRate(); // this is used for tuning the parameters of LSH only

	memset(avgRes,0,sizeof(avgRes));//initialize the results to 0
	memset(nonNear,0,sizeof(nonNear));//initialize the results to 0
	memset(rejections,0,sizeof(rejections));//initialize the results to 0
	TotalTestNum=0;//initialize to 0
	rng = new deann::FastRng(L);
	for (int q=0; q<Q ; ++q)//run it for the first Q queries
		Query(q);


	// generate nice result file name 
	auto slash_pos = dataset_fn.find_last_of("/");
	auto suffix_pos = dataset_fn.find_last_of("."); 
	stringstream ss; 
	ss << dataset_fn.substr(slash_pos + 1, suffix_pos - slash_pos - 1) << 
		"_(k=" << k << ", L=" << L << ", w=" << w << ")_results.out"; 
	auto expfile_fn = ss.str();

	std::cout << "Writing result file to " << expfile_fn << std::endl;

	ofstream fout(expfile_fn);//outputting the result
	TotalTestNum/=cases;
	for (int i=0 ; i<cases ; ++i) {
		auto query_times = times.find(i)->second;
		auto avgTime = accumulate(query_times.begin(), query_times.end(), 0.0) / query_times.size();
		fout << i << ", " << avgRes[i]/(200*TotalTestNum) << ", " << avgTime << ", " << nonNear[i] / (double) TotalTestNum  << ", " << rejections[i] / (double) TotalTestNum << 
		", " << TotalCloseColliding / (double) Q << ", " << TotalMaxDegree / (double) Q << endl; // we are dividing by 200 (2 is for the statistical distance and 100 is because we used 100*m samples per query.
	}
	fout.close();

	for (int i = 0; i < maxn; i++) 
		delete[] pnt[i];
	for (int i = 0; i < maxQ; i++) 
		delete[] query[i]; 
	for (int i = 0; i < maxL; i++) 
		for (int j = 0; j < maxk; j++) 
			delete[] hfv[i][j];

	return 0;
}
