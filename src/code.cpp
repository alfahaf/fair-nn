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
using namespace std;


const int dim = 784;//dimension
const int maxn = 10001;//maximum number of points in the data set
const int maxQ = 101;//maximum number of queries asked
const int maxL = 501;//maximum value of L considered
const int maxk = 51;//maximum value of k considered
int n , k , L, w , Q;
double R;//radius of near neighbor
double pnt[maxn][dim];//data set points
double query[maxQ][dim];//query points
double hfv[maxL][maxk][dim];//random direction vectors used for each of the (k*L) hash functions
double hfb[maxL][maxk];//random bias for shifting the 1-dimensional grid used for each of the (k*L) hash functions

void Initialize(){
	cout << "initializing ..." << endl;
	memset(pnt,0,sizeof pnt);
	n = 10000;
	k = 15;
	L = 100;//sqrt(n);
	R = 255*5;
	w= 4000;
	Q = 100;
	cout <<"Params: L is " << L << " k is " << k << " w is " << w << " R is " << R << " n is " << n << endl;
	srand(time(0));
	return;
}

void ReadInput(){
	cout << "Reading input ..." << endl;
	ifstream fin("mnist255.csv");//reading the data set points
	for (int i=0 ; i<n ; ++i)
		for (int j=0 ; j<dim ; ++j){
			char c;
			fin >> pnt[i][j];
			if (j<dim-1)
				fin >> c;
		}
	fin.close();
	fin.open("test255.csv");//reading the query points
	for (int i=0 ; i<Q ; ++i)
		for (int j=0 ; j<dim ; ++j){
			char c;
			fin >> query[i][j];
			if (j<dim-1)
				fin >> c;
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
/*void ComputeRetrievalRate(){
	int avgN =0;
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
	cout << "average False Negative is " << (avgFN*100)/close << "\%" << endl;
	return;
}
*/


int sizes[maxL];// each time we get a query, we compute the sizes of the buckets corresponding to the query.
int sumsize;//and we compute the sum of sizes[0] ...sizes[(-1] of the sizes in the sumsize variable to be able to generate a random value in that range


//This function returns a random bucket according to its size and then a random point inside the bucket.
//This is implemented by generating a random number from 1 to sumsize and then finding the point corresponding to this number
int single_sample_weighted(int iq){
	int r = rand()%sumsize;
	r++;
	int l=0;
	while (r>sizes[l]){
		r-=sizes[l];
		l++;
	}
	set<pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],-1));
	while (true){
		r--;
		if (r==0)
			return (*it).second;
		it++;
	}
	return -1;
}

//This function returns a random bucket uniformly at random and then a random point inside the bucket.
//This is implemented by generating a random number from 1 to L and then a random point inside the bucket
int single_sample_uniform(int iq){
	int l=-1;//index of the bucket
	for (l=(rand()%L) ; sizes[l]==0 ; l=(rand()%L));//avoid sampling an empty bucket
	int r = rand()%sizes[l];//index of the point in the bucket
	set<pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],-1));
	while (r){
		r--;
		it++;
	}
	return (*it).second;
}

//This function approximates the degree of a data set point (with index ip) among the buckets containing the query (with index iq)
//This is implemented by iteratively probing a random bucket and checking if it contains the point. 
//If the we see the point for the first time at iteration i, then we report L/i as an approximation of the degree
int approximate_degree(int iq , int ip){
	int num=0;
	while (num<L){
		num++;
		int l = rand()%L;
		if (hQuery[l][iq]==hPoint[l][ip])
			break;
	}
	return L/num;
}


map <int,int> Degree;//this variable is used to compute the exact degrees of the points in the bucket correnponding to the current query. This is used for comparison with the optimal algorithm. Degree[ip] shows the exact degree of the point with index ip among the buckets of the current query.

const int cases = 4;//This corresponds to the four algorithms we are implementing 
//0 is the Uniform/Uniform algorithm, 1 is the Weighted/Uniform algorithm, 2 is the optimal algorithm, 3 is our proposed algorithm

double avgRes[cases];//This variable keeps the average results for each algorithm.
int TotalTestNum=0;// This variable shows the number of times we run the algorithm for different queries.


/* this algorithm receives a query and draws snum number of samples according to the 
 * algorithm with the code cd (ranging from 0 to 3 as denoted above) 
 */
void RandomSample(int cd , int iq, int snum){
	if (snum==0)
		return;

	//computing the parameters sizes and sumsize of the buckets so that we dont have to compute it each time we draw a sample
	for (int l=0 ; l<L ; ++l){
		set<pair<HP,int>>::iterator it = buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],-1));
		int count=0;
		while (it!= buckets[l].lower_bound(pair<HP,int>(hQuery[l][iq],maxn))){
			it++;
			count++;
		}
		sizes[l] = count;
		if (l==0)
			sumsize = sizes[l];
		else
			sumsize += sizes[l];
	}

	map<int,int> mp;//This shows the distribution over the points in the neighborhood of the query.
	
	while (snum){//generating the samples
		int s=-1;
		if (cd==0)
			s = single_sample_uniform(iq);
		else
			s = single_sample_weighted(iq);
		if (ComputeDist(iq,s)>R)//reject the points outside neighborhood
			continue;
		bool valid = false;//shows if we should keep the sample or reject it.
		if (cd==0 || cd==1)//the uniform/uniform and weighted/uniform algorithms will always report the sampled point
			valid = true;
		else if (cd==2){//the optimal algorithm rejects the point w.p. 1/degree
			int d = Degree[s];
			valid = (rand()*d<RAND_MAX);
		}
		else if (cd==3){//our algorithm rejects the point w.p. 1/degree'
			int dd = approximate_degree(iq,s);
			valid = (rand()*dd<RAND_MAX);
		}
		if (valid){//if the point is not rejected then a sample is successfully retrieved
			snum--;
			if (mp.find(s)!=mp.end())//update the distribution
				mp[s] = mp[s]+1;
			else
				mp[s] = 1;
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
	for (int i=0 ; i<10 ; ++i)//repeat each test 10 times
		for (int j=0 ; j<cases ; ++j)//for each of the four algorithms
			RandomSample(j, qq,100*Degree.size());//generate 100*m samples where m is the size of the neighborhood
	return;
}

int main(){
	Initialize();//initializing the parameters
	ReadInput();//reading the data points and queries
	ReadHashFunctions();//reading randomly generated hash functions
	ComputeHashes();// computing hash value of the points
	BuildLSH();//constructing the LSH buckets
//	ComputeRetrievalRate(); // this is used for tuning the parameters of LSH only

	memset(avgRes,0,sizeof(avgRes));//initialize the results to 0
	TotalTestNum=0;//initialize to 0
	for (int q=0 ; q<Q ; ++q)//run it for the first Q queries
		Query(q);

	ofstream fout("results.out");//outputting the result
	TotalTestNum/=cases;
	for (int i=0 ; i<cases ; ++i)
		fout << i << ",\t" << avgRes[i]/(200*TotalTestNum) <<  endl; // we are dividing by 200 (2 is for the statistical distance and 100 is because we used 100*m samples per query.
	fout.close();
	return 0;
}
