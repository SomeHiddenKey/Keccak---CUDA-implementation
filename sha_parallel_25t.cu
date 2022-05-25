#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <sstream> 
#include <chrono>
#include <thread> 

#define LOGG true

typedef unsigned char uint8_t;

typedef char* string_t;

#define total 1600
#define rate 1088

#define rounds 24

#define totalBytes total/8
#define rateBytes rate/8
#define rateLanes rate/64

#define delim_begin 0x06
#define delim_end 0x80

#define ROT(a, offset) ((((uint64_t)a) << offset) ^ (((uint64_t)a) >> (64-offset))) //credit
#define state (*State)
#define lane ((uint64_t*)state._8)

#define CROSSPLANE(result,x, array) result[x] = (array[x] ^ array[x + 5] ^ array[x + 10] ^ array[x + 15] ^ array[x + 20])
#define OMEGA(cross, result, m, i) result[i] ^= (cross[m==0?4:(m-1)] ^ ROT(cross[m==4?0:(m+1)], 1))
#define RHOPI(result, array, index, i,rho) \
{ index = pi[i];  \
result[i] = ROT(array[indx], rho[indx]); \
}
#define CHI(result, array, i, z) result[i] = array[i] ^ (~array[(i+1)%5 +z*5] & array[(i+2)%5 +z*5]);
#define IOTA(result, iota, round) result[0] ^= iota[round];

__device__ __constant__ uint8_t rho[25] =
    {0, 1, 62, 28, 27,
     36, 44, 6, 55, 20,
     3, 10, 43, 25, 39,
     41, 45, 15, 21, 8,
     18, 2, 61, 56, 14};

__device__ __constant__ uint8_t pi[25] =
    {0, 6, 12, 18, 24,
     3, 9, 10, 16, 22,
     1, 7, 13, 19, 20,
     4, 5, 11, 17, 23,
     2, 8, 14, 15, 21};

__device__ __constant__ uint64_t iota[24] =
  {
    0x0000000000000001UL, 0x0000000000008082UL,0x800000000000808aUL, 0x8000000080008000UL,
    0x000000000000808bUL, 0x0000000080000001UL,0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008aUL, 0x0000000000000088UL,0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL,0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,0x000000000000800aUL, 0x800000008000000aUL,
    0x8000000080008081UL, 0x8000000000008080UL,0x0000000080000001UL, 0x8000000080008008UL
};

union INTER{
    uint64_t _64[25];
    uint8_t _8[200];
};

std::string readFile(const char* const szFilename)
{
    std::ifstream in(szFilename, std::ios::in | std::ios::binary);
    std::ostringstream contents;
    contents << in.rdbuf();
    return contents.str();
}

__global__ void Keccak(string_t input, int size, union INTER* State){ 
    
    __shared__ union INTER iState; 
    uint8_t i = threadIdx.x;
    uint8_t i_25 = blockDim.x;
    for(int j=0; j<8; j++)
		iState._8[i_25*j+i] = 0;
    while(size>0){
        if(size<rateBytes){    
            for(int i=0; i<size; i++)
                iState._8[i] ^= input[i];
            iState._8[size] ^= delim_begin;//padding
            iState._8[rateBytes - 1] ^= delim_end;
        }
        else{
            for(int i=0; i<rateLanes; i++){
                iState._64[i] ^= ((uint64_t*)input)[i];
            }   
            input += rateBytes;
        }
        
    __shared__ uint64_t CrossPlane[5];

    uint8_t i = threadIdx.x;

    uint8_t m = i%5;
    uint8_t z = i/5;
    int indx = 0;

    for(int round = 0; round<rounds; round++){
        //Omega
        //CrossPlane[m] = state._64[m] ^ state._64[m + 5] ^ state._64[m + 10] ^ state._64[m + 15] ^ state._64[m + 20];
        CROSSPLANE(CrossPlane,m,iState._64);

        //state._64[i] ^= CrossPlane[m==0?4:(m-1)] ^ ROT(CrossPlane[m==4?0:(m+1)], 1);
        OMEGA(CrossPlane, iState._64, m, i);
        
        //Rho and Pi
        //indx = pi[i];
        //iState._64[i] = ROT(state._64[indx], rho[indx]);
        RHOPI(iState._64, iState._64, indx, i, rho);
        
        //Chi
        //state._64[i] = iState._64[i] ^ (~iState._64[(i+1)%5 +z*5] & iState._64[(i+2)%5 +z*5]);
        CHI(iState._64, iState._64, i, z);

        //state._64[0] ^= iota[round];
        IOTA(iState._64, iota, round);
    }
        size -= rateBytes;
    }
    for(int j=0; j<32; j++)
        state._8[j] = iState._8[j];
}

std::ofstream myfile;

void hostr(const char* path, int times=8){
    for(int tm = 0; tm<times; tm++){
        std::string message2 = readFile(path);
        int sze = message2.length();
        unsigned char output[32] = {""};

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // cudaEventRecord(start); //withmalloc

        char* input_d;
        cudaMalloc((void**)&input_d, (sze) * sizeof(char));
        cudaMemcpy(input_d, message2.c_str(), (sze) * sizeof(char), cudaMemcpyHostToDevice);  

        union INTER* State_d;
        cudaMalloc((void**)&State_d, sizeof(union INTER));

        cudaEventRecord(start); //nomalloc

        Keccak<<<1,25>>>(input_d, sze, State_d);

        cudaMemcpy(output, State_d, 32 * sizeof(char), cudaMemcpyDeviceToHost); 

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf(" %s - ", path);
        std::cout << "elapsed time: " << milliseconds/1000 << "s\n";
        if(LOGG)myfile << path << "; " << milliseconds/1000 << "\n" ;        

        for (int i = 0; i < 32; i++)
            printf("%02x", output[i]);
        
        cudaFree(input_d);
        cudaFree(State_d);
        printf("\n\n");
    }
}

int main( int argc, char *argv[] ){
    if(LOGG)myfile.open ("25t.csv");
    for(int pt = 1; pt<argc; pt++)
        hostr(argv[pt],1);
    if(LOGG)myfile << "\n" ;
    if(LOGG)myfile.close();
}