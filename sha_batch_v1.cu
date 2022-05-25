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

__device__ void f_function(union INTER iState, union INTER* State){
    uint64_t CrossPlane[5];
    uint64_t D;

    for(int round = 0; round<rounds; round++){
            //Omega
            for(int i = 0; i<5; i++){
                CrossPlane[i] = state._64[i] ^ state._64[i + 5] ^ state._64[i + 10] ^ state._64[i + 15] ^ state._64[i + 20];
            }

            for(int i = 0; i<5; i++){
                D = CrossPlane[i==0?4:(i-1)] ^ ROT(CrossPlane[i==4?0:(i+1)], 1);
                for(int y = 0; y<5; y++)
                    {state._64[i + y*5] ^= D;}
            }

            int indx = 0;
            //Rho and Pi
            for(int i = 0; i<25; i++){
                indx = pi[i];
                iState._64[i] = ROT(state._64[indx], rho[indx]);
            }

            //Chi
            for(int y = 0; y<25; y+=5){
                for(int x = 0; x<5; x++)
                {
                    state._64[x+y] = iState._64[x+y] ^ (~iState._64[(x+1)%5 +y] & iState._64[(x+2)%5 +y]);
                }
            }
            state._64[0] ^= iota[round];
        }
}

std::string readFile(const char* const szFilename)
{
    std::ifstream in(szFilename, std::ios::in | std::ios::binary);
    std::ostringstream contents;
    contents << in.rdbuf();
    return contents.str();
}

__global__ void Keccak(string_t input, int size, union INTER* State){
    union INTER iState; 
    memset(State, 0, 200);
    //for (int i = 0; i < 300; i++) // DE 2 HIER
    while(size>0){
        if(size<rateBytes){    
            for(int i=0; i<size; i++)
                state._8[i] ^= input[i];
            state._8[size] ^= delim_begin;//padding
            state._8[rateBytes - 1] ^= delim_end;
        }
        else{
            for(int i=0; i<rateLanes; i++){
                state._64[i] ^= ((uint64_t*)input)[i];
            }   
            input += rateBytes;
        }
        
        f_function(iState, &state);
        size -= rateBytes;
    }
}

std::ofstream myfile;

void hostr(const char** path, const int times=1){
    cudaStream_t streamers[times];
    int sze[times];
    std::string message2[times];
    unsigned char output[times][32] = {{""}};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    char* input_d[times];
    union INTER* State_d[times];
    
    for(int tm = 0; tm < times; tm++){
        message2[tm] = readFile(path[tm]);
        sze[tm] = message2[tm].length();
        cudaMalloc((void**)&(input_d[tm]), (sze[tm]) * sizeof(char));
        cudaMalloc((void**)&(State_d[tm]), sizeof(union INTER));
    }

    // //call device kernal
    for(int tm = 0; tm < times; tm++){
        cudaStreamCreate(&streamers[tm]);
        cudaMemcpyAsync(input_d, message2[tm].c_str(), (sze[tm]) * sizeof(char), cudaMemcpyHostToDevice, streamers[tm]);
        Keccak<<<1,1,0,streamers[tm]>>>(input_d[tm], sze[tm], State_d[tm]);
        cudaMemcpyAsync(output[tm], State_d[tm], 32 * sizeof(char), cudaMemcpyDeviceToHost, streamers[tm]);
    }
       
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("\n%d - ", times);
    std::cout << "elapsed time: " << milliseconds/1000 << "s\n";
    if(LOGG)myfile << times << "; " << milliseconds/1000 << "\n" ;

    for(int tm = 0; tm < times; tm++){
        printf("\n     %s\t-\t", path[tm]);
        for (int i = 0; i < 32; i++)
            printf("%02x", output[tm][i]);
    }
    for(int tm = 0; tm < times; tm++){
        cudaFree(input_d[tm]);
        cudaFree(State_d[tm]);
    }
    printf("\n\n");
}

int main( int argc, char *argv[] ){
    if(LOGG)myfile.open ("buf.csv");
    const char* paths[argc-1];
    for(int pt = 0; pt<argc-1; pt++)
        paths[pt] = argv[pt+1];
    hostr(paths,argc-1);
    if(LOGG)myfile << "\n" ;
    if(LOGG)myfile.close();
}