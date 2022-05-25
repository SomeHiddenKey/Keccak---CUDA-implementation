#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <sstream> 
#include <chrono>
#include <thread> 

#define LOGG false

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

std::string readFile(const char* const szFilename)
{
    std::ifstream in(szFilename, std::ios::in | std::ios::binary);
    std::ostringstream contents;
    contents << in.rdbuf();
    return contents.str();
}

__global__ void Keccak(string_t input, int size, union INTER* State){
    union INTER iState, jState; 
    memset(&iState, 0, 200);
    //for (int i = 0; i < 300; i++) // DE 2 HIER
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
        
    uint64_t CrossPlane[5];
    uint64_t D;

    for(int round = 0; round<rounds; round++){
            //output-negative-batting-curry-reassign-headwear

            for(int i = 0; i<5; i++){
                CrossPlane[i] = iState._64[i] ^ iState._64[i + 5] ^ iState._64[i + 10] ^ iState._64[i + 15] ^ iState._64[i + 20];
            }

            for(int i = 0; i<5; i++){
                D = CrossPlane[i==0?4:(i-1)] ^ ROT(CrossPlane[i==4?0:(i+1)], 1);
                for(int y = 0; y<5; y++)
                    {iState._64[i + y*5] ^= D;}
            }

            int indx = 0;
            //Rho and Pi
            for(int i = 0; i<25; i++){
                indx = pi[i];
                jState._64[i] = ROT(iState._64[indx], rho[indx]);
            }

            //Chi
            for(int y = 0; y<25; y+=5){
                for(int x = 0; x<5; x++)
                {
                    iState._64[x+y] = jState._64[x+y] ^ (~jState._64[(x+1)%5 +y] & jState._64[(x+2)%5 +y]);
                }
            }
            iState._64[0] ^= iota[round];
        }
        size -= rateBytes;
    }
    for(int j=0; j<32; j++)
        state._8[j] = iState._8[j];
}

std::ofstream myfile;

void hostr(const char* path, int times=1){
    for(int tm = 0; tm<times; tm++){
        std::string message2 = readFile(path);
        int sze = message2.length();
        unsigned char output[32] = {""};

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // cudaEventRecord(start);

        char* input_d;
        cudaMalloc((void**)&input_d, (sze) * sizeof(char));

        union INTER* State_d;
        cudaMalloc((void**)&State_d, sizeof(union INTER));

        cudaMemcpyAsync(input_d, message2.c_str(), (sze) * sizeof(char), cudaMemcpyHostToDevice);  

        cudaEventRecord(start);

        // //call device kernal
        Keccak<<<1,1>>>(input_d, sze, State_d);
        cudaDeviceSynchronize();
        cudaMemcpyAsync(output, State_d, 32 * sizeof(char), cudaMemcpyDeviceToHost); 

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("%s - ", path);
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
    if(LOGG)myfile.open ("1t.csv");
    for(int pt = 1; pt<argc; pt++)
        hostr(argv[pt],1);
    if(LOGG)myfile << "\n" ;
    if(LOGG)myfile.close();
}