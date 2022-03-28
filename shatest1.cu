typedef unsigned long long int uint64_t;
typedef unsigned char uint8_t;

#define message "The R points to resource, the id that specific part of the resources created. use of val because we don't want this object to be changed (constant value). Instead of using loops to see what everything is doing we make use of EventHandlers that will trigger when, what you defined, happens. This will then redirect them to a predefined function. This is written as bellow (e.g. button clicking)"

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
  { 0,  1,  62, 28, 27, 
    36, 44, 6,  55, 20,
    3,  10, 43, 25,	39,
    41,	45,	15, 21,	8,
    18,	2,	61, 56,	14};

__device__ __constant__ uint8_t pi[25] =
  { 4,  5,  11, 17, 23,
    2,  8,  14, 15, 21,
    0,  6,  12, 18, 24,
    3,  9,  10, 16, 22,
    1,  7,  13, 19, 20};

__device__ __constant__ uint8_t rho[25] =
  { 0,  1,  62, 28, 27, 
    36, 44, 6,  55, 20,
    3,  10, 43, 25,	39,
    41,	45,	15, 21,	8,
    18,	2,	61, 56,	14};

__device__ __constant__ uint8_t pi[25] =
  { 0, 6, 12, 18, 24, 
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
                for(int y = 0; y<=4; y++)
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

__global__ void Keccak(string_t input, int size, unsigned char* output){
    
    union INTER iState, State;

    memset(&State, 0, 200);

    while(size>0){
        if(size<rateBytes){
            for(int i=0; i<size; i++)
                State._8[i] ^= input[i];
            State._8[size] ^= delim_begin;//padding
            State._8[rateBytes - 1] ^= delim_end;
        }
        else{
            for(int i=0; i<rateLanes; i++)
                State._64[i] ^= ((uint64_t*)input)[i];
            input += rateBytes;
        }
        f_function(iState, &State);
        
        size -= rateBytes;
    }
    cudaMemcpy(output, &State._8, 32, cudaMemcpyDeviceToHost);
}

void main(){
    unsigned char output[32] = {""};
    string_t input = (char*)calloc(totalBytes, sizeof(char));

    //copy input to device
    string_t *input_d;
    cudaMalloc((void**)&input_d, sizeof(message)-1);
    cudaMemcpy(input_d, &input, sizeof(message)-1, cudaMemcpyHostToDevice);

    //call device kernal
    Keccak<<<1,1>>>(input, sizeof(message)-1, output);
})