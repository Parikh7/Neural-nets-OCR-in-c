//
//  main.c
//  header file
//
//  Created by Parikh Oberoi on 21/01/18.
//  Copyright © 2018 Parikh Oberoi. All rights reserved.
//

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/*static inline void print_mat( const float *mat, int rows, int cols );
static inline void transpose_mat( const float *in, float *out, int rows, int cols );
static inline void mult_mat_vec( const float *mat, int rows, int cols,
                                const float *vec_in, float *vec_out );
static inline void randomise_mat( float *mat, int rows, int cols );
static inline void sigmoid( const float *vec_in, float *vec_out, int elements );
static inline void colrow_vec_mult( const float *col_vec, const float *row_vec,
                                   int rows, int cols, float *matrix_out );
*/
// my matrices, of any dimensions, are treated as 1D arrays and are arranged in
// column-major memory order. functions use column-major mathematical conventions

void read_csv( const char* file_name, int nentries, float labels[nentries], float pixels[nentries][784] );

// because i so hate strtok i gave my only func
int get_next_csv_int( const char* str, int* curr_idx ) {
    int len = (int)strlen( str );
    int v = -1;
    for (int i = *curr_idx; i < len - 1; i++ ) {
        if ( str[i] == ',' ) {
            sscanf( &str[i], ",%i", &v);
            //printf("found %i at index %i where curr_idx = %i\n", v, i, *curr_idx);
            *curr_idx = i + 1;
            return v;
        }
    }
    *curr_idx = -1;
    return v;
}

void read_csv( const char* file_name, int nentries, float labels[nentries], float pixels[nentries][784] ) {
    assert( file_name );
    assert( labels );
    assert( pixels );
    
    int line_count = 0;
    
    // 28x28 images = 784
    // label number,784 values
    
    FILE* f = fopen( file_name, "r" );
    assert( f );
    
    char line[2048];
    line[0] = '\0';
    while ( fgets ( line, 2048, f ) ) {
        if (nentries == line_count) {
            break;
        }
        int label = -1;
        sscanf( line, "%i,", &label);
        labels[line_count] = (float)label;
        //printf("label = %i\n", label);
        int curr_idx = 0;
        int pixel_count = 0;
        while (curr_idx >= 0 ) {
            int val = get_next_csv_int( line, &curr_idx );
            if (curr_idx < 0 || pixel_count >= (28 * 28)) {
                break;
            }
            //printf("pixels[%i][%i]=%i\n", line_count, pixel_count,val);
            float fval = (float)val / 255.0f;
            fval = fval * 0.99f + 0.01f;
            pixels[line_count][pixel_count] = fval;
            pixel_count++;
        }
        line_count++;
    }
    
    fclose( f );
    printf("line count %i\n", line_count);
}



static inline void print_mat( const float *mat, int rows, int cols ) {
    assert( mat );
    
    // printf( "\n" );
    for ( int r = 0; r < rows; r++ ) {
        printf( "| " );
        for ( int c = 0; c < cols; c++ ) {
            printf( "%f ", mat[c * rows + r] );
        }
        printf( "|\n" );
    }
}

// notes:
// * MxN in results in NxM out
// * can avoid this func by just accessing matrix elements in different order
static  inline void transpose_mat( const float *iq, float *ou, int rows,int cols )
{
    assert( iq );
    assert( ou);
    
    for ( int m = 0; m < rows; m++ ) {
        for ( int n = 0; n < cols; n++ ) {
            int in_idx = m + n * rows;
            int out_idx = n + m * cols;
            ou[out_idx] = iq[in_idx];
        }
    }
}

// notes:
// * input vector must have elements == matrix COLS
// * output vector must have elements == matrix ROWS
// note: can ~speed up by switching loops around. i went with formula repro 1st
static  inline void mult_mat_vec( const float *mat, int rows, int cols,const float *vec_in, float *vec_out ) {
    assert( mat );
    assert( vec_in );
    assert( vec_out );
    
    // general formula is for ea matrix row mult ea col el in row w eac vec el
    // m[row 0, col 0] * v[0] + m[row 0][col 1] * v[1] ... m[row 0][col m] * v[n]
    // m[row 1, col 0] * v[0] + m[row 1][col 1] * v[1] ... m[row 1][col m] * v[n]
    // ...
    // m[row n, col 0] * v[0] + m[row n][col 1] * v[1] ... m[row n][col m] * v[n]
    //memset( vec_out, 0, sizeof( float ) * rows );
    for ( int r = 0; r < rows; r++ ) {
        vec_out[r] = 0.0f;
        for ( int c = 0; c < cols; c++ ) {
            int mat_idx = r + c * rows;
            float m = mat[mat_idx];
            float vval = vec_in[c];
            vec_out[r] += ( m * vval );
        }
    }
}

// randomises matrix weight values between 0.1 and 0.9
// note: uses rand(). remember to srand()
static inline void randomise_mat( float *mat, int rows, int cols ) {
    assert( mat );
    
    for ( int i = 0; i < rows * cols; i++ ) {
        float val = (float)rand() / RAND_MAX;
        // val = val * 0.8 + 0.1;
        val -= 0.5f; // allow negative range -0.5 to 0.5
        mat[i] = val;
    }
}

static inline void sigmoid( const float *vec_in, float *vec_out, int elements ) {
    assert( vec_in );
    assert( vec_out );
    
    for ( int i = 0; i < elements; i++ ) {
        vec_out[i] = 1.0f / ( 1.0f + powf( M_E, -vec_in[i] ) );
    }
}

// note: final rows = rows in first, final cols = cols in second
//    |1|           | 1x2 1x3 1x4 |   | 2 3 4  |
//    |2| [2 3 4] = | 2x2 2x3 2x4 | = | 4 6 8  |
//    |3|           | 3x2 3x3 3x4 |   | 6 9 12 |
static inline void colrow_vec_mult( const float *col_vec, const float *row_vec,
                                   int rows, int cols, float *matrix_out ) {
    for ( int c = 0; c < cols; c++ ) {
        for ( int r = 0; r < rows; r++ ) {
            int mat_idx = r + c * rows;
            matrix_out[mat_idx] = col_vec[r] * row_vec[c];
        }
    }
}


//#define DEBUG_PRINT
//#define DRAW_OUT_IMAGES
//#define PRINT_TRAINING

typedef struct network_t {
    int ninputs;
    int nhiddens;
    int noutputs;
    int max_steps;
    float learning_rate;
    float *input_to_hidden_weights;
    float *input_to_hidden_delta_weights;
    float *hidden_to_output_weights;
    float *hidden_to_output_back_weights;
    float *hidden_to_output_delta_weights;
    float *inputs;
    float *hiddens;
    float *outputs;
    float *output_errors;
    float *hidden_errors;
    float *hiddens_deltas;
    float *outputs_deltas;
} network_t;

static network_t network;

void init_network( int ninputs, int nhiddens, int noutputs, float learning_rate,
                  int max_steps );
// note: In C99, you can provide the dimensions of the array before passing it
void train_network( int nsamples, int ninputs, int noutputs,
                   float inputs_list[nsamples][ninputs],
                   float targets_list[nsamples][noutputs] );
void query( const float *inputs );


int main() {
    int num_pixels = 28 * 28;
    int ninputs = num_pixels;
    int nhiddens = 100;
    int noutputs = 10;
    float learning_rate = 0.3f;
    int max_steps = 1; // i dont think they do any steps in book -- suspect this will cause overtraining
    int epochs = 10; // 1 to 10 went from ~80% to ~95% accuracy
    
    int num_lines = 2000; // more samples = more accurate.
    float labels[num_lines];
    float pixels[num_lines][num_pixels];
    read_csv( "mnist_train.csv", num_lines, labels, pixels );

    for ( int j = 0; j < num_lines; j++ ) { // write first image out to test it worked
        char tmp[128];
        sprintf( tmp, "img%i_label_%i.ppm", j, (int)labels[j] );
        FILE *f = fopen( tmp, "w" );
        assert( f );
        fprintf( f, "P3\n28 28\n255\n" );
        for ( int i = 0; i < num_pixels; i++ ) {
            int v =(int)( 255.0f - 255.0f * ( ( pixels[j][i] - 0.01f ) * ( 1.0f / 0.99f ) ) );
            fprintf( f, "%i %i %i\n", v, v, v );
        }
        fclose( f );
    }

    
    init_network( ninputs, nhiddens, noutputs, learning_rate, max_steps );
    
    for ( int epoch = 0; epoch < epochs; epoch++ ) { // training run
        // prepare targets in form       0   1   2    3   4   5 ... 9
        //                              [0.1 1.0 0.1 0.1 0.1 0.1...0.1]
        float targets_list[num_lines][noutputs];
        for ( int l = 0; l < num_lines; l++ ) {
            for ( int o = 0; o < noutputs; o++ ) {
                if ( (int)labels[l] == o ) {
                    targets_list[l][o] = 1.0f;
                } else {
                    targets_list[l][o] = 0.1f; // lowest useful signal
                }
            }
        }
        

        printf( "targets list:\n" );
        print_mat( targets_list[5], noutputs, 1 ); // '9'

        
        train_network( num_lines, ninputs, noutputs, pixels, targets_list );
    }
    
    // TODO load test set and do this
    
    {
        int num_lines = 100;
        float labels[num_lines];
        float pixels[num_lines][num_pixels];
        read_csv( "mnist_test.csv", num_lines, labels, pixels );
        int sum_correct = 0;
        for ( int i = 0; i < num_lines; i++ ) {
            query( pixels[i] );
            int maxi = 0;
            float maxf = network.outputs[0];
            for ( int j = 1; j < network.noutputs; j++ ) {
                if ( network.outputs[j] > maxf ) {
                    maxf = network.outputs[j];
                    maxi = j;
                }
            }
            printf( "queried. our answer %i (%f conf) - correct answer %i\n", maxi,
                   network.outputs[maxi], (int)labels[i] );
            if ( maxi == (int)labels[i] ) {
                sum_correct++;
            }

            print_mat( network.outputs, network.noutputs, 1 );

            //        printf( "getchar:\n" );
            //    getchar();
        }
        float accuracy = (float)sum_correct / (float)num_lines;
        printf( "total accuracy = %f\n", accuracy );
    }
    return 0;
}

void init_network( int ninputs, int nhiddens, int noutputs, float learning_rate,
                  int max_steps ) {
    printf( "initialising...\n" );
    printf( "math e constant = %.12f\n", M_E );
    unsigned int seed = (unsigned int)time( NULL );
    srand( seed );
    printf( "seed = %u\n", seed );
    
    network.ninputs = ninputs;
    network.nhiddens = nhiddens;
    network.noutputs = noutputs;
    network.max_steps = max_steps;
    network.learning_rate = learning_rate;
    
    { // program memory allocation
        size_t sz_a = sizeof( float ) * ninputs * nhiddens;
        size_t sz_b = sizeof( float ) * nhiddens * noutputs;
        size_t sz_inputs = sizeof( float ) * ninputs;
        size_t sz_hiddens = sizeof( float ) * nhiddens;
        size_t sz_outputs = sizeof( float ) * noutputs;
        // matrices between layers
        network.input_to_hidden_weights = (float *)malloc( sz_a );
        assert( network.input_to_hidden_weights );
        network.input_to_hidden_delta_weights = (float *)malloc( sz_a );
        assert( network.input_to_hidden_delta_weights );
        network.hidden_to_output_weights = (float *)malloc( sz_b );
        assert( network.hidden_to_output_weights );
        network.hidden_to_output_back_weights = (float *)malloc( sz_b );
        assert( network.hidden_to_output_back_weights );
        network.hidden_to_output_delta_weights = (float *)malloc( sz_b );
        assert( network.hidden_to_output_delta_weights );
        // vectors for each layer
        network.inputs = (float *)calloc( ninputs, sizeof( float ) );
        assert( network.inputs );
        network.hiddens = (float *)calloc( nhiddens, sizeof( float ) );
        assert( network.hiddens );
        network.hidden_errors = (float *)calloc( nhiddens, sizeof( float ) );
        assert( network.hidden_errors );
        network.outputs = (float *)calloc( noutputs, sizeof( float ) );
        assert( network.outputs );
        network.output_errors = (float *)calloc( noutputs, sizeof( float ) );
        assert( network.output_errors );
        network.hiddens_deltas = (float *)calloc( nhiddens, sizeof( float ) );
        network.outputs_deltas = (float *)calloc( noutputs, sizeof( float ) );
        size_t sz_total =
        sz_a * 2 + sz_b * 3 + sz_inputs + sz_hiddens * 3 + sz_outputs * 3;
        printf( "allocated %lu bytes (%lu kB) (%lu MB)\n", sz_total, sz_total / 1024,
               sz_total / ( 1024 * 1024 ) );

        printf( "  inputs %lu\n", sz_inputs );
        printf( "  hiddens %lu\n", sz_hiddens );
        printf( "  hidden errors %lu\n", sz_hiddens );
        printf( "  hidden deltas %lu\n", sz_hiddens );
        printf( "  outputs %lu\n", sz_outputs );
        printf( "  output errors %lu\n", sz_outputs );
        printf( "  output deltas %lu\n", sz_outputs );
        printf( "  weights input->hidden %lu\n", sz_a );
        printf( "  delta weights input->hidden %lu\n", sz_a );
        printf( "  weights hidden->outputs %lu\n", sz_b );
        printf( "  weights outputs->hidden %lu\n", sz_b );
        printf( "  delta weights hidden->outputs %lu\n", sz_b );

    }
    
    randomise_mat( network.input_to_hidden_weights, ninputs, nhiddens );
    randomise_mat( network.hidden_to_output_weights, nhiddens, noutputs );
}

void train_network( int nsamples, int ninputs, int noutputs,
                   float inputs_list[nsamples][ninputs],
                   float targets_list[nsamples][noutputs] ) {
    assert( inputs_list );
    assert( targets_list );
    
    printf( "training...\n" );
    
    
    // samples loop here
    for ( int curr_sample = 0; curr_sample < nsamples; curr_sample++ ) {
        
        // learning rate loop here somewhere -- probably vice versa?
        for ( int step = 0; step < network.max_steps; step++ ) {
            
            { // first part is same as query()
                memcpy( network.inputs, inputs_list[curr_sample],
                       network.ninputs * sizeof( float ) );
                // ANTON: i switched around row and cols vars here because valgrind told me
                // i was wrong
                mult_mat_vec( network.input_to_hidden_weights, network.nhiddens,
                             network.ninputs, network.inputs, network.hiddens );
                sigmoid( network.hiddens, network.hiddens, network.nhiddens );
                // ANTON: i switched around row and cols vars here because valgrind told me
                // i was wrong
                mult_mat_vec( network.hidden_to_output_weights, network.noutputs,
                             network.nhiddens, network.hiddens, network.outputs );
                sigmoid( network.outputs, network.outputs, network.noutputs );
            }
            { // second part compares result to desired result and back-propagates error
                for ( int i = 0; i < network.noutputs; i++ ) {
                    network.output_errors[i] =
                    targets_list[curr_sample][i] - network.outputs[i];
                    
                    // printf( "errors[%i] = %f\n", i, network.output_errors[i] );
                }
                
                // transpose the hidden->output matrix to go backwards
                transpose_mat( network.hidden_to_output_weights,
                              network.hidden_to_output_back_weights, network.nhiddens,
                              network.noutputs );
                // work out proportional errors for hidden layer too
                // ANTON: i switched around row and cols vars here because valgrind told me
                // i was wrong
                mult_mat_vec( network.hidden_to_output_back_weights, network.nhiddens,
                             network.noutputs, network.output_errors,
                             network.hidden_errors );
                
                { // adjust hidden->output weights
                    for ( int i = 0; i < network.noutputs; i++ ) {
                        network.outputs_deltas[i] = network.output_errors[i] *
                        network.outputs[i] *
                        ( 1.0f - network.outputs[i] );
                        //    printf( "output delta[%i] = %f\n", i, network.outputs_deltas[i] );
                    }
                    colrow_vec_mult( network.outputs_deltas, network.hiddens,
                                    network.noutputs, network.nhiddens,
                                    network.hidden_to_output_delta_weights );
                    
                    //    printf( "delta weights matrix:\n" );
                    //    print_mat( network.hidden_to_output_delta_weights, network.noutputs,
                    //                         network.nhiddens );
                    
                    for ( int i = 0; i < ( network.nhiddens * network.noutputs ); i++ ) {
                        network.hidden_to_output_delta_weights[i] *= network.learning_rate;
                        //    printf( "output weight %i before = %f\n", i,
                        //                    network.hidden_to_output_weights[i] );
                        network.hidden_to_output_weights[i] +=
                        network.hidden_to_output_delta_weights[i];
                        //    printf( "output weight %i after = %f\n", i,
                        //                    network.hidden_to_output_weights[i] );
                    }
                }
                
                { // adjust input->hidden weights
                    for ( int i = 0; i < network.nhiddens; i++ ) {
                        network.hiddens_deltas[i] = network.hidden_errors[i] *
                        network.hiddens[i] *
                        ( 1.0f - network.hiddens[i] );
                        //    printf( "hidden delta[%i] = %f\n", i, network.hiddens_deltas[i] );
                    }
                    colrow_vec_mult( network.hiddens_deltas, network.inputs, network.nhiddens,
                                    network.ninputs, network.input_to_hidden_delta_weights );
                    
                    // printf( "delta weights matrix:\n" );
                    //    print_mat( network.input_to_hidden_delta_weights, network.nhiddens,
                    //                     network.ninputs );
                    
                    for ( int i = 0; i < ( network.ninputs * network.nhiddens ); i++ ) {
                        network.input_to_hidden_delta_weights[i] *= network.learning_rate;
                        
                        //    printf( "output weight %i before = %f\n", i,
                        //                network.input_to_hidden_weights[i] );
                        network.input_to_hidden_weights[i] +=
                        network.input_to_hidden_delta_weights[i];
                        
                        //    printf( "output weight %i after = %f\n", i,
                        //                network.input_to_hidden_weights[i] );
                    }
                }
                
            } // end back-propagation
            

            if ( step % 100 == 0 ) {
                printf( "end of step %i\n", step );
                float error_sum = 0.0f;
                for ( int i = 0; i < network.noutputs; i++ ) {
                    printf( "output[%i] = %f target = %f\n", i, network.outputs[i],
                           targets_list[curr_sample][i] );
                    error_sum += ( targets_list[curr_sample][i] - network.outputs[i] );
                }
                printf( "error sum was %f\n", error_sum );
            }
        }
    } // end steps loop
    
   
}

// note: make sure inputs vector is populated first
void query( const float *inputs ) {
    assert( inputs );
    
    printf( "querying...\n" );
    memcpy( network.inputs, inputs, network.ninputs * sizeof( float ) );
    memset( network.outputs, 0, network.noutputs );
    
    // feed input -> hidden layer
    // ANTON reversed rows/cols here too
    mult_mat_vec( network.input_to_hidden_weights, network.nhiddens, network.ninputs,
                 network.inputs, network.hiddens );
    // apply sigmoid to dampen activation signal before output
    sigmoid( network.hiddens, network.hiddens, network.nhiddens );
    // feed hidden -> output layer
    // ANTON reversed rows/cols here too
    mult_mat_vec( network.hidden_to_output_weights, network.noutputs,
                 network.nhiddens, network.hiddens, network.outputs );
    // apply sigmoid to dampen activation signal before output
    sigmoid( network.outputs, network.outputs, network.noutputs );
}


