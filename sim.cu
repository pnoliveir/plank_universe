#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define TICKS 1
#define TOTAL_PLANKS 2000000000
#define MAX_GPU_MEMORY 16000000000ULL // 16GB in bytes for RTX 4060 Ti
#define PLANK_SIZE sizeof(Plank) // ~84 bytes
#define SEED_SIZE sizeof(unsigned int) // 4 bytes
#define MAX_GPU_PLANKS ((MAX_GPU_MEMORY - (MAX_GPU_MEMORY / 20)) / (PLANK_SIZE + SEED_SIZE)) // ~184M, 5% overhead
#define GROWTH_FACTOR 2.5f
#define MAX_BATCH_INPUT ((int)(MAX_GPU_PLANKS / GROWTH_FACTOR)) // ~73M
#define BLOCK_SIZE 256
#define MAX_NEIGHBORS 6
#define min(a,b) ((a) < (b) ? (a) : (b))

typedef struct {
    float weight;
    int type;
} Boson;

typedef struct {
    float energy;
    int particle_type;
    int tick_interval;
    int id;
    int neighbors[MAX_NEIGHBORS];
    Boson bosons[MAX_NEIGHBORS];
} Plank;

Plank* plank_array;
int plankCount;
FILE *tick_file, *hist_file, *map_file;

__host__ __device__ float rand_float(float min_val, float max_val, unsigned int* seed) {
    *seed = *seed * 1103515245 + 12345;
    float r = (float)(*seed & 0x7FFFFFFF) / 0x7FFFFFFF;
    return min_val + r * (max_val - min_val);
}

__device__ void init_plank_device(Plank* plank, int id, unsigned int* seed) {
    plank->energy = rand_float(2.0, 5.0, seed);
    plank->particle_type = 0;
    plank->tick_interval = 1;
    plank->id = id;
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        plank->neighbors[i] = -1;
        plank->bosons[i].weight = 0.0;
        plank->bosons[i].type = 0;
    }
}

__device__ float get_mass(Plank* plank) {
    float mass = 0.0;
    for (int d = 0; d < MAX_NEIGHBORS; d++) mass += plank->bosons[d].weight;
    return mass;
}

__global__ void plank_tick_kernel(Plank* planks, int* count, unsigned int* seeds, int tick, int global_offset, int total_planks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *count) return;

    Plank* plank = &planks[idx];
    unsigned int seed = seeds[idx];
    
    for (int t = 0; t < TICKS; t++) {
        if (t % plank->tick_interval == 0) {
            plank->energy += rand_float(-1.0, 1.0, &seed);
            if (plank->energy > 6.0) plank->energy = 6.0;
            if (plank->energy < 0.0) plank->energy = 0.1;
            float mass = get_mass(plank);
            float r = rand_float(0, 1, &seed);

            if (plank->energy > 0.2 && plank->energy < 6.0 && r < 0.82 && *count < MAX_GPU_PLANKS - 2) {
                int free_slots[2] = {-1, -1};
                int slot_count = 0;
                for (int d = 0; d < MAX_NEIGHBORS && slot_count < 2; d++) {
                    if (plank->neighbors[d] == -1) {
                        free_slots[slot_count++] = d;
                    }
                }
                for (int s = 0; s < slot_count; s++) {
                    int local_new_id = atomicAdd(count, 1);
                    if (local_new_id < MAX_GPU_PLANKS) {
                        int new_id = global_offset + local_new_id;
                        if (new_id < total_planks) {
                            Plank* new_plank = &planks[local_new_id];
                            init_plank_device(new_plank, new_id, &seed);
                            plank->neighbors[free_slots[s]] = new_id;
                            new_plank->neighbors[(free_slots[s] + 3) % MAX_NEIGHBORS] = plank->id;
                            plank->bosons[free_slots[s]].weight = rand_float(0.1, 2.0, &seed);
                            plank->bosons[free_slots[s]].type = 1;
                        }
                    }
                }
            } else if (mass > 2.0 && r < 0.27) {
                plank->tick_interval = 2;
            } else if (r < 0.05 && plank->particle_type == 0) {
                plank->particle_type = ((int)(rand_float(0, 4, &seed)) % 4) + 1;
                plank->tick_interval = 3;
                for (int d = 0; d < MAX_NEIGHBORS; d++) {
                    if (plank->bosons[d].weight > 0) 
                        plank->bosons[d].type = (plank->particle_type == 3) ? 1 : 2;
                }
            } else if (plank->particle_type && r < 0.01) {
                for (int d = 0; d < MAX_NEIGHBORS; d++) {
                    int n_idx = plank->neighbors[d];
                    if (n_idx != -1 && n_idx >= global_offset && n_idx < global_offset + MAX_GPU_PLANKS) {
                        int local_n_idx = n_idx - global_offset;
                        if (local_n_idx >= 0 && local_n_idx < MAX_GPU_PLANKS) {
                            if (planks[local_n_idx].energy + 1.0 < plank->energy) {
                                planks[local_n_idx].particle_type = plank->particle_type;
                                plank->particle_type = 0;
                                plank->bosons[d].type = (planks[local_n_idx].particle_type == 3) ? 1 : 2;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    seeds[idx] = seed;
}

void log_tick(int tick, Plank* planks, int count) {
    fprintf(tick_file, "%d,%d\n", tick, count);
    if (count > 0) {
        printf("Tick %2d - Plank 0: energy=%.3f, neighbors=[%d,%d,%d,%d,%d,%d], count=%9d, mem=%.1fGB\n", 
               tick, planks[0].energy, 
               planks[0].neighbors[0], planks[0].neighbors[1],
               planks[0].neighbors[2], planks[0].neighbors[3],
               planks[0].neighbors[4], planks[0].neighbors[5], count,
               (float)(count * sizeof(Plank)) / (1024 * 1024 * 1024));
    }
    for (int i = 0; i < count; i++) {
        fprintf(hist_file, "%d,%d,,%d,%f,%d\n", tick, planks[i].id, planks[i].id,
                planks[i].energy, planks[i].particle_type);
        for (int d = 0; d < MAX_NEIGHBORS; d++) {
            if (planks[i].bosons[d].weight > 0) {
                fprintf(map_file, "%d,%d,,%d,%f,%d\n", tick, planks[i].id, 
                        planks[i].neighbors[d], planks[i].bosons[d].weight, 
                        planks[i].bosons[d].type);
            }
        }
    }
    fflush(tick_file); fflush(hist_file); fflush(map_file);
}

void init_plank(Plank* plank, int id, unsigned int* seed) {
    plank->energy = rand_float(2.0, 5.0, seed);
    plank->particle_type = 0;
    plank->tick_interval = 1;
    plank->id = id;
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        plank->neighbors[i] = -1;
        plank->bosons[i].weight = 0.0;
        plank->bosons[i].type = 0;
    }
}

void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    unsigned int seed = time(NULL);
    srand(seed);

    tick_file = fopen("tick_summary.csv", "w");
    hist_file = fopen("cell_history.csv", "w");
    map_file = fopen("map_info.csv", "w");
    fprintf(tick_file, "tick,plank_count\n");
    fprintf(hist_file, "main_tick,plank_id,tick,new_plank_id,energy,particle_type\n");
    fprintf(map_file, "main_tick,plank_id,tick,new_plank_id,weight,conn_type\n");

    plank_array = (Plank*)malloc(TOTAL_PLANKS * sizeof(Plank));
    if (!plank_array) {
        printf("Host memory allocation failed\n");
        exit(1);
    }
    init_plank(&plank_array[0], 0, &seed);
    init_plank(&plank_array[1], 1, &seed); plank_array[0].neighbors[0] = 1; plank_array[1].neighbors[3] = 0;
    init_plank(&plank_array[2], 2, &seed); plank_array[0].neighbors[1] = 2; plank_array[2].neighbors[4] = 0;
    plankCount = 3;

    Plank* d_planks;
    int* d_count;
    unsigned int* d_seeds;
    cudaMalloc(&d_planks, MAX_GPU_PLANKS * sizeof(Plank));
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_seeds, MAX_GPU_PLANKS * sizeof(unsigned int));
    check_cuda_error("memory allocation");

    unsigned int* seeds = (unsigned int*)malloc(MAX_GPU_PLANKS * sizeof(unsigned int));
    if (!seeds) {
        printf("Seed memory allocation failed\n");
        exit(1);
    }
    for (int i = 0; i < MAX_GPU_PLANKS; i++) seeds[i] = seed + i;
    cudaMemcpy(d_seeds, seeds, MAX_GPU_PLANKS * sizeof(unsigned int), cudaMemcpyHostToDevice);

    size_t max_bytes = MAX_GPU_PLANKS * sizeof(Plank);
    printf("Max GPU planks per batch: %lld, Max input per batch: %d, Max bytes: %.1fGB, Total target: %d\n", 
           MAX_GPU_PLANKS, MAX_BATCH_INPUT, (float)max_bytes / (1024 * 1024 * 1024), TOTAL_PLANKS);

    for (int t = 0; t < 32 && plankCount < TOTAL_PLANKS; t++) {
        int start_count = plankCount;
        int num_active_batches = (start_count + MAX_BATCH_INPUT - 1) / MAX_BATCH_INPUT;
        int total_new_planks = 0;

        for (int batch = 0; batch < num_active_batches; batch++) {
            int batch_offset = batch * MAX_BATCH_INPUT;
            int current_batch_size = min(MAX_BATCH_INPUT, start_count - batch_offset);
            if (current_batch_size <= 0) break;

            cudaMemcpy(d_planks, plank_array + batch_offset, current_batch_size * sizeof(Plank), cudaMemcpyHostToDevice);
            cudaMemcpy(d_count, &current_batch_size, sizeof(int), cudaMemcpyHostToDevice);
            check_cuda_error("batch memory copy");

            int blocks = (current_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            plank_tick_kernel<<<blocks, BLOCK_SIZE>>>(d_planks, d_count, d_seeds, t, total_new_planks, TOTAL_PLANKS);
            cudaDeviceSynchronize();
            check_cuda_error("kernel launch");

            int new_count;
            cudaMemcpy(&new_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
            check_cuda_error("count copy back");

            int max_copy = min(new_count, min(MAX_GPU_PLANKS, TOTAL_PLANKS - total_new_planks));
            printf("Tick %d, Batch %d: batch_offset=%d, current_batch_size=%d, new_count=%d, max_copy=%d, total_new_planks=%d\n", 
                   t, batch, batch_offset, current_batch_size, new_count, max_copy, total_new_planks);
            if (max_copy > 0) {
                cudaMemcpy(plank_array + total_new_planks, d_planks, max_copy * sizeof(Plank), cudaMemcpyDeviceToHost);
                check_cuda_error("batch copy back");
            }
            total_new_planks += max_copy;
        }
        plankCount = total_new_planks;

        for (int i = 0; i < plankCount; i++) {
            for (int d = 0; d < MAX_NEIGHBORS; d++) {
                int n_id = plank_array[i].neighbors[d];
                if (n_id != -1 && (n_id >= TOTAL_PLANKS || n_id >= plankCount)) {
                    plank_array[i].neighbors[d] = -1;
                    plank_array[i].bosons[d].weight = 0.0;
                    plank_array[i].bosons[d].type = 0;
                }
            }
        }

        log_tick(t, plank_array, plankCount);
/*        if (plankCount > 10000000 && plankCount < TOTAL_PLANKS) {
            printf("Reached 10M at tick %d: %d planks, continuing...\n", t, plankCount);
        } */
        if (plankCount >= TOTAL_PLANKS) {
            printf("Reached target at tick %d: %d planks\n", t, plankCount);
            break;
        }
        sleep(1);
    }

    cudaFree(d_planks); cudaFree(d_count); cudaFree(d_seeds);
    free(plank_array); free(seeds);
    fclose(tick_file); fclose(hist_file); fclose(map_file);
    return 0;
}
